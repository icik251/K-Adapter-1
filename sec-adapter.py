"""
Pre-train SEC sentiment analysis Adapter
"""

from collections import defaultdict
import shutil
import sys
import os
import time

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from sklearn.metrics import f1_score
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from pytorch_transformers import BertModel, BasicTokenizer
from pytorch_transformers import AdamW, WarmupLinearSchedule

from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
    DistributedSampler,
)

from pytorch_transformers.tokenization_bert import BertTokenizer as BertTokenizerLocal
from transformers import BertTokenizer as BertTokenizerHugging

from examples.utils_sec import (
    output_modes,
    processors,
    convert_examples_to_features_sec_adapter,
    SECDataset,
)


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class PretrainedModel(nn.Module):
    def __init__(self, args):
        super(PretrainedModel, self).__init__()
        self.model = BertModel.from_pretrained(
            pretrained_model_name_or_path=args.finbert_path, output_hidden_states=True
        )
        self.config = self.model.config

        for p in self.parameters():
            p.requires_grad = False

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        start_id=None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        return outputs  # (loss), logits, (hidden_states), (attentions)


"""
Adapter model
"""

from pytorch_transformers.modeling_bert import BertEncoder


class Adapter(nn.Module):
    def __init__(self, args, adapter_config):
        super(Adapter, self).__init__()
        self.adapter_config = adapter_config
        self.args = args
        self.down_project = nn.Linear(
            self.adapter_config.project_hidden_size,
            self.adapter_config.adapter_size,
        )
        self.encoder = BertEncoder(self.adapter_config)
        self.up_project = nn.Linear(
            self.adapter_config.adapter_size, adapter_config.project_hidden_size
        )
        self.init_weights()

    def forward(self, hidden_states):
        # This is the core of the Adapter with the down projected, up projected layers
        down_projected = self.down_project(hidden_states)

        input_shape = down_projected.size()[:-1]
        attention_mask = torch.ones(input_shape, device=self.args.device)
        encoder_attention_mask = torch.ones(input_shape, device=self.args.device)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        head_mask = [None] * self.adapter_config.num_hidden_layers
        encoder_outputs = self.encoder(
            down_projected, attention_mask=extended_attention_mask, head_mask=head_mask
        )

        up_projected = self.up_project(encoder_outputs[0])
        return hidden_states + up_projected

    def init_weights(self):
        self.down_project.weight.data.normal_(
            mean=0.0, std=self.adapter_config.adapter_initializer_range
        )
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(
            mean=0.0, std=self.adapter_config.adapter_initializer_range
        )
        self.up_project.bias.data.zero_()


class AdapterEnsembleModel(nn.Module):
    def __init__(self, args, pretrained_model_config):
        super(AdapterEnsembleModel, self).__init__()
        self.config = pretrained_model_config
        self.args = args
        self.adapter_size = self.args.adapter_size

        class AdapterConfig:
            project_hidden_size: int = self.config.hidden_size
            hidden_act: str = "gelu"
            adapter_size: int = self.adapter_size  # 64
            adapter_initializer_range: float = 0.0002
            is_decoder: bool = False
            attention_probs_dropout_prob: float = 0.1
            hidden_dropout_prob: float = 0.1
            hidden_size: int = 768
            initializer_range: float = 0.02
            intermediate_size: int = 3072
            layer_norm_eps: float = 1e-05
            max_position_embeddings: int = 512
            num_attention_heads: int = 12
            num_hidden_layers: int = self.args.adapter_transformer_layers
            num_labels: int = 2
            output_attentions: bool = False
            output_hidden_states: bool = False
            torchscript: bool = False
            type_vocab_size: int = 2
            vocab_size: int = 30878

        self.adapter_skip_layers = self.args.adapter_skip_layers
        self.num_labels = args.sentiment_analysis_num_labels
        # self.config.output_hidden_states=True
        self.adapter_list = args.adapter_list
        # self.adapter_list =[int(i) for i in self.adapter_list]
        self.adapter_num = len(self.adapter_list)
        # self.adapter = Adapter(args, AdapterConfig)

        # here we create this module list that contains the adapterconfig for the adapter list of 0,11,23
        # so in the forward for loop we can train
        self.adapter = nn.ModuleList(
            [Adapter(args, AdapterConfig) for _ in range(self.adapter_num)]
        )

        self.com_dense = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.config.hidden_size, 1)

    def forward(
        self,
        pretrained_model_outputs,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        subj_special_start_id=None,
        obj_special_start_id=None,
    ):

        outputs = pretrained_model_outputs
        sequence_output = outputs[0]  # 12-th hidden layer
        # pooler_output = outputs[1]
        hidden_states = outputs[2]  # all hidden layers so we can take 0,5,11 later
        num = len(hidden_states)
        hidden_states_last = torch.zeros(sequence_output.size()).to(self.args.device)

        adapter_hidden_states = []
        adapter_hidden_states_count = 0
        for i, adapter_module in enumerate(self.adapter):
            # Sum the current hidden that comes out of the adapter with the hidden from the pre-trained BERT
            fusion_state = hidden_states[self.adapter_list[i]] + hidden_states_last
            hidden_states_last = adapter_module(fusion_state)
            adapter_hidden_states.append(hidden_states_last)
            adapter_hidden_states_count += 1
            if (
                self.adapter_skip_layers >= 1
            ):  # if adapter_skip_layers>=1, skip connection
                # If that happens and adapter_skip_layers == 3, we sum the last hidden with the pre-last from the adapter 
                if adapter_hidden_states_count % self.adapter_skip_layers == 0:
                    hidden_states_last = (
                        hidden_states_last
                        + adapter_hidden_states[
                            int(adapter_hidden_states_count / self.adapter_skip_layers)
                        ]
                    )

        ##### drop below parameters when doing downstream tasks
        com_features = self.com_dense(
            torch.cat([sequence_output, hidden_states_last], dim=2)
        )
        com_features_squeezed = com_features[:, 0, :]

        logits = self.out_proj(self.dropout(self.dense(com_features_squeezed)))

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.squeeze(1), labels)
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)

    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self
        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Saving model checkpoint to %s", save_directory)


def load_and_cache_examples(args, task, tokenizer, dataset_type, evaluate=False):
    # Modify for our dataset
    # dataset_type: train, dev, test
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            dataset_type,
            list(filter(None, args.finbert_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    # Remove "not" when finished with development
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = (
            processor.get_dev_examples(args.data_dir, dataset_type)
            if evaluate
            else processor.get_train_examples(args.data_dir, dataset_type)
        )
        label_list = processor.get_labels()
        features = convert_examples_to_features_sec_adapter(
            examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            output_mode,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )

    return dataset


def train(args, train_dataset, val_dataset, model, tokenizer):
    """Train the model"""
    pretrained_finbert_model = model[0]
    adapter_ensemble_model = model[1]

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir="runs/" + args.my_model_name)

    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size // args.gradient_accumulation_steps,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)

    """
        Optimizer and Scheduler for Adapter model
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in adapter_ensemble_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in adapter_ensemble_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=args.warmup_steps, t_total=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    logger.info("Try resume from checkpoint")
    if args.restore:
        if os.path.exists(os.path.join(args.output_dir, "global_step.bin")):
            logger.info("Load last checkpoint data")
            global_step = torch.load(os.path.join(args.output_dir, "global_step.bin"))
            global_step += 1
            start_epoch = int(global_step / len(train_dataloader)) - 1
            output_dir = os.path.join(
                args.output_dir, "checkpoint-{}".format(start_epoch)
            )
            logger.info("Load from output_dir {}".format(output_dir))

            optimizer.load_state_dict(
                torch.load(os.path.join(output_dir, "optimizer.bin"))
            )
            scheduler.load_state_dict(
                torch.load(os.path.join(output_dir, "scheduler.bin"))
            )
            adapter_ensemble_model.load_state_dict(
                torch.load(os.path.join(output_dir, "pytorch_model.bin"))
            )

            # global_step += 1
            # start_step = global_step - start_epoch * len(train_dataloader) - 1
            # Load the epoch that ended and continue from the next one
            start_epoch += 1
            logger.info(
                "Start from global_step={} epoch={}".format(global_step, start_epoch)
            )

            if args.local_rank in [-1, 0]:
                tb_writer = SummaryWriter(
                    log_dir="runs/" + args.my_model_name, purge_step=global_step
                )

        else:
            global_step = 0
            start_epoch = 0
            start_step = 0
            if args.local_rank in [-1, 0]:
                tb_writer = SummaryWriter(
                    log_dir="runs/" + args.my_model_name, purge_step=global_step
                )

            logger.info("Start from scratch")
    else:
        global_step = 0
        start_epoch = 0
        start_step = 0
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter(
                log_dir="runs/" + args.my_model_name, purge_step=global_step
            )
            pass
        logger.info("Start from scratch")

    tr_loss, logging_loss = 0.0, 0.0
    pretrained_finbert_model.zero_grad()
    adapter_ensemble_model.zero_grad()
    for epoch_step in range(start_epoch, int(args.num_train_epochs)):
        epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            start = time.time()
            # if args.restore and (step < start_step):
            #     continue
            # if args.restore and (flag_count < global_step):
            #     flag_count+=1
            #     continue
            pretrained_finbert_model.eval()
            adapter_ensemble_model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                # XLM and RoBERTa don't use segment_ids
                "labels": batch[3],
            }
            pretrained_model_outputs = pretrained_finbert_model(**inputs)
            outputs = adapter_ensemble_model(pretrained_model_outputs, **inputs)

            loss = outputs[
                0
            ]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # epoch_iterator.set_description("loss {}".format(loss))
            logger.info(
                "Epoch {}/{} - Iter {} / {}, loss = {:.5f}, time used = {:.3f}s".format(
                    epoch_step,
                    int(args.num_train_epochs),
                    step,
                    len(train_dataloader),
                    loss.item(),
                    time.time() - start,
                )
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                adapter_ensemble_model.parameters(), args.max_grad_norm
            )

            tr_loss += loss.item()
            epoch_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                # model.zero_grad()
                pretrained_finbert_model.zero_grad()
                adapter_ensemble_model.zero_grad()
                global_step += 1
                # if (
                #     args.local_rank in [-1, 0]
                #     and args.logging_steps > 0
                #     and global_step % args.logging_steps == 0
                # ):
                #     # Log metrics
                #     tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                #     tb_writer.add_scalar(
                #         "loss",
                #         (tr_loss - logging_loss) / args.logging_steps,
                #         global_step,
                #     )
                #     logging_loss = tr_loss

                # if (
                #     args.local_rank == -1
                #     and args.evaluate_during_training
                #     and global_step % args.eval_steps == 0
                # ):  # Only evaluate when single GPU otherwise metrics may not average well
                #     model = (pretrained_finbert_model, adapter_ensemble_model)
                #     results = evaluate(args, val_dataset, model, tokenizer)
                #     for key, value in results.items():
                #         tb_writer.add_scalar("eval_{}".format(key), value, global_step)

        # Epoch ended
        # Make start_step = 0
        # start_step=0
        # Log metrics training
        tb_writer.add_scalar("lr", scheduler.get_lr()[0], epoch_step)
        tb_writer.add_scalar("loss", epoch_loss / step, epoch_step)
        # Log metrics evaluation
        results = evaluate(args, val_dataset, model)
        for key, value in results.items():
            tb_writer.add_scalar("eval_{}".format(key), value, epoch_step)

        if (
            args.local_rank in [-1, 0]
            and args.save_epoch_steps > 0
            and epoch_step % args.save_epoch_steps == 0
        ):
            # Save model checkpoint
            output_dir = os.path.join(
                args.output_dir, "checkpoint-{}".format(epoch_step)
            )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                adapter_ensemble_model.module
                if hasattr(adapter_ensemble_model, "module")
                else adapter_ensemble_model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(
                output_dir
            )  # save to pytorch_model.bin  model.state_dict()

            torch.save(
                optimizer.state_dict(),
                os.path.join(output_dir, "optimizer.bin"),
            )
            torch.save(
                scheduler.state_dict(),
                os.path.join(output_dir, "scheduler.bin"),
            )
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            torch.save(global_step, os.path.join(args.output_dir, "global_step.bin"))

            logger.info(
                "Saving model checkpoint, optimizer, global_step to %s",
                output_dir,
            )
            if (epoch_step / args.save_epoch_steps) > args.max_save_checkpoints:
                try:
                    shutil.rmtree(
                        os.path.join(
                            args.output_dir,
                            "checkpoint-{}".format(
                                epoch_step
                                - args.max_save_checkpoints * args.save_epoch_steps
                            ),
                        )
                    )
                except OSError as e:
                    print(e)

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


save_results = []


def evaluate(args, val_dataset, model):
    pretrained_finbert_model = model[0]
    adapter_ensemble_model = model[1]
    results = {}

    val_sampler = (
        SequentialSampler(val_dataset)
        if args.local_rank == -1
        else DistributedSampler(val_dataset)
    )
    val_dataloader = DataLoader(
        val_dataset, sampler=val_sampler, batch_size=args.eval_batch_size
    )

    # validation.
    logging.info("***** Running validation *****")
    logging.info(f"  Num val_examples = {len(val_dataset)}")
    logging.info(" Validation Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    prediction = []
    gold_result = []
    start = time.time()
    for step, batch in enumerate(val_dataloader):
        pretrained_finbert_model.eval()
        adapter_ensemble_model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
            }

            pretrained_model_outputs = pretrained_finbert_model(**inputs)
            outputs = adapter_ensemble_model(pretrained_model_outputs, **inputs)

            tmp_eval_loss, logits = outputs[:2]
            sigmoid_preds = torch.sigmoid(logits)
            preds = torch.round(sigmoid_preds)
            # preds = logits.argmax(dim=1)
            prediction += preds.tolist()
            gold_result += inputs["labels"].tolist()
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        logger.info(
            "Validation Iter {} / {}, loss = {:.5f}, time used = {:.3f}s".format(
                step,
                len(val_dataloader),
                tmp_eval_loss.mean().item(),
                time.time() - start,
            )
        )

    micro_F1 = f1_score(y_true=gold_result, y_pred=prediction, average="micro")
    macro_F1 = f1_score(y_true=gold_result, y_pred=prediction, average="macro")

    logger.info("The micro_f1 on dev dataset: %f", micro_F1)
    logger.info("The macro_f1 on dev dataset: %f", macro_F1)
    results["micro_F1"] = micro_F1
    results["macro_F1"] = macro_F1
    results["loss"] = eval_loss
    output_eval_file = os.path.join(
        args.output_dir, args.my_model_name + "eval_results.txt"
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results  *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))
    return results


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--finbert_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: "
        + ", ".join(processors.keys()),
    )
    parser.add_argument("--comment", default="", type=str, help="The comment")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--adapter_transformer_layers",
        default=2,
        type=int,
        help="The transformer layers of adapter.",
    )
    parser.add_argument(
        "--adapter_size", default=768, type=int, help="The hidden size of adapter."
    )
    parser.add_argument(
        "--adapter_list",
        default="0,5,10",
        type=str,
        help="The layer where add an adapter",
    )
    parser.add_argument(
        "--adapter_skip_layers",
        default=0,
        type=int,
        help="The skip_layers of adapter according to bert layers",
    )
    parser.add_argument(
        "--meta_adapter_model",
        default="",
        type=str,
        help="the pretrained factual adapter model",
    )

    ## Other

    parser.add_argument(
        "--restore",
        type=bool,
        default=True,
        help="Whether restore from the last checkpoint, is nochenckpoints, start from scartch",
    )

    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--sentiment_analysis_num_labels",
        default=2,
        type=int,
        help="The number of labels for the sentiment analysis task.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )

    parser.add_argument(
        "--train_batch_size",
        default=64,
        type=int,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=64,
        type=int,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=100,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_epoch_steps",
        type=int,
        default=1,
        help="Save checkpoint every X epochs steps.",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=None, help="eval every X updates steps."
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--max_save_checkpoints",
        type=int,
        default=3,
        help="The max amounts of checkpoint saving. Bigger than it will delete the former checkpoints",
    )

    # parser.add_argument(
    #     "--fp16",
    #     action="store_true",
    #     help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    # )
    # parser.add_argument(
    #     "--fp16_opt_level",
    #     type=str,
    #     default="O1",
    #     help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    #     "See details at https://nvidia.github.io/apex/amp.html",
    # )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--meta_bertmodel", default="", type=str, help="the pretrained bert model"
    )
    parser.add_argument(
        "--save_model_iteration", type=int, help="when to save the model.."
    )

    args = parser.parse_args()

    args.adapter_list = args.adapter_list.split(",")
    args.adapter_list = [int(i) for i in args.adapter_list]

    # name_prefix = f"{list(filter(None, args.finbert_path.split('/'))).pop()}_{args.percentage_change_type}_kfold-{args.data_dir.split('_')[-1]}_max_seq-{args.max_seq_length}_rnn_num_layers-{args.rnn_num_layers}_rnn_hidden_size-{args.rnn_hidden_size}_batch-{args.train_batch_size}_lr-{args.learning_rate}_warmup-{args.warmup_steps}_epoch-{args.num_train_epochs}_comment-{args.comment}"
    name_prefix = (
        "maxlen-"
        + str(args.max_seq_length)
        + "_"
        + "kfold-"
        + str(args.data_dir.split('_')[-1])
        + "_"
        + "batch-"
        + str(args.train_batch_size)
        + "_"
        + "lr-"
        + str(args.learning_rate)
        + "_"
        + "warmup-"
        + str(args.warmup_steps)
        + "_"
        + "epoch-"
        + str(args.num_train_epochs)
        + "_"
        + str(args.comment)
    )
    args.my_model_name = args.task_name + "_" + name_prefix
    args.output_dir = os.path.join(args.output_dir, args.my_model_name)

    if args.eval_steps is None:
        args.eval_steps = args.save_epoch_steps * 10

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    # else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     torch.distributed.init_process_group(backend="nccl")
    #     args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    # logger.warning(
    #     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    #     args.local_rank,
    #     device,
    #     args.n_gpu,
    #     bool(args.local_rank != -1),
    #     args.fp16,
    # )

    logger.warning("Process rank: %s, device: %s", args.local_rank, device)

    # Set seed
    set_seed(args)

    args.output_mode = output_modes[args.task_name]

    # Choose tokenizer for BERT or FinBERT
    if list(filter(None, args.finbert_path.split("/"))).pop() == "bert-base-uncased":
        tokenizer = BertTokenizerLocal.from_pretrained("bert-base-uncased")
    elif list(filter(None, args.finbert_path.split("/"))).pop() == "FinBERT":
        tokenizer = BertTokenizerHugging.from_pretrained(
            "yiyanghkust/finbert-tone", model_max_length=args.max_seq_length
        )
    pretrained_model = PretrainedModel(args)
    adapter_ensemble_model = AdapterEnsembleModel(args, pretrained_model.config)

    if args.meta_adapter_model:
        model_dict = adapter_ensemble_model.state_dict()
        logger.info("Adapter model weight:")
        logger.info(adapter_ensemble_model.state_dict().keys())
        logger.info("Load model state dict from {}".format(args.meta_adapter_model))
        adapter_meta_dict = torch.load(
            args.meta_adapter_model, map_location=lambda storage, loc: storage
        )
        logger.info("Load pretraiend adapter model state dict ")
        logger.info(adapter_meta_dict.keys())

        changed_adapter_meta = {}
        for key in adapter_meta_dict.keys():
            changed_adapter_meta[
                key.replace("encoder.", "adapter.encoder.")
            ] = adapter_meta_dict[key]

        changed_adapter_meta = {
            k: v for k, v in changed_adapter_meta.items() if k in model_dict.keys()
        }
        model_dict.update(changed_adapter_meta)
        adapter_ensemble_model.load_state_dict(model_dict)
    pretrained_model.to(args.device)
    adapter_ensemble_model.to(args.device)

    model = (pretrained_model, adapter_ensemble_model)

    logger.info("Training/evaluation parameters %s", args)

    val_dataset = load_and_cache_examples(
        args, args.task_name, tokenizer, "val", evaluate=True
    )

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(
            args, args.task_name, tokenizer, "train", evaluate=False
        )

        global_step, tr_loss = train(args, train_dataset, val_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # model_to_save = (
        #     model.module if hasattr(model, "module") else model
        # )  # Take care of distributed/parallel training
        model_to_save = adapter_ensemble_model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


if __name__ == "__main__":
    main()
