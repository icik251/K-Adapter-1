import argparse
import os
import random
import shutil
import numpy as np
from torch import nn
import torch
import logging

from torch.nn import MSELoss

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)

from examples.utils_sec import (
    output_modes,
    processors,
)

from pytorch_transformers import AdamW, WarmupLinearSchedule

logger = logging.getLogger(__name__)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class KPIModel(nn.Module):
    def __init__(self, args):
        super(KPIModel, self).__init__()
        self.input_size = args.kpi_input_size
        self.num_classes = args.kpi_num_classes
        self.hidden_size = args.kpi_hidden_size
        self.hidden_layers = args.kpi_hidden_layers

        if self.hidden_layers == 0:
            self.layers = nn.Linear(self.input_size, self.num_classes)
        elif self.hidden_layers == 1:
            self.layers = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.Dropout(args.kpi_dropout_prob),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.num_classes),
            )

    def forward(self, x, labels=None):
        outputs = self.layers(x)
        
        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(outputs.squeeze(1), labels)
            outputs = (loss, outputs)
            
        return outputs

    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model can be saved"
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "kpi_pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Saving model checkpoint to %s", save_directory)


def load_and_cache_examples(args, task, dataset_type, evaluate=False):
    # Modify for our dataset
    # dataset_type: train, dev, test
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            dataset_type,
            str(task),
            args.percentage_change_type
        ),
    )
    # Remove "not" when finished with development
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        features = (
            processor.get_dev_examples(
                args.data_dir, args.percentage_change_type, dataset_type
            )
            if evaluate
            else processor.get_train_examples(
                args.data_dir, args.percentage_change_type, dataset_type
            )
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_features = torch.tensor([f.input_features for f in features], dtype=torch.float)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_features, all_label_ids)

    return dataset


def train(args, train_dataset, val_dataset, model):
    """Train the model"""

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir="runs/" + args.my_model_name)

    train_sampler = RandomSampler(train_dataset)
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
        Optimizer and Scheduler for KPI model
    """

    # Comment out if using the real grouped parameters
    optimizer_grouped_parameters = model.parameters()
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

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()

    train_iterator = trange(
        int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for epoch_step in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        epoch_loss = 0
        for step, batch in enumerate(epoch_iterator):
            model.train()

            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_features": batch[0],
                "labels": batch[1],
            }
            outputs = model(inputs["input_features"], inputs["labels"])
            loss = outputs[0]

            # if args.n_gpu > 1:
            #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
            # if args.gradient_accumulation_steps > 1:
            #     loss = loss / args.gradient_accumulation_steps

            loss.backward()

            """Clipping gradients"""
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            tr_loss += loss.item()
            epoch_loss += loss.item()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1
                # if (
                #     args.local_rank in [-1, 0]
                #     and args.logging_steps > 0
                #     and global_step % args.logging_steps == 0
                # ):
                #     # Log metrics
                #     tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                #     tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                #     logging_loss = tr_loss

                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step)
                    )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model.save_pretrained(
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
                    torch.save(
                        global_step, os.path.join(args.output_dir, "global_step.bin")
                    )

                    logger.info(
                        "Saving model checkpoint, optimizer, global_step to %s",
                        output_dir,
                    )
                    if (global_step / args.save_steps) > args.max_save_checkpoints:
                        try:
                            shutil.rmtree(
                                os.path.join(
                                    args.output_dir,
                                    "checkpoint-{}".format(
                                        global_step
                                        - args.max_save_checkpoints * args.save_steps
                                    ),
                                )
                            )
                        except OSError as e:
                            print(e)
                # if (
                #     args.local_rank == -1
                #     and args.evaluate_during_training
                #     and global_step % args.eval_steps == 0
                # ):  # Only evaluate when single GPU otherwise metrics may not average well
                #     results = evaluate(args, val_dataset, model)
                #     for key, value in results.items():
                #         tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
            if args.max_steps > 0 and global_step > args.max_steps:
                break
        
        # Epoch ended
        # Log metrics training
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch_step)
        tb_writer.add_scalar('loss', epoch_loss / step, epoch_step)
        # Log metrics evaluation
        results = evaluate(args, val_dataset, model)
        for key, value in results.items():
            tb_writer.add_scalar('eval_{}'.format(key), value, epoch_step)
        
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

save_results = []


def evaluate(args, eval_dataset, model, prefix=""):
    results = {}
    # for dataset_type in ["dev", "test"]:

    # for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # args.eval_batch_size = args.eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = (SequentialSampler(eval_dataset))
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    # eval_acc = 0
    index = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        index += 1

        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_features": batch[0],
                "labels": batch[1],
            }
            outputs = model(inputs["input_features"], inputs["labels"])
            tmp_eval_loss = outputs[0]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        logits = outputs[1]
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )

    eval_loss = eval_loss / nb_eval_steps
    # if args.output_mode == "classification":
    #     preds = np.argmax(preds, axis=1)
    # elif args.output_mode == "regression":
    #     preds = np.squeeze(preds)

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
        "--percentage_change_type",
        default="percentage_change",
        type=str,
        required=True,
        help="The percentage change type for the label. Can be: percentage_change, percentage_change_standard, percentage_change_min_max",
    )
    parser.add_argument(
        "--kpi_input_size",
        default=116,
        type=int,
        help="Input size for KPI model.",
    )
    parser.add_argument(
        "--kpi_hidden_layers",
        default=1,
        type=int,
        help="Number of hidden layers for the regression task for KPI model.",
    )
    parser.add_argument(
        "--kpi_hidden_size",
        default=64,
        type=int,
        help="Number of neurons in hidden layer for the regression task for KPI model.",
    )
    parser.add_argument(
        "--kpi_dropout_prob",
        default=0.2,
        type=float,
        help="Number of neurons in hidden layer for the regression task for KPI model.",
    )
    parser.add_argument(
        "--kpi_num_classes",
        default=1,
        type=int,
        help="Output for the regression task for KPI model.",
    )

    ## Other parameters
    parser.add_argument(
        "--eval_steps", type=int, default=10, help="eval every X updates steps."
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
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
        default=8,
        type=int,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
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
        default=100.0,
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
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X updates steps.",
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
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--save_model_iteration", type=int, help="when to save the model.."
    )
    parser.add_argument(
        "--max_save_checkpoints",
        type=int,
        default=5,
        help="The max amounts of checkpoint saving. Bigger than it will delete the former checkpoints",
    )

    args = parser.parse_args()

    name_prefix = f"kpi-symbolic_{args.percentage_change_type}_kfold-{args.data_dir.split('_')[-1]}_batch-{args.train_batch_size}_lr-{args.learning_rate}_warmup-{args.warmup_steps}_epoch-{args.num_train_epochs}_comment-{args.comment}"
    args.my_model_name = args.task_name + "_" + name_prefix
    args.output_dir = os.path.join(args.output_dir, args.my_model_name)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.warning("Process rank: %s, device: %s", args.local_rank, device)

    # Set seed
    set_seed(args)

    kpi_model = KPIModel(args)
    kpi_model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    val_dataset = load_and_cache_examples(
        args, args.task_name, "val", evaluate=True
    )
    
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(
            args, args.task_name, "train", evaluate=False
        )
        # print("Features created!")
        # return
        global_step, tr_loss = train(args, train_dataset, val_dataset, kpi_model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # Saving KPI symbolic model
        logger.info("Saving KPI symbolic model checkpoint to %s", args.output_dir)
        # They can then be reloaded using `from_pretrained()`
        kpi_model.save_pretrained(args.output_dir)

    # # Evaluation
    # results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         model = model_class.from_pretrained(checkpoint)
    #         model.to(args.device)
    #         result = evaluate(args, model, tokenizer, prefix=global_step)
    #         logger.info('micro f1:{}'.format(result))
    #         result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    #         results.update(result)
    # save_result = str(results)
    # save_results.append(save_result)

    # result_file = open(os.path.join(args.output_dir, args.my_model_name + '_result.txt'), 'w')
    # for line in save_results:
    #     result_file.write(str(line) + '\n')
    # result_file.close()

    # return results


if __name__ == "__main__":
    main()
