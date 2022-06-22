import argparse
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import os
from torch import nn
import torch
from pytorch_transformers import BertModel
from pytorch_transformers.modeling_bert import BertEncoder
import logging

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pretrained_adapter(adapter, adapter_path):
    new_adapter = adapter
    model_dict = new_adapter.state_dict()
    logger.info("Adapter model weight:")
    logger.info(new_adapter.state_dict().keys())
    # print(model_dict['bert.encoder.layer.2.intermediate.dense.weight'])
    logger.info("Load model state dict from {}".format(adapter_path))
    adapter_meta_dict = torch.load(
        adapter_path, map_location=lambda storage, loc: storage
    )
    for item in [
        "out_proj.bias",
        "out_proj.weight",
        "dense.weight",
        "dense.bias",
    ]:  # 'adapter.down_project.weight','adapter.down_project.bias','adapter.up_project.weight','adapter.up_project.bias'
        if item in adapter_meta_dict:
            adapter_meta_dict.pop(item)
    changed_adapter_meta = {}
    for key in adapter_meta_dict.keys():
        changed_adapter_meta[key.replace("adapter.", "adapter.")] = adapter_meta_dict[
            key
        ]
    changed_adapter_meta = {
        k: v for k, v in changed_adapter_meta.items() if k in model_dict.keys()
    }
    model_dict.update(changed_adapter_meta)
    new_adapter.load_state_dict(model_dict)
    return new_adapter


class Adapter(nn.Module):
    def __init__(self, adapter_config):
        super(Adapter, self).__init__()
        self.adapter_config = adapter_config
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
        attention_mask = torch.ones(input_shape, device=DEVICE)
        encoder_attention_mask = torch.ones(input_shape, device=DEVICE)
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


class AdapterModel(nn.Module):
    def __init__(self, pretrained_model_config):
        super(AdapterModel, self).__init__()
        self.config = pretrained_model_config

        class AdapterConfig:
            project_hidden_size: int = self.config.hidden_size
            hidden_act: str = "gelu"
            adapter_size: int = 768  # 64
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
            num_hidden_layers: int = 2
            num_labels: int = 2
            output_attentions: bool = False
            output_hidden_states: bool = False
            torchscript: bool = False
            type_vocab_size: int = 2
            vocab_size: int = 30878

        self.adapter_skip_layers = 0
        self.adapter_list = [0, 5, 11]
        self.adapter_num = len(self.adapter_list)
        self.adapter = nn.ModuleList(
            [Adapter(AdapterConfig) for _ in range(self.adapter_num)]
        )

    def forward(self, pretrained_model_outputs):

        outputs = pretrained_model_outputs
        sequence_output = outputs[0]  # 12-th hidden layer (11th idx)
        # pooler_output = outputs[1]
        hidden_states = outputs[2]  # all hidden layers so we can take 0,5,11 later
        num = len(hidden_states)
        hidden_states_last = torch.zeros(sequence_output.size()).to(DEVICE)

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

        outputs = (hidden_states_last,) + outputs[2:]
        return outputs  # (loss), logits, (hidden_states), (attentions)


class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.input_size = 768
        self.hidden_size = 256
        self.num_layers = 2
        self.num_classes = 1

        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, self.num_layers, batch_first=True
        )
        self.linear_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.linear_layers(out[:, -1, :])
        return out

    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model can be saved"
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "rnn_pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Saving model checkpoint to %s", save_directory)


class AdapterEnsembleModel(nn.Module):
    def __init__(self, pretrained_model_config, sec_adapter):
        super(AdapterEnsembleModel, self).__init__()
        self.config = pretrained_model_config

        self.sec_adapter = sec_adapter

        if self.sec_adapter is not None:
            for p in self.sec_adapter.parameters():
                p.requires_grad = False

            # self.task_dense_lin = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)
        self.task_dense_sec = nn.Linear(
            self.config.hidden_size + self.config.hidden_size,
            self.config.hidden_size,
        )

    def forward(
        self,
        pretrained_model_outputs,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        start_id=None,
    ):
        pretrained_model_last_hidden_states = pretrained_model_outputs[0]
        if self.sec_adapter is not None:
            sec_adapter_outputs, _ = self.sec_adapter(pretrained_model_outputs)
            combine_features = pretrained_model_last_hidden_states
            sec_features = self.task_dense_sec(
                torch.cat([combine_features, sec_adapter_outputs], dim=2)
            )
        else:
            sec_features = pretrained_model_last_hidden_states

        sec_features_squeezed = sec_features[:, 0, :]
        return sec_features_squeezed


class FinBERTModel(nn.Module):
    def __init__(self, finbert_path):
        super(FinBERTModel, self).__init__()
        self.model = BertModel.from_pretrained(
            pretrained_model_name_or_path=finbert_path, output_hidden_states=True
        )
        self.config = self.model.config
        self.config.freeze_adapter = True
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
