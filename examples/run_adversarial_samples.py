from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from sklearn.metrics import mean_squared_error
import torch
from pytorch_transformers.my_modeling_finbert import (
    FinBERTModel,
    AdapterEnsembleModel,
    RNNModel,
    AdapterModel,
    load_pretrained_adapter,
)
from transformers import BertTokenizer as BertTokenizerHugging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DecisionSupportSystem:
    def __init__(
        self, finbert_path, adapter_path, adapter_ensemble_path, rnn_path
    ) -> None:
        self.tokenizer = BertTokenizerHugging.from_pretrained(
            "yiyanghkust/finbert-tone", model_max_length=256
        )
        self.pretrained_finbert_model = FinBERTModel(finbert_path)
        if adapter_path:
            self.sec_adapter = AdapterModel(self.pretrained_finbert_model.config)
            self.sec_adapter = load_pretrained_adapter(self.sec_adapter, adapter_path)
        else:
            self.sec_adapter = None

        self.adapter_ensemble_model = AdapterEnsembleModel(
            self.pretrained_finbert_model.config, sec_adapter=self.sec_adapter
        )
        self.adapter_ensemble_model.load_state_dict(torch.load(adapter_ensemble_path))
        self.rnn_model = RNNModel()
        self.rnn_model.load_state_dict(torch.load(rnn_path))

        self.pretrained_finbert_model.to(DEVICE)
        self.sec_adapter.to(DEVICE)
        self.adapter_ensemble_model.to(DEVICE)
        self.rnn_model.to(DEVICE)

    def predict(self, list_of_paragraphs):
        # Encode paragraphs
        all_paragraphs_ids = []
        for paragraph in list_of_paragraphs:
            all_paragraphs_ids.append(
                torch.tensor(
                    [
                        self.tokenizer.encode(
                            paragraph, add_special_tokens=True, max_length=256
                        )
                    ]
                ).to(DEVICE)
            )

        curr_filing_encoded_paragraphs = []
        with torch.no_grad():
            for input_curr_paragraph in all_paragraphs_ids:
                pretrained_model_outputs = self.pretrained_finbert_model(
                    input_curr_paragraph
                )
                encoded_paragraph = self.adapter_ensemble_model(
                    pretrained_model_outputs, input_curr_paragraph
                )
                curr_filing_encoded_paragraphs.append(encoded_paragraph.squeeze(0))

            curr_filing_encoded_paragraphs = torch.stack(curr_filing_encoded_paragraphs)

            curr_filing_encoded_paragraphs = curr_filing_encoded_paragraphs.unsqueeze(
                0
            ).to(DEVICE)
            rnn_output_for_filing = self.rnn_model(curr_filing_encoded_paragraphs)

            return rnn_output_for_filing


class SamplesHandler:
    def __init__(self, path_to_sample_json) -> None:
        with open(path_to_sample_json, "r") as f:
            self.list_of_samples = json.load(f)

    def get_samples(self, only_adversarial=True):
        list_of_filings = []
        if only_adversarial:
            for filing_dict in self.list_of_samples:
                if len(filing_dict["adversarial_samples"]) > 0:
                    label = filing_dict["percentage_change_robust"]
                    if "mda_paragraphs" not in filing_dict.keys():
                        print(f"MDA missing | label: {label}")
                        continue

                    list_of_texts_for_filing = list(
                        filing_dict["mda_paragraphs"].values()
                    )
                    if not list_of_texts_for_filing:
                        print(f"Absolutely empty filing | label: {label}")
                        continue

                    # add adversarial
                    # adversarial_sentence = filing_dict["adversarial_samples"][0] + ". "
                    adversarial_sentence = "The company is doing awesome."
                    list_of_texts_for_filing = [
                        adversarial_sentence + item for item in list_of_texts_for_filing
                    ]

                    list_of_filings.append((list_of_texts_for_filing, label))
        else:
            pass

        return list_of_filings


finbert_path = "D:/Master Thesis/Data/FinBERT/"
adapter_path = "./pretrained_models/sec-adapter/pytorch_model.bin"
adapter_ensemble_path = (
    "./pretrained_models/Test_model_sys_4_epochs0/adapter_ensemble_pytorch_model.bin"
)
# The different LSTM depending on the system
rnn_path_sys_1 = "./pretrained_models/Test_model_sys_4_epochs0/rnn_pytorch_model.bin"
rnn_path_sys_2 = "./pretrained_models/Test_model_sys_4_epochs0/rnn_pytorch_model.bin"
rnn_path_sys_3 = "./pretrained_models/Test_model_sys_4_epochs0/rnn_pytorch_model.bin"
rnn_path_sys_4 = "./pretrained_models/Test_model_sys_4_epochs0/rnn_pytorch_model.bin"

# system_1 = DecisionSupportSystem(finbert_path, None, adapter_ensemble_path, rnn_path_sys_1)
# system_2 = DecisionSupportSystem(finbert_path, adapter_path, adapter_ensemble_path, rnn_path_sys_2)
# system_3 = DecisionSupportSystem(finbert_path, None, adapter_ensemble_path, rnn_path_sys_3)
system_4 = DecisionSupportSystem(
    finbert_path, adapter_path, adapter_ensemble_path, rnn_path_sys_4
)

# Prepare all samples from the test set
test_samples_handler = SamplesHandler(
    "./data/real_input_sec_train_test_adversarial/final_data_train_test/val.json"
)
train_samples_handler = SamplesHandler(
    "./data/real_input_sec_train_test_adversarial/final_data_train_test/train.json"
)

list_of_filings = test_samples_handler.get_samples(only_adversarial=True)
list_of_preds = []
list_of_actuals = []
for idx, filing in enumerate(list_of_filings):
    pred = system_4.predict(filing[0])
    pred = pred.detach().cpu().item()
    list_of_preds.append(pred)
    list_of_actuals.append(filing[1])

    if idx == 0:
        print("Predictions | Actual")
    print(f"{round(pred, 3)} | {round(filing[1], 3)}")
print(f"MSE for {idx+1} filings: {mean_squared_error(list_of_actuals, list_of_preds)}")
