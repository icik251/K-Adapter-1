""" BERT regression fine-tuning: utilities to work with SEC reports"""

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open
import json
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class InputExampleParagraph(object):
    """A single training/test example for a single paragraph."""

    def __init__(self, text_a):
        """Constructs a InputExampleParagraph.

        Args:
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        """
        self.text_a = text_a


class InputExampleFiling(object):
    """A single training/test example for a single filing."""

    def __init__(self, guid, list_input_examples_paragraphs, label):
        """Constructs a InputExampleFiling.

        Args:
            guid: Unique id for the example.
            list_input_examples_paragraphs: list. List containing InputExampleParagraphs.
            label: (Optional) float. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.list_input_examples_paragraphs = list_input_examples_paragraphs
        self.label = label


class InputExampleSentiment(object):
    """A single training/test example for a single sentiment analysis sameple and label"""

    def __init__(self, guid, text_a, label):
        """Constructs a InputExampleFiling.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.label = label


class InputFeaturesParagraph(object):
    """A single set of features for just one paragraph."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        # self.label_id = label_id
        # self.start_id = start_id


class InputFeaturesFiling(object):
    """A single set of features for a whole filing"""

    def __init__(self, list_input_features_paragraphs, label_id) -> None:
        self.list_input_features_paragraphs = list_input_features_paragraphs
        self.label_id = label_id


class InputFeaturesSentiment(object):
    """A single set of features for the sentiment analysis sample"""

    def __init__(self, input_ids, input_mask, segment_ids, label_id) -> None:
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, "utf-8") for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r", encoding="utf8") as f:
            return json.load(f)

    @classmethod
    def _read_semeval_txt(clas, input_file):
        with open(input_file, "r", encoding="utf8") as f:
            examples = []
            example = []
            for line in f:
                if line.strip() == "":
                    examples.append(example)
                    example = []
                else:
                    example.append(line.strip())
            return examples


class SECProcessor(DataProcessor):
    """Processor for our SEC filings data"""

    def get_train_examples(self, data_dir, percentage_change_type, dataset_type=None):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")),
            percentage_change_type,
        )

    def get_dev_examples(self, data_dir, percentage_change_type, dataset_type):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))),
            percentage_change_type,
        )

    def get_labels(self):
        """See base class."""
        return 0

    def _create_examples(self, list_of_dicts, percentage_change_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, curr_dict_input) in enumerate(list_of_dicts):
            if "mda_paragraphs" not in curr_dict_input.keys():
                print(f"MDA missing {i} | {curr_dict_input[percentage_change_type]}")
                continue

            list_input_examples_paragraphs = []
            for k, mda_paragraph in curr_dict_input["mda_paragraphs"].items():
                text_a = mda_paragraph
                curr_paragraph_per_filing = InputExampleParagraph(text_a=text_a)
                list_input_examples_paragraphs.append(curr_paragraph_per_filing)
            guid = i
            label = curr_dict_input[percentage_change_type]

            if list_input_examples_paragraphs:
                examples.append(
                    InputExampleFiling(guid, list_input_examples_paragraphs, label)
                )
            else:
                print(f"Absolutely empty filing: {i} | {label}")

        return examples


class SECAdapterProcessor(DataProcessor):
    """Processor for our SEC sentiment analysis data"""

    def get_train_examples(self, data_dir, dataset_type=None):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json"))
        )

    def get_dev_examples(self, data_dir, dataset_type):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type)))
        )

    def get_labels(self):
        """See base class."""
        return ["positive", "negative"]

    def _create_examples(self, list_of_dicts):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, curr_dict_input) in enumerate(list_of_dicts):
            """Implement logic for sentiment analysis data"""

            examples.append(
                InputExampleSentiment(
                    i, curr_dict_input["text"], curr_dict_input["label"]
                )
            )

        return examples


# Modified from entity typing
def convert_examples_to_features_sec(
    examples,
    max_seq_length,
    tokenizer,
    output_mode,
    cls_token="[CLS]",
    sep_token="[SEP]",
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=1,
    mask_padding_with_zero=True,
):
    """Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    # check if pad_token_segment_id should be 0
    features = []
    for (ex_index, filing_example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        list_input_features_paragraphs = []
        for input_example_paragraph in filing_example.list_input_examples_paragraphs:
            sentence = input_example_paragraph.text_a
            tokens_sentence = tokenizer.tokenize(sentence)
            # truncate if needed
            tokens_sentence = (
                [cls_token] + tokens_sentence[: max_seq_length - 2] + [sep_token]
            )

            segment_ids = [sequence_a_segment_id] * len(tokens_sentence)
            input_ids = tokenizer.convert_tokens_to_ids(tokens_sentence)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = (
                    [0 if mask_padding_with_zero else 1] * padding_length
                ) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length
                )
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            curr_input_features_paragraph_per_filing = InputFeaturesParagraph(
                input_ids, input_mask, segment_ids
            )
            list_input_features_paragraphs.append(
                curr_input_features_paragraph_per_filing
            )

        # start_id = np.zeros(max_seq_length)
        # start_id[start] = 1

        if output_mode == "classification":
            label_id = filing_example.label
        elif output_mode == "regression":
            label_id = float(filing_example.label)
        else:
            raise KeyError(output_mode)

        # TODO: Log the correct info because here we are logging a single example paragraph
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (filing_example.guid))
            # logger.info("tokens: %s" % " ".join(
            #     [str(x) for x in tokens_sentence]))
            # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            # logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            # logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(label_id))

        features.append(InputFeaturesFiling(list_input_features_paragraphs, label_id))

        # Only for testing purposes
        # if len(features) >= 1000:
        #     break

    return features


def convert_examples_to_features_sec_adapter(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    output_mode,
    cls_token="[CLS]",
    sep_token="[SEP]",
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=1,
    mask_padding_with_zero=True,
):
    """Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    # check if pad_token_segment_id should be 0
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, adapter_example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        sentence = adapter_example.text_a
        tokens_sentence = tokenizer.tokenize(sentence)
        # truncate if needed
        tokens_sentence = (
            [cls_token] + tokens_sentence[: max_seq_length - 2] + [sep_token]
        )

        segment_ids = [sequence_a_segment_id] * len(tokens_sentence)
        input_ids = tokenizer.convert_tokens_to_ids(tokens_sentence)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + (
                [0 if mask_padding_with_zero else 1] * padding_length
            )
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[adapter_example.label]

        curr_input_features_sentiment = InputFeaturesSentiment(
            input_ids, input_mask, segment_ids, label_id
        )

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (adapter_example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens_sentence]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(label_id))

        features.append(curr_input_features_sentiment)

        # Only for testing purposes
        # if len(features) >= 1000:
        #     break

    return features


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "regression":
        pass
        # Some MSE calculation
        # return figer_scores(preds, labels)
    elif task_name == "classification":
        pass
        # return micro_f1_tacred(preds, labels)
    else:
        raise KeyError(task_name)


class SECDataset(Dataset):
    def __init__(self, filings_features, labels_ids, max_seq_length):

        self.filings_features = []
        longest_list_paragraphs = 0
        for idx_f, curr_filing_features in enumerate(filings_features):
            curr_list_of_paragraphs = []
            for idx_p, paragraph_features in enumerate(
                curr_filing_features.list_input_features_paragraphs
            ):
                curr_dict_filing = {}
                curr_dict_filing["input_ids"] = paragraph_features.input_ids
                curr_dict_filing["input_mask"] = paragraph_features.input_mask
                curr_dict_filing["segment_ids"] = paragraph_features.segment_ids
                curr_list_of_paragraphs.append(curr_dict_filing)

            if len(curr_list_of_paragraphs) > longest_list_paragraphs:
                longest_list_paragraphs = len(curr_list_of_paragraphs)

            self.filings_features.append([curr_list_of_paragraphs, labels_ids[idx_f]])

        for idx_for_pad, (list_of_paragraphs, _) in enumerate(self.filings_features):
            num_to_pad = longest_list_paragraphs - len(list_of_paragraphs)
            tensor_to_pad = torch.zeros(max_seq_length, dtype=torch.long)
            self.filings_features[idx_for_pad][0] = (
                list_of_paragraphs
                + [
                    {
                        "input_ids": tensor_to_pad,
                        "input_mask": tensor_to_pad,
                        "segment_ids": tensor_to_pad,
                    }
                ]
                * num_to_pad
            )

        # self.filings_features = filings_features
        # self.labels_ids = labels_ids

    def __len__(self):
        return len(self.filings_features)

    def __getitem__(self, index):
        # convert to input features filing to dict to be able to pass it in the batch

        return self.filings_features[index]


processors = {"sec_regressor": SECProcessor, "sec_adapter": SECAdapterProcessor}

output_modes = {"sec_regressor": "regression", "sec_adapter": "classification"}

SEC_TASKS_NUM_LABELS = {"sec_regressor": 1, "sec_adapter": 1}
