#!/usr/bin/env python

import json
import os
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import RobertaTokenizer

from dataset_utils.collate_functions import collate_to_max_length
from datasets import load_dataset

class ANLIDataset(Dataset):
    label_map = {
        "contradiction": 0, 
        'neutral': 1,
        "entailment": 2,
        "0": 2,
        "1": 1,
        "2": 0,
    }

    def __init__(self, directory, prefix, bert_path, max_length: int = 512):
        super().__init__()
        self.max_length = max_length

        self.dataset = load_dataset('anli')
        self.dataset = ConcatDataset([
            self.dataset[prefix + '_r1'],
            self.dataset[prefix + '_r2'],
            self.dataset[prefix + '_r3'],
        ])
        self.tokenizer = RobertaTokenizer.from_pretrained(bert_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        sentence_1, sentence_2 = row['premise'], row['hypothesis']
        label = self.label_map[str(row['label'])]
        # remove .
        if sentence_1.endswith("."):
            sentence_1 = sentence_1[:-1]
        if sentence_2.endswith("."):
            sentence_2 = sentence_2[:-1]
        sentence_1_input_ids = self.tokenizer.encode(sentence_1, add_special_tokens=False)
        sentence_2_input_ids = self.tokenizer.encode(sentence_2, add_special_tokens=False)
        input_ids = sentence_1_input_ids + [2] + sentence_2_input_ids
        if len(input_ids) > self.max_length - 2:
            input_ids = input_ids[:self.max_length - 2]
        # convert list to tensor
        length = torch.LongTensor([len(input_ids) + 2])
        input_ids = torch.LongTensor([0] + input_ids + [2])
        label = torch.LongTensor([label])
        return input_ids, label, length

