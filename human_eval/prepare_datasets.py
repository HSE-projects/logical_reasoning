import random
from dataclasses import dataclass, field
from typing import Optional

import random

import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from datasets import load_dataset

from transformers import (
    HfArgumentParser,
    PreTrainedTokenizer
)

@dataclass
class PreprocessArguments:
    model_name: Optional[str] = field(
        default='roberta-base', metadata={"help": "Pretrained tokenizer name or path"}
    )
    cache_dir: Optional[str] = field(
        default='.cache', metadata={"help": "Where is your cache dir"}
    )
    datasets_dir: Optional[str] = field(
        default='datasets', metadata={"help": "Where are datasets stored"}
    )
    files_path: Optional[str] = field(
        default='adv/contr', metadata={"help": ""}
    )

def shuffle_snli(examples):
    premise = examples['premise'][:-1].split(' ')
    random.shuffle(premise)
    examples['shuffled_premise'] = ' '.join(premise) + '.'
    hypothesis = examples['hypothesis'][:-1].split(' ')
    random.shuffle(hypothesis)
    examples['shuffled_hypothesis'] = ' '.join(hypothesis) + '.'
    return examples

import pandas as pd
def parse_adversarial(filename):
    premises = []
    hypotheses = []
    labels = []
    with open(filename, 'r') as f:
        cnt = 0
        for line in f:
            if cnt == 0:
                info, premise = line.split('\t')
                premises.append(premise.strip())
                cnt += 1
            elif cnt == 1:
                cnt += 1
            elif cnt == 2:
                info, hypothesis = line.split('\t')
                hypotheses.append(hypothesis.strip())
                if 'contradiction' in info:
                    labels.append(2)
                elif 'neutral' in info:
                    labels.append(1)
                else:
                    labels.append(0)
                cnt += 1
            elif cnt == 3:
                cnt = 0
    ans = pd.DataFrame()
    ans['premise'] = premises
    ans['hypothesis'] = hypotheses
    ans['label'] = labels

    return ans

def parse_contrastive_snli(filename):
    data = pd.read_excel(filename, index_col=0)
    labels = []
    premises = []
    hypotheses = []
    for row in data.iterrows():
        label = row[1]['gold_label']
        if label == 'contradiction':
            label = 2
        elif label == 'neutral':
            label = 1
        else:
            label = 0
        premise = row[1]['sentence1']
        hypothesis = row[1]['sentence2']

        if row[1]['captionID'] != 'original':
            premises.append(premise)
            labels.append(label)
            hypotheses.append(hypothesis)
    ans = pd.DataFrame()
    ans['premise'] = premises
    ans['hypothesis'] = hypotheses
    ans['label'] = labels

    return ans


def parse_contrastive_mnli(filename):
    data = pd.read_csv(filename, sep='\t')
    labels = []
    premises = []
    hypotheses = []
    for row in data.iterrows():
        label = row[1]['gold_label']
        if label == 'contradiction':
            label = 2
        elif label == 'neutral':
            label = 1
        else:
            label = 0
        premise = row[1]['sentence1']
        hypothesis = row[1]['sentence2']

        if row[1]['promptID'] != 'original':
            premises.append(premise)
            labels.append(label)
            hypotheses.append(hypothesis)
    ans = pd.DataFrame()
    ans['premise'] = premises
    ans['hypothesis'] = hypotheses
    ans['label'] = labels

    return ans


if __name__ == "__main__":
    parser = HfArgumentParser((PreprocessArguments,))
    model_args, = parser.parse_args_into_dataclasses()
     
#     snli = load_dataset("snli", cache_dir=model_args.cache_dir)
#     preprocess_snli = snli.map(shuffle_snli, batched=False)
#     preprocess_snli.save_to_disk(os.path.join(model_args.datasets_dir, 'shuffle_snli'))
    
#     mnli = load_dataset("multi_nli", cache_dir=model_args.cache_dir)
#     preprocess_mnli = mnli.map(shuffle_snli, batched=False)
#     preprocess_mnli.save_to_disk(os.path.join(model_args.datasets_dir, 'shuffle_mnli'))
    
#     pd_dataset = parse_adversarial(os.path.join(model_args.files_path, 'snli_bert'))
#     pd_dataset.to_csv(os.path.join(model_args.datasets_dir, 'adv_snli_tmp.csv'), index=False)
#     data_files = {"test": os.path.join(model_args.datasets_dir, 'adv_snli_tmp.csv')}
#     dataset = load_dataset("csv", data_files=data_files)
#     dataset.save_to_disk(os.path.join(model_args.datasets_dir, 'adv_snli'))
    
#     pd_dataset = parse_adversarial(os.path.join(model_args.files_path, 'mnli_matched_bert'))
#     pd_dataset.to_csv(os.path.join(model_args.datasets_dir, 'adv_mnli_tmp.csv'), index=False)
#     data_files = {"test": os.path.join(model_args.datasets_dir, 'adv_mnli_tmp.csv')}
#     dataset = load_dataset("csv", data_files=data_files)
#     dataset.save_to_disk(os.path.join(model_args.datasets_dir, 'adv_mnli_matched'))
    
#     pd_dataset = parse_adversarial(os.path.join(model_args.files_path, 'mnli_mismatched_bert'))
#     pd_dataset.to_csv(os.path.join(model_args.datasets_dir, 'adv_mnli_tmp.csv'), index=False)
#     data_files = {"test": os.path.join(model_args.datasets_dir, 'adv_mnli_tmp.csv')}
#     dataset = load_dataset("csv", data_files=data_files)
#     dataset.save_to_disk(os.path.join(model_args.datasets_dir, 'adv_mnli_mismatched'))

    pd_dataset = parse_contrastive_snli(os.path.join(model_args.files_path, 'snli_contrastive.xlsx'))
    pd_dataset.to_csv(os.path.join(model_args.datasets_dir, 'con_snli_tmp.csv'), index=False)
    data_files = {"test": os.path.join(model_args.datasets_dir, 'con_snli_tmp.csv')}
    dataset = load_dataset("csv", data_files=data_files)
    dataset.save_to_disk(os.path.join(model_args.datasets_dir, 'contrastive_snli'))
    
    pd_dataset = parse_contrastive_mnli(os.path.join(model_args.files_path, 'mnli_dev_matched.tsv'))
    pd_dataset.to_csv(os.path.join(model_args.datasets_dir, 'con_mnli_tmp.csv'), index=False)
    data_files = {"test": os.path.join(model_args.datasets_dir, 'con_mnli_tmp.csv')}
    dataset = load_dataset("csv", data_files=data_files)
    dataset.save_to_disk(os.path.join(model_args.datasets_dir, 'contrastive_mnli_matched'))
    
    pd_dataset = parse_contrastive_mnli(os.path.join(model_args.files_path, 'mnli_dev_mismatched.tsv'))
    pd_dataset.to_csv(os.path.join(model_args.datasets_dir, 'con_mnli_tmp.csv'), index=False)
    data_files = {"test": os.path.join(model_args.datasets_dir, 'con_mnli_tmp.csv')}
    dataset = load_dataset("csv", data_files=data_files)
    dataset.save_to_disk(os.path.join(model_args.datasets_dir, 'contrastive_mnli_mismatched'))

