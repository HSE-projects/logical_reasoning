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
        default='datasets', metadata={"help": "Where do you want to store the datasets"}
    )

def shuffle_snli(examples):
    premise = examples['premise'][:-1].split(' ')
    random.shuffle(premise)
    examples['shuffled_premise'] = ' '.join(premise) + '.'
    hypothesis = examples['hypothesis'][:-1].split(' ')
    random.shuffle(hypothesis)
    examples['shuffled_hypothesis'] = ' '.join(hypothesis) + '.'
    return examples

if __name__ == "__main__":
    parser = HfArgumentParser((PreprocessArguments,))
    model_args, = parser.parse_args_into_dataclasses()
     
    snli = load_dataset("snli", cache_dir=model_args.cache_dir)
    preprocess_snli = snli.map(shuffle_snli, batched=False)
    preprocess_snli.save_to_disk(os.path.join(model_args.datasets_dir, 'shuffle_snli'))
    
    mnli = load_dataset("multi_nli", cache_dir=model_args.cache_dir)
    preprocess_mnli = mnli.map(shuffle_snli, batched=False)
    preprocess_mnli.save_to_disk(os.path.join(model_args.datasets_dir, 'shuffle_mnli'))