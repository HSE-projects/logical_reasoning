# this file should be used to train cline on existing dataset
import random
from dataclasses import dataclass, field
from typing import Optional

import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from datasets import Dataset
import datasets
from transformers import AutoTokenizer

from transformers import (
    HfArgumentParser,
    PreTrainedTokenizer
)

from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)

@dataclass
class PreprocessArguments:
    model_name: Optional[str] = field(
        default='roberta-base', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default='.cache', metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    dataset_file: Optional[str] = field(
        default='datasets/cline_snli', metadata={"help": "Where do you want preprocessed dataset"}
    )
    output_dir: Optional[str] = field(
        default='models/robeta_base_cline_snli', metadata={"help": "Where do you want preprocessed dataset"}
    )
        
        
from datacollator import DataCollatorForLEC
from tokenizer import LecbertTokenizer
from model import LecbertForPreTraining, LecbertConfig

if __name__ == "__main__":
    parser = HfArgumentParser((PreprocessArguments,))
    model_args, = parser.parse_args_into_dataclasses()
    
    dataset = datasets.load_from_disk(model_args.dataset_file)
    set_seed(42)
    tokenizer = LecbertTokenizer.from_pretrained(model_args.model_name, cache_dir=model_args.cache_dir)
    data_collator = DataCollatorForLEC(tokenizer=tokenizer)
    
    config = LecbertConfig.from_pretrained('roberta-base', cache_dir='.cache')

    model = LecbertForPreTraining.from_pretrained('roberta-base', config=config, cache_dir='.cache')
    
    small_train_dataset = dataset["train"].shuffle(seed=42)
    small_eval_dataset = dataset["test"].shuffle(seed=42)
    
    training_args = TrainingArguments(
        output_dir=model_args.output_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy='epoch',
        num_train_epochs=5,
        save_total_limit=1,
        load_best_model_at_end=True,
        seed=42,
        learning_rate=2e-5,
        weight_decay=0.01,  # strength of weight decay
        max_grad_norm=1.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()