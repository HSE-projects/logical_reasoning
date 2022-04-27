import os
import sys
print(sys.prefix)
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback
)


from transformers import (
    HfArgumentParser,
    PreTrainedTokenizer
)

def preprocess_shuffled(examples):
    x = tokenizer(examples["shuffled_hypothesis"], examples["shuffled_premise"])
    return x

import numpy as np
from datasets import load_metric
import numpy as np
import torch.nn as nn
import torch

def fix_negative(examples):
    narr = np.array(examples['label'])
    narr[narr < 0] *= -1
    examples['label'] = narr
    return examples

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

@dataclass
class PreprocessArguments:
    model_name: Optional[str] = field(
        default='roberta-base', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    model_path: Optional[str] = field(
        default='roberta-base', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    datasets_dir: Optional[str] = field(
        default='datasets', metadata={"help": "Where do you want to store the datasets"}
    )
    cache_dir: Optional[str] = field(
        default='/home/vapavlov_4/.cache', metadata={"help": "Cache dir"}
    )
    output_dir: Optional[str] = field(
        default='exps/exp0', metadata={"help": "Where do you want to store results"}
    )
    eval_batch_size: Optional[int] = field(
        default=128, metadata={"help": "Per device batch size"}
    )

import torch
        
if __name__ == "__main__":
    n_gpus = max(1, torch.cuda.device_count())
    print("Found {} GPUs".format(n_gpus))
    parser = HfArgumentParser((PreprocessArguments,))
    model_args, = parser.parse_args_into_dataclasses()
    name = model_args.model_name
    
    metric = load_metric("accuracy")
    set_seed(42)

    for task_name in ['snli', 'mnli']:
        tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=model_args.cache_dir)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        if model_args.task_name == 'snli':
            snli = datasets.load_from_disk(os.path.join(model_args.datasets_dir, 'shuffled_snli'))
            tokenized_snli = snli.map(preprocess_snli, batched=True).map(fix_negative, batched=True, remove_columns=["shuffled_premise", "shuffled_hypothesis", "premise", "hypothesis"])
            test_datasets = [("test", tokenized_snli["test"])]
        if model_args.task_name == 'mnli':
            mnli = load_dataset("multi_nli", cache_dir=model_args.cache_dir)
            # not typo in preprocess_snli, because they both have hypothesis and premise
            tokenized_mnli = mnli.map(preprocess_snli, batched=True).map(fix_negative, batched=True, remove_columns=["shuffled_premise", "shuffled_hypothesis", 'promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse', 'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre'])
            test_datasets = [("validation_matched", tokenized_mnli["validation_matched"]), ("validation_mismatched", tokenized_mnli["validation_mismatched"])]

        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_args.output_dir, '{}_{}_final_model'.format(task_name, test_datasets[0][0])), num_labels=3, cache_dir=model_args.cache_dir)
        
        training_args = TrainingArguments(
            output_dir=os.path.join(model_args.output_dir, 'human_eval_logs'.format(model_args.task_name)),
            per_device_train_batch_size=model_args.batch_size // n_gpus,
            per_device_eval_batch_size=model_args.eval_batch_size // n_gpus,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy='epoch',
            seed=42,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        with open(os.path.join(model_args.output_dir, 'human_results.txt'.format(model_args.task_name)), 'a') as f:
            for test_name, test_dataset in test_datasets:
                res = trainer.predict(test_dataset)
                if trainer.is_world_process_zero():
                    print(test_name, res.metrics['test_accuracy'], file=f)
