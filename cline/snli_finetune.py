import os
from dataclasses import dataclass, field
from typing import Optional
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

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

def preprocess_snli(examples):
    x = tokenizer(examples["hypothesis"], examples["premise"], truncation=True, max_length=30)
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
    cache_dir: Optional[str] = field(
        default='/home/vapavlov_4/.cache', metadata={"help": "Cache dir"}
    )
    output_dir: Optional[str] = field(
        default='exps/exp0', metadata={"help": "Where do you want to store results"}
    )
    batch_size: Optional[int] = field(
        default=32, metadata={"help": "Per device batch size"}
    )
    eval_batch_size: Optional[int] = field(
        default=128, metadata={"help": "Per device batch size"}
    )
    exp_name: Optional[str] = field(
        default='exp0', metadata={"help": "Experiment name for wandb log"}
    )

class TrainMetricsCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

        
if __name__ == "__main__":
    parser = HfArgumentParser((PreprocessArguments,))
    model_args, = parser.parse_args_into_dataclasses()
    name = model_args.model_name
    
    snli = load_dataset("snli", cache_dir=model_args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=model_args.cache_dir)
    tokenized_snli = snli.map(preprocess_snli, batched=True).map(fix_negative, batched=True, remove_columns=["premise", "hypothesis"])
    metric = load_metric("accuracy")
    set_seed(42)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_path, num_labels=3, cache_dir=model_args.cache_dir)

    training_args = TrainingArguments(
        output_dir=os.path.join(model_args.output_dir, 'snli'),
        per_device_train_batch_size=model_args.batch_size,
        per_device_eval_batch_size=model_args.eval_batch_size,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy='epoch',
        num_train_epochs=7,
        save_total_limit=1,
        load_best_model_at_end=True,
        seed=42,
        learning_rate=2e-5,
        weight_decay=0.01,  # strength of weight decay
        max_grad_norm=1.0,
        report_to="wandb",
        run_name=model_args.exp_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_snli["train"],
        eval_dataset=tokenized_snli["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.add_callback(TrainMetricsCallback(trainer)) 
    trainer.train()
    
    res = trainer.predict(tokenized_snli["test"])
    if trainer.is_world_process_zero():
        with open(os.path.join(model_args.output_dir, 'results.txt'), 'w') as f:
            print(res.metrics['test_accuracy'], file=f)
        trainer.save_model(os.path.join(model_args.output_dir, 'final_model'))
