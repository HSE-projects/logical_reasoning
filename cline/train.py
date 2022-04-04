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
        default='/home/vapavlov_4/.cache', metadata={"help": "Cache dir"}
    )
    dataset_file: Optional[str] = field(
        default='/home/vapavlov_4/datasets/cline_snli', metadata={"help": "Where is the dataset"}
    )
    output_dir: Optional[str] = field(
        default='/home/vapavlov_4/models/robeta_base_cline_snli', metadata={"help": "Where do you want to store model"}
    )
    mlm_layer6: Optional[bool] = field(
        default=False, metadata={"help": "Use mlm loss on layer 6 instead of last layer"}
    )
    tec_layer6: Optional[bool] = field(
        default=False, metadata={"help": "Use tec loss on layer 6 instead of last layer"}
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
        
        
from datacollator import DataCollatorForLEC
from tokenizer import LecbertTokenizer
from model import LecbertForPreTraining, LecbertConfig

if __name__ == "__main__":
    print("START")
    parser = HfArgumentParser((PreprocessArguments,))
    model_args, = parser.parse_args_into_dataclasses()
    
    dataset = datasets.load_from_disk(model_args.dataset_file)
    set_seed(42)
    tokenizer = LecbertTokenizer.from_pretrained(model_args.model_name, cache_dir=model_args.cache_dir)
    data_collator = DataCollatorForLEC(tokenizer=tokenizer)
    
    config = LecbertConfig.from_pretrained(model_args.model_name, cache_dir=model_args.cache_dir)
    config.mlm_layer6 = model_args.mlm_layer6
    config.tec_layer6 = model_args.tec_layer6

    model = LecbertForPreTraining.from_pretrained(model_args.model_name, config=config, cache_dir=model_args.cache_dir)
    
    train_dataset = dataset["train"].shuffle(seed=42)
    eval_dataset = dataset["validation"].shuffle(seed=42)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(model_args.output_dir, 'cline'),
        per_device_train_batch_size=model_args.batch_size,
        per_device_eval_batch_size=model_args.eval_batch_size,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy='epoch',
        num_train_epochs=20,
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    if trainer.is_world_process_zero():
        trainer.save_model(os.path.join(model_args.output_dir, 'best_cline'))
