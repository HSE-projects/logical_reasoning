import os
if os.environ.get("OFFLINE_MODE", "false") == "true":
    print('Turning on offline mode')
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import transformers as ts
from datasets import load_metric, load_dataset, concatenate_datasets
import numpy as np
from tqdm.notebook import tqdm
import torch
from copy import deepcopy
import gc
import logging
import sys

GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", -1)

# Set logger
if not os.path.exists("workdir"):
    os.mkdir("workdir")
logging.basicConfig(filename=f'workdir/log_{GPU_ID}.txt', level=logging.INFO)
fileh = logging.FileHandler(f'workdir/log_{GPU_ID}.txt', 'a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fileh.setFormatter(formatter)

log = logging.getLogger()  # root logger
for hndlr in log.handlers[:]:  # remove all old handlers
    log.removeHandler(hndlr)
log.addHandler(fileh)

INIT_TRAIN_SIZE = 2
NUM_LABELS = 3
MODEL_CHECKPOINT = "roberta-base"
NUM_QUERIES = int(os.environ.get("NUM_QUERIES", 100))
SEED = int(os.environ.get("SEED", 42))
VALID_SUBSAMPLE_SIZE = int(os.environ.get("VALID_SUBSAMPLE_SIZE", 50))
QUERY_SUBSAMPLE_SIZE = int(os.environ.get("QUERY_SUBSAMPLE_SIZE", 100))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 16))
DATASET = os.environ.get("DATASET_NAME", "snli")
PREMISE_COLUMN_NAME = os.environ.get("PREMISE_COLUMN_NAME", "premise")
HYPOTHESIS_COLUMN_NAME = os.environ.get("HYPOTHESIS_COLUMN_NAME", "hypothesis")
LABEL_COLUMN_NAME = os.environ.get("LABEL_COLUMN_NAME", "label")

TRAIN_DATASET_NAME = os.environ.get("TRAIN_DATASET_NAME", "train")
VALID_DATASET_NAME = os.environ.get("VALID_DATASET_NAME", "validation")

if __name__ == "__main__":
    log.info("Started")
    data = load_dataset(DATASET, cache_dir="cache/data")
    tokenizer = ts.AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, cache_dir=f"cache/tokenizer/{MODEL_CHECKPOINT}")
    model = ts.AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, cache_dir=f"cache/model/{MODEL_CHECKPOINT}", num_labels=NUM_LABELS)
    log.info("Downloaded data, tokenizer and model")

    def is_correct_label(instance):
        return 0 <= instance[LABEL_COLUMN_NAME] <= 2

    def to_text(instance):
        instance["text"] = instance[PREMISE_COLUMN_NAME] + tokenizer.sep_token + instance[HYPOTHESIS_COLUMN_NAME]
        del instance[PREMISE_COLUMN_NAME]
        del instance[HYPOTHESIS_COLUMN_NAME]
        return instance

    def tokenize_fn(instances):
        encoded = tokenizer(instances["text"], truncation=True)
        encoded["labels"] = instances[LABEL_COLUMN_NAME]
        return encoded

    valid_data = data[VALID_DATASET_NAME].filter(is_correct_label).map(to_text)
    np.random.seed(SEED)
    subsample_idx = np.random.choice(range(len(valid_data)), VALID_SUBSAMPLE_SIZE, False)
    valid_subsample = valid_data.select(subsample_idx).map(tokenize_fn, batched=True).remove_columns(
        list(valid_data.features.keys()))

    train_data = data[TRAIN_DATASET_NAME].filter(is_correct_label).map(to_text).map(tokenize_fn, batched=True).remove_columns(list(valid_data.features.keys()))
    np.random.seed(SEED)
    init_train_idx = np.random.choice(range(len(train_data)), INIT_TRAIN_SIZE)
    train_sample = train_data.select(init_train_idx)
    unlabeled_data = train_data.select(np.setdiff1d(range(len(train_data)), init_train_idx))

    data_collator = ts.DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    log.info("Created all data stuff")

    metric = load_metric("accuracy", cache_dir="cache/metric")
    additional_metrics = [load_metric('f1', cache_dir="cache/metric")]

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        preds = logits.argmax(axis=-1)
        metrics_dict = metric.compute(predictions=preds, references=labels)
        for add_metric in additional_metrics:
            if add_metric.name == "f1" and NUM_LABELS > 2:
                for average in ["micro", "macro", "weighted"]:
                    add_metric_dict = add_metric.compute(
                        predictions=preds, references=labels, average=average
                    )
                    metrics_dict.update({f"f1_{average}": add_metric_dict["f1"]})
            else:
                add_metric_dict = add_metric.compute(
                    predictions=preds, references=labels
                )
                metrics_dict.update(add_metric_dict)

        return metrics_dict

    def model_init():
        return deepcopy(model)

    training_args = ts.TrainingArguments(
        output_dir=f'workdir/{DATASET}_model_output_{SEED}',
        # Steps & Batch size args
        max_steps=200,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=VALID_SUBSAMPLE_SIZE,
        # Optimizer args
        learning_rate=2e-5,
        weight_decay=0.03,
        max_grad_norm=0.3,
        # Scheduler args
        warmup_ratio=0.,
        # Eval args
        metric_for_best_model='accuracy',
        greater_is_better=True,
        load_best_model_at_end=True,
        evaluation_strategy='steps',
        logging_strategy='steps',
        save_strategy='steps',
        eval_steps=10,
        save_steps=10,
        save_total_limit=1,
        # WANDB args
        report_to="none",  # enable logging to W&B
        # General args
        seed=SEED,
    )
    callbacks = [ts.EarlyStoppingCallback(early_stopping_patience=3)]

    best_metrics = []
    log.info("Starting ideal query search")
    for i_query in tqdm(range(NUM_QUERIES)):
        log.info(f"Query {i_query}")
        np.random.seed(i_query + SEED)
        subsample_idx = np.random.choice(range(len(unlabeled_data)), QUERY_SUBSAMPLE_SIZE, False)

        for idx in subsample_idx:
            metrics = []
            query = unlabeled_data.select([idx])
            train_data_with_query = concatenate_datasets([train_sample, query], info=train_data.info)
            training_args.eval_steps = training_args.save_steps = (1 + i_query // training_args.per_device_train_batch_size) * 5

            trainer = ts.Trainer(
                model_init=model_init,
                args=training_args,
                train_dataset=train_data_with_query,
                eval_dataset=valid_subsample,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
                data_collator=data_collator
            )
            trainer.train()
            metrics.append(trainer.evaluate()['eval_accuracy'])

            del trainer.model
            torch.cuda.empty_cache()
            log.info(f"Checking iteration {i_query} for instance {idx}\n "
                     f"Instance with text: {tokenizer.decode(query['input_ids'][0])}\n "
                     f"Label: {query['labels'][0]}")
            log.info(f"Accuracy: {metrics[-1]}")
            gc.collect()

        query_true_idx = subsample_idx[np.argmax(metrics)]
        query = unlabeled_data.select([query_true_idx])
        train_sample = concatenate_datasets([train_sample, query], info=train_data.info)
        unlabeled_data = unlabeled_data.select(np.setdiff1d(range(len(unlabeled_data)), query_true_idx))
        best_metrics.append(np.max(metrics))

        log.info(
            f"Iteration {i_query};\n "
            f"Best instance text: {tokenizer.decode(query['input_ids'][0])};\n "
            f"Label: {query['labels'][0]};\n "
            f"Accuracy: {metrics[-1]}"
        )

        torch.save(train_sample, f"workdir/{DATASET}_train_sample_{i_query}")
        with open(f"workdir/{DATASET}_best_metrics.yaml", "w") as f:
            for val in metrics:
                f.write(str(val) + "\n")

    torch.save(train_sample, f"workdir/{DATASET}_final_query")
