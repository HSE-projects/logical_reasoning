import os

if os.environ.get("OFFLINE_MODE", "false") == "true":
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
else:
    print('ONLINE MODE')

import transformers as ts
from datasets import load_metric, load_dataset, concatenate_datasets
import numpy as np
from tqdm.notebook import tqdm
import torch
from copy import deepcopy
from sklearn.model_selection import train_test_split
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


INIT_TRAIN_SIZE = "auto"
MODEL_CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "roberta-base")
NUM_QUERIES = int(os.environ.get("NUM_QUERIES", 50))
SEED = int(os.environ.get("SEED", 42))
VALID_SUBSAMPLE_SIZE = int(os.environ.get("VALID_SUBSAMPLE_SIZE", 400))
QUERY_SUBSAMPLE_SIZE = int(os.environ.get("QUERY_SUBSAMPLE_SIZE", 50))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 16))
DATASET = os.environ.get("DATASET_NAME", "snli")
PREMISE_COLUMN_NAME = os.environ.get("PREMISE_COLUMN_NAME", "premise")
HYPOTHESIS_COLUMN_NAME = os.environ.get("HYPOTHESIS_COLUMN_NAME", "hypothesis")
LABEL_COLUMN_NAME = os.environ.get("LABEL_COLUMN_NAME", "label")

TRAIN_DATASET_NAME = os.environ.get("TRAIN_DATASET_NAME", "train")
VALID_DATASET_NAME = os.environ.get("VALID_DATASET_NAME", "validation")

if __name__ == "__main__":
    if " " in DATASET:
        data = load_dataset(*DATASET.split(), cache_dir="cache/data")
    else:
        data = load_dataset(DATASET, cache_dir="cache/data")
    tokenizer = ts.AutoTokenizer.from_pretrained(
        MODEL_CHECKPOINT, cache_dir=f"cache/tokenizer/{MODEL_CHECKPOINT}"
    )
    model = ts.AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        cache_dir=f"cache/model/{MODEL_CHECKPOINT}",
        num_labels=len(set(data[TRAIN_DATASET_NAME][LABEL_COLUMN_NAME])),
    )


    def is_correct_label(instance):
        return 0 <= instance[LABEL_COLUMN_NAME] <= 2

    def to_text(instance):
        instance["text"] = instance[PREMISE_COLUMN_NAME] + tokenizer.sep_token + instance[HYPOTHESIS_COLUMN_NAME]
        return instance


    def tokenize_function(instances):
        encoding = tokenizer(instances["text"], truncation=True)
        if LABEL_COLUMN_NAME != "labels":
            encoding["labels"] = instances[LABEL_COLUMN_NAME]
        return encoding

    valid_data = (
        data[VALID_DATASET_NAME]
        .filter(is_correct_label)
        .map(to_text)
        .remove_columns([PREMISE_COLUMN_NAME, HYPOTHESIS_COLUMN_NAME])
    )
    if VALID_SUBSAMPLE_SIZE == -1:
        valid_subsample = valid_data
    else:
        subsample_idx, _ = train_test_split(
            range(len(valid_data)),
            train_size=VALID_SUBSAMPLE_SIZE,
            stratify=valid_data[LABEL_COLUMN_NAME],
            random_state=SEED,
        )
        valid_subsample = valid_data.select(subsample_idx)
    valid_subsample = valid_subsample.map(
        tokenize_function, batched=True
    ).remove_columns(list(valid_data.features.keys()))

    train_data = (
        data[TRAIN_DATASET_NAME]
        .map(to_text)
        .remove_columns([PREMISE_COLUMN_NAME, HYPOTHESIS_COLUMN_NAME])
        .map(tokenize_function, batched=True)
        .remove_columns(list(valid_data.features.keys()))
    )
    if INIT_TRAIN_SIZE == "auto":
        INIT_TRAIN_SIZE = len(set(data[TRAIN_DATASET_NAME][LABEL_COLUMN_NAME]))
    init_train_idx, _ = train_test_split(
        range(len(train_data)),
        train_size=INIT_TRAIN_SIZE,
        stratify=train_data["labels"],
        random_state=SEED,
    )
    train_sample = train_data.select(init_train_idx)
    unlabeled_data = train_data.select(
        np.setdiff1d(range(len(train_data)), init_train_idx)
    )

    data_collator = ts.DataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest"
    )

    metric = load_metric("accuracy", cache_dir="cache/metric")
    additional_metrics = [load_metric("f1", cache_dir="cache/metric")]

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        preds = logits.argmax(axis=-1)
        metrics_dict = metric.compute(predictions=preds, references=labels)
        for add_metric in additional_metrics:
            if add_metric.name == "f1":
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

    if VALID_SUBSAMPLE_SIZE > 0:
        eval_batch_size = VALID_SUBSAMPLE_SIZE
    else:
        eval_batch_size = 100
    training_args = ts.TrainingArguments(
        output_dir=f"workdir/{DATASET}_model_output_{SEED}",
        # Steps & Batch size args
        max_steps=100,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=eval_batch_size,
        # Optimizer args
        learning_rate=2e-5,
        weight_decay=0.03,
        max_grad_norm=0.3,
        # Scheduler args
        warmup_ratio=0.0,
        # Eval args
        metric_for_best_model="accuracy",
        greater_is_better=True,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        eval_steps=10,
        save_steps=10,
        save_total_limit=1,
        # WANDB args
        report_to="none",  # enable logging to W&B
        # General args
        seed=SEED,
        fp16=True,
        fp16_full_eval=False,
    )
    callbacks = [ts.EarlyStoppingCallback(early_stopping_patience=3)]

    best_metrics = []
    log.info("Starting ideal query search")
    for i_query in tqdm(range(NUM_QUERIES)):
        log.info(f"Query {i_query}")
        np.random.seed(i_query + SEED)
        subsample_idx = np.random.choice(
            range(len(unlabeled_data)), QUERY_SUBSAMPLE_SIZE, False
        )
        metrics = []

        for idx in subsample_idx:
            query = unlabeled_data.select([idx])
            train_data_with_query = concatenate_datasets(
                [train_sample, query], info=train_data.info
            )
            training_args.eval_steps = training_args.save_steps = (
                1 + i_query // training_args.per_device_train_batch_size
            ) * 5

            trainer = ts.Trainer(
                model_init=model_init,
                args=training_args,
                train_dataset=train_data_with_query,
                eval_dataset=valid_subsample,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
                data_collator=data_collator,
            )
            trainer.train()
            metrics.append(trainer.evaluate()["eval_accuracy"])

            del trainer.model
            torch.cuda.empty_cache()
            log.info(f"Checking iteration {i_query} for instance {idx}\n "
                     f"Instance with text: {tokenizer.decode(query['input_ids'][0])}\n "
                     f"Label: {query['labels'][0]}")
            log.info(f"Accuracy: {metrics[-1]}")
            gc.collect()

        query_true_idx = subsample_idx[np.argmax(metrics)]
        query = unlabeled_data.select([query_true_idx])
        train_sample = concatenate_datasets(
            [train_sample, query], info=train_data.info
        )
        unlabeled_data = unlabeled_data.select(
            np.setdiff1d(range(len(unlabeled_data)), query_true_idx)
        )
        best_metrics.append(np.max(metrics))

        log.info(
            f"Iteration {i_query};\n "
            f"Best instance text: {tokenizer.decode(query['input_ids'][0])};\n "
            f"Label: {query['labels'][0]};\n "
            f"Accuracy: {metrics[-1]}"
        )

        torch.save(
            tokenizer.batch_decode(
                unlabeled_data.select(subsample_idx)["input_ids"],
                skip_special_tokens=True,
            ),
            f"workdir/{DATASET}_candidate_query_documents_{i_query}",
        )
        torch.save(
            unlabeled_data.select(subsample_idx)["labels"],
            f"workdir/{DATASET}_candidate_query_labels_{i_query}",
        )
        torch.save(metrics, f"workdir/{DATASET}_metrics_{i_query}")

        torch.save(train_sample, f"workdir/{DATASET}_train_sample_{i_query}")
        with open(f"workdir/{DATASET}_best_metrics.yaml", "w") as f:
            for val in best_metrics:
                f.write(str(val) + "\n")

    torch.save(train_sample, f"workdir/{DATASET}_final_query")
