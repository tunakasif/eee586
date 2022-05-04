from pathlib import Path
from typing import Union, Tuple
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from eee586 import BERT_MODEL_DIR, BERT_DEFAULT_MODEL_NAME
from eee586.utils.generic import get_time


def get_num_label(dataset: DatasetDict) -> int:
    train_dataset = dataset.get("train")
    try:
        labels = [sample["label"] for sample in train_dataset]
    except KeyError as e:
        print(f"Dataset {dataset} does not have label.")
        raise e
    labels_set = set(labels)
    return len(labels_set)


def pretrain_bert_model(
    bert_model_name: str = BERT_DEFAULT_MODEL_NAME,
    dataset_name: Union[str, Tuple[str]] = "imdb",
    output_dir: str = None,
    dataset_sample_size: int = None,
):
    if type(dataset_name) is tuple:
        dataset = load_dataset(*dataset_name)
    else:
        dataset = load_dataset(dataset_name)
    num_labels = get_num_label(dataset)
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True), batched=True
    )
    if dataset_sample_size is not None:
        r = range(dataset_sample_size)
        small_train_dataset = tokenized_dataset["train"].shuffle().select(r)
        small_eval_dataset = tokenized_dataset["test"].shuffle().select(r)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        bert_model_name, num_labels=num_labels
    )

    if output_dir is None:
        output_dir = Path.joinpath(
            BERT_MODEL_DIR,
            bert_model_name,
            get_time(),
        )
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
