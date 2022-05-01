# %%
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from eee586 import BERT_MODEL_DIR, BERT_DEFAULT_MODEL_NAME
from utils import get_time


def preprocess_function(tokenizer, examples):
    return tokenizer(examples["text"], truncation=True)


def pretrain_bert_model(
    bert_model_name: str = BERT_DEFAULT_MODEL_NAME,
    dataset_name: str = "imdb",
    output_dir: str = None,
    dataset_sample_size: int = None,
):
    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    if dataset_sample_size is not None:
        r = range(dataset_sample_size)
        small_train_dataset = tokenized_dataset["train"].shuffle().select(r)
        small_eval_dataset = tokenized_dataset["test"].shuffle().select(r)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        bert_model_name, num_labels=2
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
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
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
