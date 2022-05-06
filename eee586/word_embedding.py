from transformers import BertTokenizer
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict
import nltk
from nltk.corpus import stopwords
from itertools import chain

from eee586 import BERT_DEFAULT_MODEL_NAME, PKL_DIR
from eee586.utils.generic import picklize


def _remove_stopwords(
    tokenizer: BertTokenizer,
    input_ids: List[List[int]],
    labels: List[int],
) -> List[List[int]]:
    nltk.download("stopwords")
    stop_words = list(set(stopwords.words("english")))
    stop_tokenized_endcoded = tokenizer.batch_encode_plus(stop_words)
    stop_input_ids = stop_tokenized_endcoded["input_ids"]
    stop_input_ids = list(chain(*stop_input_ids))
    input_ids = [
        [j for j in ii if j not in stop_input_ids]
        for ii in tqdm(input_ids, desc="Removing stopwords")
    ]

    empty_idx = [i for i, ii in enumerate(input_ids) if len(ii) == 0]
    input_ids = [ii for i, ii in enumerate(input_ids) if i not in empty_idx]
    labels = [labels[i] for i in range(len(labels)) if i not in empty_idx]
    return input_ids, labels


def _get_input_ids_and_labels(
    tokenizer,
    dataset,
    remove_stop: bool = True,
) -> Dict[str, List[int]]:
    texts_list = [sample["text"] for sample in dataset]
    tokenized_encoded = tokenizer.batch_encode_plus(
        tqdm(texts_list),
    )
    input_ids = tokenized_encoded["input_ids"]
    labels = [sample["label"] for sample in dataset]
    if remove_stop:
        input_ids, labels = _remove_stopwords(tokenizer, input_ids, labels)
    token_enc_dict = {
        "input_ids": input_ids,
        "labels": labels,
    }
    return token_enc_dict


def get_token_encodings(
    sub_dataset_name: str,  # train/test
    *,
    remove_stopword: bool = True,
    enforce_recompute: bool = False,
    model_name=BERT_DEFAULT_MODEL_NAME,
    dataset_name="SetFit/20_newsgroups",
):
    if not sub_dataset_name in ["train", "test"]:
        raise ValueError("sub_dataset must be either 'train' or 'test'")

    pkl_enc_path = Path.joinpath(
        PKL_DIR,
        f"{dataset_name.replace('/', '_')}",
        BERT_DEFAULT_MODEL_NAME,
    )
    if remove_stopword:
        pkl_enc_path = pkl_enc_path.joinpath("wo_stop")
    Path.mkdir(pkl_enc_path, exist_ok=True, parents=True)
    sub_pkl_enc_path = Path.joinpath(pkl_enc_path, f"{sub_dataset_name}_token_enc.pkl")
    if sub_pkl_enc_path.exists() and not enforce_recompute:
        sub_token_enc_dict = picklize(
            None,
            sub_pkl_enc_path,
        )
    else:
        dataset = load_dataset(dataset_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)

        sub_dataset = dataset.get(f"{sub_dataset_name}")
        sub_token_enc_dict = picklize(
            _get_input_ids_and_labels,
            sub_pkl_enc_path,
            tokenizer,
            sub_dataset,
            enforce=enforce_recompute,
        )
    return sub_token_enc_dict
