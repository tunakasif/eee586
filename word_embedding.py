# %%
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict

from simpletransformers.language_representation import RepresentationModel

from eee586 import BERT_DEFAULT_MODEL_NAME, PKL_DIR
from eee586.utils.generic import picklize


def _get_input_ids_and_labels(tokenizer, dataset) -> Dict[str, List[int]]:
    texts_list = [sample["text"] for sample in dataset]
    tokenized_encoded = tokenizer.batch_encode_plus(
        tqdm(texts_list),
    )
    input_ids = tokenized_encoded["input_ids"]
    labels = [sample["label"] for sample in dataset]
    token_enc_dict = {
        "input_ids": input_ids,
        "labels": labels,
    }
    return token_enc_dict


def get_token_encodings(
    sub_dataset_name: str,  # train/test
    *,
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

    # def get_word_embeddings(
    #     *,
    #     enforce_recompute: bool = False,
    #     model_name=BERT_DEFAULT_MODEL_NAME,
    #     dataset_name="SetFit/20_newsgroups",
    # ):


def mean_across_all_tokens(token_vectors):
    return torch.mean(token_vectors, dim=1)


def batch_iterable(iterable, batch_size=1):
    l = len(iterable)
    for i in range(0, l, batch_size):
        yield iterable[i : min(i + batch_size, l)]


enforce_recompute: bool = False
model_name = BERT_DEFAULT_MODEL_NAME
dataset_name = "SetFit/20_newsgroups"
batch_size = 64

train_token_encodings = get_token_encodings(
    "train",
    enforce_recompute=enforce_recompute,
    model_name=model_name,
    dataset_name=dataset_name,
)

documents = train_token_encodings["input_ids"]
max_embed_length = max([len(doc) for doc in documents])
all_vocab = [set(doc) for doc in documents]
all_vocab = np.array(list(set.union(*all_vocab)))
all_vocab = np.sort(all_vocab)

documents = [torch.tensor([doc]) for doc in documents]
all_vocab = torch.tensor([all_vocab])

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BertModel.from_pretrained(
    model_name,
    output_hidden_states=True,
    max_position_embeddings=max_embed_length,
)
model.to(device)
model.eval()

# %%
batches = batch_iterable(documents, batch_size=batch_size)
embeddings = []
with torch.no_grad():
    for i, doc in enumerate(tqdm(documents)):
        outputs = model(doc).last_hidden_state

# %%
model = BertModel.from_pretrained(
    model_name,
    max_position_embeddings=max_embed_length,
)

# %%
batch = documents[:10]

# %%
