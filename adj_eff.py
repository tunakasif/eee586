# %%
from typing import Tuple, Dict, List
import numpy as np
from numba import njit
from itertools import chain, combinations
from tqdm import tqdm, trange
from pathlib import Path

from eee586 import PKL_DIR
from eee586.word_embedding import get_token_encodings
from eee586.utils import picklize


def _get_windows(
    documents: List[List[int]],
    window_size: int = 20,
    stride: int = 1,
):
    corpus = np.array(list(chain(*documents)))
    N = len(corpus)
    try:
        assert stride < window_size
        assert np.mod(N - window_size, stride) == 0
    except AssertionError as e:
        print("Window size and stride is not compatible with the corpus size")
        raise e

    doc_vocabs = [set(doc) for doc in documents]
    all_vocab = np.array(list(set.union(*doc_vocabs)))
    all_vocab = np.sort(all_vocab)

    windows = (
        np.array(
            [corpus[i : i + window_size] for i in range(0, N - window_size + 1, stride)]
        )
        if window_size < N
        else np.array([corpus])
    )
    return windows, corpus, all_vocab


def _get_words_occurrence(windows) -> Dict[int, int]:
    words_occurrence = {}
    for window in tqdm(windows):
        window_words_unique = np.unique(window)
        for word in window_words_unique:
            words_occurrence[word] = words_occurrence.get(word, 0) + 1
    return words_occurrence


def _get_word_pairs_occurrence(windows) -> Dict[Tuple[int, int], int]:
    word_pairs_occurrence = {}
    for window in tqdm(windows):
        window_words_unique = np.unique(window)
        window_word_pairs = combinations(window_words_unique, 2)
        for pair in window_word_pairs:
            word_pairs_occurrence[pair] = word_pairs_occurrence.get(pair, 0) + 1
    return word_pairs_occurrence


def get_words_occurrence(
    dataset_dir: Path,
    documents: List[List[int]],
    window_size: int = 20,
    stride: int = 1,
) -> Dict[int, int]:
    windows_path = Path(dataset_dir, "windows.pkl")
    windows, *_ = picklize(
        _get_windows,
        windows_path,
        documents,
        window_size=window_size,
        stride=stride,
    )

    word_occurrence_path = Path(dataset_dir, "words_occurrence.pkl")
    words_occurrence = picklize(
        _get_words_occurrence,
        word_occurrence_path,
        windows,
    )

    return words_occurrence


def get_word_pairs_occurrence(
    dataset_dir: Path,
    documents: List[List[int]],
    window_size: int = 3,
    stride: int = 1,
) -> Dict[Tuple[int, int], int]:
    windows_path = Path(dataset_dir, "windows.pkl")
    windows, *_ = picklize(
        _get_windows,
        windows_path,
        documents,
        window_size=window_size,
        stride=stride,
    )

    word_pairs_occurrence_path = Path(dataset_dir, "word_pairs_occurrence.pkl")
    word_pairs_occurrence = picklize(
        _get_word_pairs_occurrence,
        word_pairs_occurrence_path,
        windows,
    )

    return word_pairs_occurrence


# %%
# if __name__ == "__main__":
window_size = 20
stride = 1
dataset_name = "SetFit/20_newsgroups"
f"{dataset_name.replace('/', '_')}",
DATASET_DIR = Path.joinpath(
    PKL_DIR,
    f"{dataset_name.replace('/', '_')}",
    f"win{window_size}_s{stride}",
)
Path.mkdir(DATASET_DIR, parents=True, exist_ok=True)

# documents = [
#     [1, 3, 2, 1],
#     [4, 5, 1, 6, 6],
#     [3, 2, 7],
# ]
train_token_enc = get_token_encodings("train")
documents = train_token_enc.get("input_ids")
# documents = documents[:1000]

# %%
windows, corpus, all_vocab = _get_windows(
    documents,
    window_size=window_size,
    stride=stride,
)

words_occurrence = get_words_occurrence(
    DATASET_DIR,
    documents,
    window_size,
    stride,
)

# %%
word_pairs_occurrence = get_word_pairs_occurrence(
    DATASET_DIR,
    documents,
    window_size,
    stride,
)
word_pairs_occurrence


@njit()
def pmi(n_i: int, n_j: int, n_ij: int, n_win: int, relu: bool = True) -> float:
    if n_i <= 0 or n_j <= 0 or n_ij <= 0 or n_win <= 0:
        pmi = 0
    else:
        pmi = np.log(n_ij * n_win / (n_i * n_j))
    result = np.maximum(pmi, 0) if relu else pmi
    return result


# %%
n_vocab = len(all_vocab)
pmi_matrix = np.zeros((n_vocab, n_vocab))
for i in trange(n_vocab):
    for j in range(i + 1, n_vocab):
        word1 = all_vocab[i]
        word2 = all_vocab[j]

        n_i = words_occurrence.get(word1, 0)
        n_j = words_occurrence.get(word2, 0)
        n_ij = word_pairs_occurrence.get((word1, word2), 0)

        pmi_matrix[i, j] = pmi(
            n_i=n_i,
            n_j=n_j,
            n_ij=n_ij,
            n_win=window_size,
            relu=True,
        )
pmi_matrix = pmi_matrix + pmi_matrix.T + np.eye(pmi_matrix.shape[0])
pmi_matrix

# %%
