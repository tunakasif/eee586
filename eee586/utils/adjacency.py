from typing import Tuple, Dict, List
import numpy as np
from numba import njit, vectorize
from numba import float64 as numba_float64
from numba import int64 as numba_int64
from numba.core import types as numba_types
from numba.typed import Dict as NumbaDict
from itertools import chain, combinations
from tqdm import tqdm, trange
from pathlib import Path

from eee586 import PKL_DIR
from eee586.word_embedding import get_token_encodings
from eee586.utils.generic import picklize

WORDS_OCC_KEY_TYPE = numba_types.int64
WORD_PAIRS_OCC_KEY_TYPE = numba_types.Tuple((numba_types.int64, numba_types.int64))
WORD_OCC_VAL_TYPE = numba_types.int64


@vectorize([numba_float64(numba_int64, numba_int64)])
def _idf(df: numba_int64, N: numba_int64) -> numba_float64:
    return np.log(N / (1 + df))


def get_tfidf_matrix(
    dataset_dir: Path,
    documents: List[List[int]],
    all_vocab: np.ndarray,
):
    tfidf_matrix_path = Path(dataset_dir, "tfidf_matrix.pkl")
    tfidf_matrix = picklize(
        _get_tfidf_matrix,
        tfidf_matrix_path,
        documents,
        all_vocab,
    )
    return tfidf_matrix


def _get_tfidf_matrix(
    documents: List[List[int]],
    all_vocab: np.ndarray,
) -> np.ndarray:
    tf_dict = NumbaDict.empty(
        key_type=WORD_PAIRS_OCC_KEY_TYPE,
        value_type=numba_types.float64,
    )

    df_dict = NumbaDict.empty(
        key_type=WORDS_OCC_KEY_TYPE,
        value_type=WORD_OCC_VAL_TYPE,
    )

    for word in tqdm(all_vocab, desc="TF-IDF", unit="word"):
        for doc_id, doc in enumerate(documents):
            tf = np.count_nonzero(doc == word) / len(doc)
            if tf > 0:
                tf_dict[(doc_id, word)] = tf
                df_dict[word] = df_dict.get(word, 0) + 1

    N = len(documents)
    tfidf = np.zeros((N, len(all_vocab)))
    word_idx = {word: idx for idx, word in enumerate(all_vocab)}
    for (doc_id, word), tf in tf_dict.items():
        df = df_dict[word]
        idf = _idf(df, N)
        curr_word_idx = word_idx[word]
        tfidf[doc_id, curr_word_idx] = tf * idf
    return tfidf


def _convert_dict_to_numba_dict(d, key_type, value_type) -> NumbaDict:
    nd = NumbaDict.empty(
        key_type=key_type,
        value_type=value_type,
    )
    for k, v in d.items():
        nd[k] = v
    return nd


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


def _get_words_occurrence(windows) -> Dict[int, int]:
    words_occurrence = {}
    for window in tqdm(windows, desc="Words Occurrence", unit="window"):
        window_words_unique = np.unique(window)
        for word in window_words_unique:
            words_occurrence[word] = words_occurrence.get(word, 0) + 1
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


def _get_word_pairs_occurrence(windows) -> Dict[Tuple[int, int], int]:
    word_pairs_occurrence = {}
    for window in tqdm(windows, desc="Words Occurrence", unit="window"):
        window_words_unique = np.unique(window)
        window_word_pairs = combinations(window_words_unique, 2)
        for pair in window_word_pairs:
            word_pairs_occurrence[pair] = word_pairs_occurrence.get(pair, 0) + 1
    return word_pairs_occurrence


@njit()
def pmi(n_i: int, n_j: int, n_ij: int, n_win: int, relu: bool = True) -> float:
    if n_i <= 0 or n_j <= 0 or n_ij <= 0 or n_win <= 0:
        pmi = 0
    else:
        pmi = np.log(n_ij * n_win / (n_i * n_j))
    result = np.maximum(pmi, 0) if relu else pmi
    return result


def get_pmi_matrix(
    dataset_dir: Path,
    documents: List[List[int]],
    window_size: int = 20,
    stride: int = 1,
) -> np.ndarray:
    windows_path = Path(dataset_dir, "windows.pkl")
    windows, _, all_vocab = picklize(
        _get_windows,
        windows_path,
        documents,
        window_size=window_size,
        stride=stride,
    )

    words_occurrence_path = Path(dataset_dir, "words_occurrence.pkl")
    words_occurrence = picklize(
        _get_words_occurrence,
        words_occurrence_path,
        windows,
    )
    word_pairs_occurrence_path = Path(dataset_dir, "word_pairs_occurrence.pkl")
    word_pairs_occurrence = picklize(
        _get_word_pairs_occurrence,
        word_pairs_occurrence_path,
        windows,
    )
    print("Converting Numba Dict")
    words_occurrence = _convert_dict_to_numba_dict(
        words_occurrence,
        key_type=WORDS_OCC_KEY_TYPE,
        value_type=WORD_OCC_VAL_TYPE,
    )
    word_pairs_occurrence = _convert_dict_to_numba_dict(
        word_pairs_occurrence,
        key_type=WORD_PAIRS_OCC_KEY_TYPE,
        value_type=WORD_OCC_VAL_TYPE,
    )

    print("Computing PMI")
    pmi_matrix_path = Path(dataset_dir, "pmi_matrix.pkl")
    pmi_matrix = picklize(
        _get_pmi_matrix,
        pmi_matrix_path,
        all_vocab,
        words_occurrence,
        word_pairs_occurrence,
        window_size,
    )

    return pmi_matrix


@njit()
def _get_pmi_matrix(
    all_vocab: np.ndarray,
    words_occurrence: Dict[int, int],
    word_pairs_occurrence: Dict[Tuple[int, int], int],
    window_size: int,
) -> np.ndarray:
    n_vocab = len(all_vocab)
    pmi_matrix = np.zeros((n_vocab, n_vocab))
    for i in range(n_vocab):
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
    return pmi_matrix


def generate_adj_matrix(
    documents: list[list[int]],
    dataset_name: str = "SetFit/20_newsgroups",
    window_size: int = 20,
    stride: int = 1,
) -> np.ndarray:
    doc_vocabs = [set(doc) for doc in documents]
    all_vocab = np.array(list(set.union(*doc_vocabs)))

    n_docs = len(documents)
    n_vocab = len(all_vocab)
    print(n_docs, n_vocab)

    dataset_dir = Path.joinpath(
        PKL_DIR,
        f"{dataset_name.replace('/', '_')}",
        f"win{window_size}_s{stride}",
    )
    Path.mkdir(dataset_dir, parents=True, exist_ok=True)
    tf_idf_matrix = get_tfidf_matrix(dataset_dir, documents, all_vocab)
    pmi_matrix = get_pmi_matrix(dataset_dir, documents, window_size, stride)

    upper_left = np.eye(n_docs)
    upper_right = tf_idf_matrix
    lower_left = tf_idf_matrix.T
    lower_right = pmi_matrix
    adj_matrix = np.block(
        [
            [upper_left, upper_right],
            [lower_left, lower_right],
        ],
    )
    return adj_matrix


def main(
    dataset_name: str = "SetFit/20_newsgroups",
    window_size: int = 20,
    stride: int = 1,
    pkl_dir: Path = PKL_DIR,
):
    dataset_dir = Path.joinpath(
        pkl_dir,
        f"{dataset_name.replace('/', '_')}",
        f"win{window_size}_s{stride}",
    )
    Path.mkdir(dataset_dir, parents=True, exist_ok=True)
    train_token_enc = get_token_encodings("train")
    documents = train_token_enc.get("input_ids")
    documents = documents[:200]

    A = generate_adj_matrix(
        documents=documents,
        dataset_name=dataset_name,
        window_size=window_size,
        stride=stride,
    )

    nz_count = np.count_nonzero(A)
    print(f"Shape, Size: {A.shape}, {A.size}")
    print(f"Non-zero count: {nz_count}/{A.size}")
    print(f"Non-zero ratio: {(A != 0).sum() / A.size * 100:.2f}%")


if __name__ == "__main__":
    main()
