import time
import pickle
from pathlib import Path
import numpy as np


def pmi(
    documents: list[list[int]],
    word1_id: int,
    word2_id: int,
):
    # TODO: Implement this function
    return np.random.randint(0, 10)


def tf_idf(
    documents: list[list[int]],
    document_index: int,
    word_id: int,
):
    # TODO: Implement this function
    return np.random.randint(0, 10)


def generate_adj_matrix(documents: list[list[int]]) -> np.ndarray:
    doc_vocabs = [set(doc) for doc in documents]
    all_vocab = list(set.union(*doc_vocabs))

    n_docs = len(documents)
    n_vocab = len(all_vocab)

    tf_idf_matrix = np.zeros((n_docs, n_vocab))
    for i in range(n_docs):
        for j, word in enumerate(all_vocab):
            tf_idf_matrix[i, j] = tf_idf(
                documents,
                i,
                word,
            )

    pmi_matrix = np.zeros((n_vocab, n_vocab))
    for i, word1 in enumerate(all_vocab):
        for j in range(i + 1, n_vocab):
            word2 = all_vocab[j]
            pmi_matrix[i, j] = pmi_matrix[j, i] = pmi(
                documents,
                word1,
                word2,
            )
    pmi_matrix = pmi_matrix + np.eye(pmi_matrix.shape[0])

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


def get_time(format: str = "%Y_%m_%d_%H_%M"):
    return time.strftime(format)


def pickle_dump(obj: object, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(path: Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


def picklize(func, path: Path, *args, enforce: bool = False, **kwargs):
    if enforce or not path.exists():
        result = func(*args, **kwargs)
        pickle_dump(result, path)
    else:
        result = pickle_load(path)
    return result
