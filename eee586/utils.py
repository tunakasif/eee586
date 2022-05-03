import time
import pickle
from pathlib import Path
import numpy as np
import itertools


def pmi(
    w1_id: int,
    w2_id: int,
    documents: list[list[int]],
    window_lenght: int = 10,
    stride: int = 1,
):
    corpus = list(itertools.chain(*documents))
    p_i, p_j, p_ij = 0, 0, 0
    total_num_of_windows = (len(corpus) - window_lenght + 1) / stride

    if w1_id == w2_id:
        pmi_score = 1
    else:
        for i in range(0, len(corpus) - window_lenght, stride):
            if w1_id in corpus[i : i + window_lenght]:
                p_i += 1
            if w2_id in corpus[i : i + window_lenght]:
                p_j += 1
            if (
                w1_id in corpus[i : i + window_lenght]
                and w2_id in corpus[i : i + window_lenght]
            ):
                p_ij += 1
        pmi_score = np.log(p_ij / (p_i * p_j * total_num_of_windows))
    return pmi_score


def tf_idf(
    documents: list[list[int]],
    document_index: int,
    word_id: int,
):
    """
    t — term (word)
    d — document (set of words)
    N — count of corpus
    corpus — the total document set
    """
    corpus = np.array(list(itertools.chain(*documents)))

    d = np.array(documents[document_index])
    tf = np.count_nonzero(d == word_id) / len(d)
    df = np.count_nonzero(corpus == word_id) / len(d)
    idf = np.log(len(documents) / (df + 1))
    tf_idf_score = tf * idf
    return tf_idf_score


def generate_adj_matrix(
    documents: list[list[int]],
    window_lenght: int = 10,
    stride: int = 1,
) -> np.ndarray:
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
                window_lenght=window_lenght,
                stride=stride,
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
