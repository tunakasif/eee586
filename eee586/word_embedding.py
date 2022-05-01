# %%
from transformers import BertTokenizer, BertModel
from pathlib import Path

from eee586 import BERT_DEFAULT_MODEL_NAME


def embed_sentence(sentence, model_name=BERT_DEFAULT_MODEL_NAME):
    tokenizer = BertTokenizer.from_pretrained(BERT_DEFAULT_MODEL_NAME)
    model = BertModel.from_pretrained(BERT_DEFAULT_MODEL_NAME)
    encoded_input = tokenizer.encode(sentence, return_tensors="pt")
    output = model(encoded_input)
    return output
