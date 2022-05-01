from pathlib import Path

__version__ = "0.1.0"

SRC_DIR = Path(__file__).parent.absolute()
WORK_DIR = SRC_DIR.parent
BERT_MODEL_DIR = Path.joinpath(WORK_DIR, "bert_models")
BERT_DEFAULT_MODEL_NAME = "distilbert-base-uncased"
