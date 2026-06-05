from pathlib import Path

ROOT_DIR= Path(__file__).parent.parent
RAW_DATA_DIR = ROOT_DIR / "data"/ "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data"/ "process"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

SEQ_LEN=128
BATCH_SIZE=64
EMBEDDING_DIM=128
# # 一般要比EMBEDDING_DIM大一些，至少要和EMBEDDING_DIM一样大
HIDDEN_SIZE=256
LEARNING_RATE=1e-3
EPOCHS=10