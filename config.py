import torch
import os

BATCH_SIZE = 8
NUM_WORKERS = 4

DEVICE = "cuda:0" if torch.cuda.is_available() else 'cpu'

MODEL_ID = "Salesforce/blip-vqa-base"
MAX_LENGTH=512 # context window
# Whether also input is outputted or not
SKIP_INPUT = False
IMG_SIZE = (384, 384)

# #MODEL_ID = "google/paligemma-3b-mix-224"
# MAX_LENGTH=256
# # # Whether also input is outputted or not
# SKIP_INPUT = True
# IMG_SIZE = (224, 224)

TRAINED = False
HF_TOKEN = "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"

DATA_ROOT = "/teamspace/uploads"
JSON_FILE = "./data/HighCountVQA_combined.json"

EVAL_DATA_PATH = './data/evaluation'
PLOT_DATA_PATH = './data/plots'

MAX_OBJ_NUMBER = 15

# List of number words from 0 to 25
number_words = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "twenty-one", "twenty-two", "twenty-three", "twenty-four", "twenty-five"
]

NUM_DICT = {word: i for i, word in enumerate(number_words)}


# Training
TRAIN_PATH="/teamspace/studios/this_studio/data/HighCountVQA_train.json"
VAL_PATH="/teamspace/studios/this_studio/data/HighCountVQA_val.json"
TEST_PATH="/teamspace/studios/this_studio/data/HighCountVQA_test.json"

OUTPUT_DIR=os.path.join("/teamspace/studios/this_studio/checkpoints", MODEL_ID)
BEST_CHECKPOINT=os.path.join(OUTPUT_DIR, 'best_checkpoint')
EPOCHS=15

# following are default huggingface training parameter
# AdamW optimizer is used with default parameters
LR=5e-05
LR_SCHEDULER='linear'
WARMUP_STEPS=0
WARMUP_RATIO=0

GRADIENT_ACCUMULATION = 8 // BATCH_SIZE # we used batch_size 8 for BLIP

EARLY_STOPPING=True
PATIENCE=3
