import torch

BATCH_SIZE = 16
NUM_WORKERS = 4

DEVICE = "cuda:0" if torch.cuda.is_available() else 'cpu'
MODEL_ID = "Salesforce/blip-vqa-base"
IMG_SIZE = (384, 384) # for BLIP

DATA_ROOT = "/teamspace/uploads"
JSON_FILE = "./data/HighCountVQA_val.json"

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