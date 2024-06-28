import torch

BATCH_SIZE = 16
NUM_WORKERS = 4

DEVICE = "cuda:0" if torch.cuda.is_available() else 'cpu'
MODEL_ID = "Salesforce/blip-vqa-base"
IMG_SIZE = (384, 384) # for BLIP

DATA_ROOT = "/teamspace/uploads"
JSON_FILE = "./data/HighCountVQA_val.json"

MAX_OBJ_NUMBER = 15