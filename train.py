import torch
import dataset
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering, AutoModel, PaliGemmaForConditionalGeneration

from dataset import get_dataset

from tqdm import tqdm