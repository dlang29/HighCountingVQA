from tqdm import tqdm
import numpy as np
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import transformers
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVisualQuestionAnswering

from dataset import HighCountVQADataset
import config
from plots import create_bar_plots, create_dual_axis_bar_plots


# List of number words from 0 to 25
number_words = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "twenty-one", "twenty-two", "twenty-three", "twenty-four", "twenty-five"
]

NUM_DICT = {word: i for i, word in enumerate(number_words)}

# declare all necessary components (here just for inference)
transform = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor()
])
test_set = HighCountVQADataset(data_root=config.DATA_ROOT,
                               json_file=config.JSON_FILE,
                               transform=transform)
test_loader = DataLoader(test_set,
                         batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)


# Blip specific loading

# processor instead of tokenizer -> contains BERT tokenizer + BLIP image processor
processor = AutoProcessor.from_pretrained(config.MODEL_ID, do_rescale=False) #do_rescale false, because ToTensor() already does that
model = AutoModelForVisualQuestionAnswering.from_pretrained(config.MODEL_ID).to(config.DEVICE)



total_iterations = int(len(test_set) / config.BATCH_SIZE)

# data structures for tracking accuracies
total_count = 0

# track the total occurence of each number of objects
bin_count = np.zeros(config.MAX_OBJ_NUMBER)
# track the true predictions for each number of objects
true_count = np.zeros(config.MAX_OBJ_NUMBER)
# track the absolute error for each number of objects (if possible)
total_abs_err = np.zeros(config.MAX_OBJ_NUMBER)
# track amount of NaN outputs (not int casting possible)
nan_count = np.zeros(config.MAX_OBJ_NUMBER)


with torch.no_grad():
  for idx, data in enumerate(tqdm(test_loader, total=total_iterations)):
    questions = data['question']
    imgs = data['image']
    answers = np.array(data['answer'])

    inputs = processor(imgs, questions, padding=True, return_tensors="pt").to(config.DEVICE)
    # "generate" is only for inference to get the full answer instead of just the next token
    outputs = model.generate(**inputs)
    predictions = processor.batch_decode(outputs, skip_special_tokens=True)
    # convert written numbers to there strings "one"->"1" and map nonsense outputs to np.inf
    predictions = [pred if pred.isdigit() else NUM_DICT.get(pred, "-1") for pred in predictions]
    predictions = np.array(predictions, dtype=np.int32)

    # update accuracy counts
    true_pred = (predictions == answers)

    total_count += true_pred.sum()

    # add 1 to each ground truth occurence
    np.add.at(bin_count, answers - 1, 1)
    # add 1 to each correct count of those
    np.add.at(true_count, answers[true_pred] - 1, 1)
    # add the absolute errors to the bins
    valid_idx = np.where(predictions != -1)[0]
    np.add.at(total_abs_err, answers[valid_idx] - 1, abs(predictions[valid_idx] - answers[valid_idx]))
    # add amount of nan outputs
    invalid_idx = np.ones(predictions.shape[0], dtype=bool)
    invalid_idx[valid_idx] = False
    np.add.at(nan_count, answers[invalid_idx] - 1, 1)

# we need this to avoid NaN values -> if bin_count is 0 then true_count is anyways (also abs_err)
no_zero_bin_count = np.where(bin_count == 0, 1, bin_count)

bin_acc = (true_count / no_zero_bin_count) * 100
# compute average absolute error per bin (exclude nan_counts)
bin_avg_abs_err = (total_abs_err / (no_zero_bin_count - nan_count))

# NOTE: can be loaded again via data = np.load('./data/evaluation/blip.npz') => data is a dictionary (e.g. data['bin_count'])
eval_data_path = './data/evaluation'
os.makedirs(eval_data_path, exist_ok=True)
np.savez(os.path.join(eval_data_path, 'blip.npz'), bin_count=bin_count, bin_acc=bin_acc, bin_avg_abs_err=bin_avg_abs_err, nan_count=nan_count)


# show results, NOTE: For now Blip used twice to test the plotting functions
plot_data_path = './data/plots'
os.makedirs(plot_data_path, exist_ok=True)
x = np.arange(1, config.MAX_OBJ_NUMBER + 1)

label_dict = {'x_label': 'Number of Objects', 'y_label': 'Total Occurences', 'title': 'Total data distribution', 'subtitles': ['BLIP-VQA-Base', 'BLIP-VQA-Base']}
create_bar_plots(x=x, Y=np.vstack((bin_count, bin_count)), labels=label_dict, output_path=os.path.join(plot_data_path, "distribution.png"))

label_dict.update({'y_label': 'Accuracy', 'title': 'Accuracy over individual numbers of objects'})
create_bar_plots(x=x, Y=np.vstack((bin_acc, bin_acc)), labels=label_dict, output_path=os.path.join(plot_data_path, "accuracy.png"))

label_dict.update({'y_label': 'Average Absolute Error', 'y_label2': 'NaN Outputs', 'title': 'Average Absolute Error and NaN Outputs for every number of objects'})
blip_dual_data = np.vstack((bin_avg_abs_err, nan_count))[None, :, :]
create_dual_axis_bar_plots(x=x, Y=np.vstack((blip_dual_data, blip_dual_data)), labels=label_dict, output_path=os.path.join(plot_data_path, "abs_error.png"))



print(f'Total Accuracy: {(total_count / len(test_set))*100}%')
for i in range(config.MAX_OBJ_NUMBER):
  print(f'Accuracy for {i+1} objects: {bin_acc[i]}%')
