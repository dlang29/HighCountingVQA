from tqdm import tqdm
import numpy as np
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import transformers
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

from dataset import HighCountVQADataset, get_dataset
import config
from plots import plot_function

test_set_length, total_iterations, test_loader = get_dataset(config)

def write_to_csv(config, df, name):
    # Write results to CSV
    filename = (config.MODEL_ID+config.JSON_FILE.split(".")[1]+"_"+name+'.csv').replace("/","_")
    csv_file_path = os.path.join(config.EVAL_DATA_PATH, filename)
    df.to_csv(csv_file_path, index=False)

def calculate_metrics(config, results_df):
    bin_range = range(1, config.MAX_OBJ_NUMBER + 1)
    bin_df = pd.DataFrame(index=bin_range, columns=['bin_count', 'true_count', 'total_abs_err', 'nan_count']).fillna(0)
    # Counting entries for each bin
    for bin_number in bin_range:
        bin_filter = results_df['real_answer'] == bin_number
        bin_df.at[bin_number, 'bin_count'] = bin_filter.sum()
        bin_df.at[bin_number, 'true_count'] = (results_df[bin_filter]['formatted_pred'] == bin_number).sum()
        
        valid_preds = results_df[bin_filter & (results_df['formatted_pred'] != -1)]
        bin_df.at[bin_number, 'total_abs_err'] += abs(valid_preds['formatted_pred'] - bin_number).sum()
        bin_df.at[bin_number, 'nan_count'] += (results_df[bin_filter]['formatted_pred'] == -1).sum()
    # Calculating metrics
    bin_df['bin_acc'] = (bin_df['true_count'] / bin_df['bin_count'].replace(0, np.nan)) * 100
    bin_df['bin_avg_abs_err'] = bin_df['total_abs_err'] / (bin_df['bin_count'] - bin_df['nan_count'].replace(0, np.nan))
    return bin_df

# Blip specific loading
# processor instead of tokenizer -> contains BERT tokenizer + BLIP image processor
processor = AutoProcessor.from_pretrained(config.MODEL_ID, do_rescale=False) #do_rescale false, because ToTensor() already does that
model = AutoModelForVisualQuestionAnswering.from_pretrained(config.MODEL_ID).to(config.DEVICE)

# Initialize a dictionary to hold all data
data_collection = {"image_path": [],"question": [],"real_answer": [],"formatted_pred": [],"prediction": []}

with torch.no_grad():
    for batch in tqdm(test_loader, total=total_iterations):
        questions, imgs, answers, image_paths = batch['question'], batch['image'], batch['answer'], batch['image_path']

        inputs = processor(imgs, questions, padding=True, return_tensors="pt").to(config.DEVICE)
        outputs = model.generate(**inputs)
        predictions = processor.batch_decode(outputs, skip_special_tokens=True)
        formatted_predictions = [pred if pred.isdigit() else config.NUM_DICT.get(pred, "-1") for pred in predictions]

        data_collection['image_path'].extend(image_paths)
        data_collection['question'].extend(questions)
        data_collection['real_answer'].extend(answers)
        data_collection['formatted_pred'].extend(formatted_predictions)
        data_collection['prediction'].extend(predictions)

# Convert dictionary to DataFrame
results_df = pd.DataFrame(data_collection)
results_df['real_answer'] = results_df['real_answer'].astype(int)
results_df['formatted_pred'] = results_df['formatted_pred'].astype(int)
write_to_csv(config, results_df, "results")
bin_df = calculate_metrics(config, results_df)
write_to_csv(config, bin_df, "bin")

plot_function(bin_df, results_df, test_set_length, config)