import numpy as np
import config
import os
import pandas as pd

def get_filename(name, model_id=config.MODEL_ID, trained = config.TRAINED):
    """Get standardized filename based on config."""
    if trained:
        model_id = model_id + "_trained"
    filename = (model_id+config.JSON_FILE.split(".")[1]+"_"+name+'.csv').replace("/","_")
    csv_file_path = os.path.join(config.EVAL_DATA_PATH, filename)
    return csv_file_path

def write_to_csv(df, name):
    df.to_csv(get_filename(name), index=False)

def get_csv(name, model_id=config.MODEL_ID, trained = config.TRAINED):
    return pd.read_csv(get_filename(name, model_id, trained))

def calculate_metrics(results_df):
    # create a bin for every object number
    bin_range = range(0, config.MAX_OBJ_NUMBER + 1)
    bin_df = pd.DataFrame(index=bin_range, columns=['bin_count', 'true_count', 'total_abs_err', 'nan_count']).fillna(0)
    # Counting entries for each bin
    for bin_number in bin_range:
        bin_filter = results_df['real_answer'] == bin_number
        # calculate the total datapoints for this bin
        bin_df.at[bin_number, 'bin_count'] = bin_filter.sum()
        # calculate the total count of correct predictions for this bin
        bin_df.at[bin_number, 'true_count'] = (results_df[bin_filter]['formatted_pred'] == bin_number).sum()
        
        valid_preds = results_df[bin_filter & (results_df['formatted_pred'] != -1)]
        # calculate the total absolute error of predictions and labels for this bin (without NaN outputs)
        bin_df.at[bin_number, 'total_abs_err'] = abs(valid_preds['formatted_pred'] - bin_number).sum()
        # calculate the number of NaN outputs for this bin (e.g. "many")
        bin_df.at[bin_number, 'nan_count'] = (results_df[bin_filter]['formatted_pred'] == -1).sum()
    # Calculating metrics
    bin_df['bin_acc'] = (bin_df['true_count'] / bin_df['bin_count'].replace(0, np.nan)) * 100
    bin_df['bin_avg_abs_err'] = bin_df['total_abs_err'] / (bin_df['bin_count'] - bin_df['nan_count']).replace(0, np.nan)
    return bin_df