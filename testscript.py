import os
import config
import pandas as pd
import numpy as np

results_df = pd.read_csv("/teamspace/studios/this_studio/data/evaluation/blip.csv")

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

bin_df = calculate_metrics(config, results_df)
write_to_csv(config, bin_df, "bin")