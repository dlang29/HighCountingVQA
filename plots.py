import matplotlib.pyplot as plt
import numpy as np

import os

import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_function(bin_df, results_df, test_set_length, config):
    total_count = (results_df["prediction"]==results_df["answers"]).sum()
    # Calculate Total Accuracy
    print(f'Total Accuracy: {(total_count / test_set_length) * 100}%')
    for i in range(config.MAX_OBJ_NUMBER):
        print(f'Accuracy for {i+1} objects: {bin_df.loc[i+1, 'bin_acc']}%')
    
    os.makedirs(config.PLOT_DATA_PATH, exist_ok=True)
    x = np.arange(1, config.MAX_OBJ_NUMBER + 1)

    # Plotting Total Occurrences
    label_dict = {'x_label': 'Number of Objects', 'y_label': 'Total Occurrences', 'title': 'Total data distribution', 'subtitles': ['BLIP-VQA-Base', 'BLIP-VQA-Base']}
    create_bar_plots(x=x, Y=np.vstack((bin_df['bin_count'].values, bin_df['bin_count'].values)), labels=label_dict, output_path=os.path.join(config.PLOT_DATA_PATH, "distribution.png"))

    # Plotting Accuracy
    label_dict.update({'y_label': 'Accuracy', 'title': 'Accuracy over individual numbers of objects'})
    create_bar_plots(x=x, Y=np.vstack((bin_df['bin_acc'].values, bin_df['bin_acc'].values)), labels=label_dict, output_path=os.path.join(config.PLOT_DATA_PATH, "accuracy.png"))

    # Plotting Average Absolute Error and NaN Outputs
    label_dict.update({'y_label': 'Average Absolute Error', 'y_label2': 'NaN Outputs', 'title': 'Average Absolute Error and NaN Outputs for every number of objects'})
    dual_data = np.vstack((bin_df['bin_avg_abs_err'].values, bin_df['nan_count'].values))[None, :, :]
    create_dual_axis_bar_plots(x=x, Y=np.vstack((dual_data, dual_data)), labels=label_dict, output_path=os.path.join(config.PLOT_DATA_PATH, "abs_error.png"))

def create_bar_plots(x, Y, labels=None, output_path="barplots.png"):
    """
    Create a figure with multiple bar plots. 
    
    Params:
    x: Shared x-values for the barplots, shape (num_data_points, )
    Y: Set of y-values for the barplots, shape (num_plots, num_datapoints)
    labels: Optional dictionary to set labels {'x_label': 'x-axis label', 'y_label': 'y-axis label', title': 'title name', 'subtitles': ['model1', 'model2',...]}
    output_path: Where to store the created figure
    """

    num_plots = Y.shape[0]
    width = 2
    height = int(np.ceil(num_plots / width))
    fig, axes = plt.subplots(height, width, figsize=(10 * width, 5 * height), squeeze=False)
    
    if labels is not None:
        x_label = labels.get('x_label', 'x-axis')
        y_label = labels.get('y_label', 'y-axis')
        title = labels.get('title', 'Barplots')
        subtitles = labels.get('subtitles', [f'Model {i}' for i in range(num_plots)])

    for i in range(width):
        for j in range(height):
            if (i+1) * (j+1) <= num_plots:
                axes[j, i].bar(x, Y[i+j])
                axes[j, i].set_title(subtitles[i+j])

                axes[j, i].set_xlabel(x_label)
                axes[j, i].set_ylabel(y_label)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    plt.savefig(output_path)



def create_dual_axis_bar_plots(x, Y, labels=None, output_path="dual_barplots.png"):
    """
    Create a figure with 2 barplots per axis. 
    
    Params:
    x: Shared x-values for the barplots, shape (num_data_points, )
    Y: Set of y-values for the barplots, shape (num_plots, 2, num_datapoints)
    labels: Optional dictionary to set labels {'x_label': 'x-axis label', 'y_label': 'y-axis label', 'y_label2': '2. y-axis label', title': 'title name', 'subtitles': ['model1', 'model2',...]}
    output_path: Where to store the created figure
    """

    num_plots = Y.shape[0]
    width = 2
    height = int(np.ceil(num_plots / width))
    fig, axes = plt.subplots(height, width, figsize=(10 * width, 5 * height), squeeze=False)
    
    if labels is not None:
        x_label = labels.get('x_label', 'x-axis')
        y_label = labels.get('y_label', 'y-axis')
        y_label2 = labels.get('y_label2', '2. y-axis')
        title = labels.get('title', 'Dual-Barplots')
        subtitles = labels.get('subtitles', [f'Model {i}' for i in range(num_plots)])

    bar_width = 0.4
    for i in range(width):
        for j in range(height):
            if (i+1) * (j+1) <= num_plots:
                ax1 = axes[j, i]
                ax1.bar(x - bar_width/2, Y[i+j, 0], bar_width, color='g', label=y_label)
                ax1.set_xlabel(x_label)
                ax1.set_ylabel(y_label, color='g')
                ax1.tick_params(axis='y', labelcolor='g')
                
                ax2 = ax1.twinx()
                ax2.bar(x + bar_width/2, Y[i+j, 1], bar_width, color='b', label=y_label2)
                ax2.set_ylabel(y_label2, color='b')
                ax2.tick_params(axis='y', labelcolor='b')
                
                ax1.set_title(subtitles[i+j])
                ax1.set_xticks(x)
                ax1.set_xticklabels(x)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    plt.savefig(output_path)

def create_confusion_matrices(model_id, evaluation_path, plot_path):
    # Load the CSV file
    file_path = './data/evaluation/blip.csv'
    output_path = './data/plots/confusion_matrices.png'

    data = pd.read_csv(file_path)

    # Extract the relevant columns
    real_answers = data['real_answer'].astype(str)
    formatted_predictions = data['formatted_prediction'].astype(str)
    non_formatted_predictions = data['non_formatted_prediction'].astype(str)

    count = 25

    # Get the 15 most common predictions from both formatted and non-formatted predictions
    most_common_formatted = formatted_predictions.value_counts().head(count).index
    most_common_non_formatted = non_formatted_predictions.value_counts().head(count).index

    # Create a combined list of most common predictions
    most_common_predictions = pd.Index(most_common_formatted).union(pd.Index(most_common_non_formatted)).tolist()
    most_common_predictions = list(map(str, most_common_predictions))
    most_common_predictions.append('Other')  # Ensure 'Other' is included in the labels

    # Filter the predictions
    formatted_predictions_filtered = formatted_predictions.apply(lambda x: x if x in most_common_predictions else 'Other')
    non_formatted_predictions_filtered = non_formatted_predictions.apply(lambda x: x if x in most_common_predictions else 'Other')

    # Sort real labels by their integer casted versions
    real_answers_sorted = sorted(real_answers.unique(), key=lambda x: int(x) if x.isdigit() else float('inf'))

    # Sort predicted labels: integer castable ones first, followed by non-integer castable ones sorted alphabetically
    int_castable_predictions = sorted([p for p in most_common_predictions if p.isdigit()], key=int)
    non_int_castable_predictions = sorted([p for p in most_common_predictions if not p.isdigit() and p != 'Other'])
    sorted_predictions = int_castable_predictions + non_int_castable_predictions + ['Other']

    # Compute the confusion matrices for formatted and non-formatted predictions
    cm_formatted = pd.crosstab(real_answers, formatted_predictions_filtered, rownames=['True'], colnames=['Predicted'], dropna=False)
    cm_non_formatted = pd.crosstab(real_answers, non_formatted_predictions_filtered, rownames=['True'], colnames=['Predicted'], dropna=False)

    # Ensure all columns are present in the confusion matrix
    for col in sorted_predictions:
        if col not in cm_formatted.columns:
            cm_formatted[col] = 0
        if col not in cm_non_formatted.columns:
            cm_non_formatted[col] = 0

    # Reindex to ensure order
    cm_formatted = cm_formatted.reindex(index=real_answers_sorted, columns=sorted_predictions, fill_value=0)
    cm_non_formatted = cm_non_formatted.reindex(index=real_answers_sorted, columns=sorted_predictions, fill_value=0)

    # Remove columns with a sum of 0
    cm_formatted = cm_formatted.loc[:, (cm_formatted != 0).any(axis=0)]
    cm_non_formatted = cm_non_formatted.loc[:, (cm_non_formatted != 0).any(axis=0)]

    # Get the updated sorted predictions after removing zero columns
    updated_sorted_predictions_formatted = cm_formatted.columns.tolist()
    updated_sorted_predictions_non_formatted = cm_non_formatted.columns.tolist()

    # Plot the confusion matrices side by side and save the plot
    fig, axes = plt.subplots(1, 2, figsize=(30, 10))

    # Confusion matrix for formatted predictions
    sns.heatmap(cm_formatted, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=updated_sorted_predictions_formatted, yticklabels=real_answers_sorted, ax=axes[0], annot_kws={"size": 15})
    axes[0].set_title('Confusion Matrix - Formatted Predictions', fontsize=20)
    axes[0].set_xlabel('Predicted Labels', fontsize=15)
    axes[0].set_ylabel('True Labels', fontsize=15)
    axes[0].xaxis.set_ticks_position('both')
    axes[0].yaxis.set_ticks_position('both')
    axes[0].tick_params(labeltop=True, labelright=True)

    # Confusion matrix for non-formatted predictions
    sns.heatmap(cm_non_formatted, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=updated_sorted_predictions_non_formatted, yticklabels=real_answers_sorted, ax=axes[1], annot_kws={"size": 15})
    axes[1].set_title('Confusion Matrix - Non-Formatted Predictions', fontsize=20)
    axes[1].set_xlabel('Predicted Labels', fontsize=15)
    axes[1].set_ylabel('True Labels', fontsize=15)
    axes[1].xaxis.set_ticks_position('both')
    axes[1].yaxis.set_ticks_position('both')
    axes[1].tick_params(labeltop=True, labelright=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
