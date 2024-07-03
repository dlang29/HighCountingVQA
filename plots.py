import os
import json
import config
import numpy as np
import pandas as pd
import seaborn as sns
from utils import get_csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_function(bin_dfs, results_dfs, model_names, test_set_length):
    """
    Call Plot functions based on evaluation data
    
    Params:
    bin_dfs: Array with bin metrics for every model
    results_df: Array with all evaluation predictions per model
    model_names: Different names of the models to use for the plots
    test_set_length: Length of the test dataset
    """

    for (model_name, bin_df, results_df) in zip(model_names, bin_dfs, results_dfs):
        total_count = (results_df["formatted_pred"]==results_df["real_answer"]).sum()
        # Calculate Total Accuracy
        print(f'Model: {model_name}')
        print(f'Total Accuracy: {(total_count / test_set_length) * 100}%')
        for i in range(0, config.MAX_OBJ_NUMBER + 1):
            print(f'Accuracy for {i} objects: {bin_df.loc[i, "bin_acc"]}%')
    
    os.makedirs(config.PLOT_DATA_PATH, exist_ok=True)
    x = np.arange(0, config.MAX_OBJ_NUMBER + 1)
    
    # Plotting Accuracy
    label_dict = {'x_label': 'Number of Objects', 'y_label': 'Accuracy',  'title': 'Accuracy over individual numbers of objects', 'subtitles': model_names}
    create_bar_plots(x=x, Y=np.vstack([bin_df['bin_acc'].values for bin_df in bin_dfs]), labels=label_dict, output_path=os.path.join(config.PLOT_DATA_PATH, "accuracy.png"))

    # Plotting Average Absolute Error and NaN Outputs
    label_dict.update({'y_label': 'Average Absolute Error', 'y_label2': 'NaN Outputs', 'title': 'Average Absolute Error and NaN Outputs for every number of objects'})
    dual_data = [np.vstack((bin_df['bin_avg_abs_err'].values, bin_df['nan_count'].values))[None, :, :] for bin_df in bin_dfs]
    create_dual_axis_bar_plots(x=x, Y=np.vstack(dual_data), labels=label_dict, output_path=os.path.join(config.PLOT_DATA_PATH, "abs_error.png"))

def create_data_distribution(dataset_name = config.JSON_FILE):
    with open(dataset_name, 'r') as file:
      data = json.load(file)
    df = pd.DataFrame(data)
    label_counts = df['answer'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    bars = label_counts.plot(kind='bar', color='skyblue')
    plt.title('Count of Each Label in answer Column')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    for bar in bars.patches:
        bars.annotate(format(bar.get_height(), '.0f'), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha = 'center', va = 'center', size = 12, xytext = (0, 8), textcoords = 'offset points')
    plt.savefig(os.path.join(config.PLOT_DATA_PATH, dataset_name.split("/")[2].split(".")[0]+"_distribution.png"))

def create_data_distributions(filenames):
    combined_label_counts = {}
    print(filenames)
    for filename in filenames:
        print(filename)
        with open(filename, 'r') as file:
            data = json.load(file)
        df = pd.DataFrame(data)
        label_counts = df['answer'].value_counts().sort_index()
        combined_label_counts[os.path.basename(filename)] = label_counts

    combined_df = pd.DataFrame(combined_label_counts).fillna(0)
    
    combined_df.plot(kind='bar', figsize=(15, 8))
    
    plt.title('Count of Each Label in answer Column for Multiple Datasets')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    for dataset in combined_df.columns:
        for i, count in enumerate(combined_df[dataset]):
            plt.text(i, count + 0.05, int(count), ha='center', va='bottom', fontsize=10, rotation=90)
    
    plt.legend(title='Datasets')
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DATA_PATH, "combined_distribution"+"".join(filenames).replace("/","")+".png"))

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

def create_confusion_matrices(data, output_path):
    # Extract the relevant columns
    real_answers = data['real_answer'].astype(str)
    formatted_predictions = data['formatted_pred'].astype(str)
    non_formatted_predictions = data['prediction'].astype(str)

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


if __name__ == "__main__":
    create_data_distribution("./data/tallyqa.json")
    create_data_distribution("./data/HighCountVQA_val.json")
    create_data_distribution("./data/HighCountVQA_test.json")
    create_data_distribution("./data/HighCountVQA_train.json")
    create_data_distribution("./data/HighCountVQA_combined.json")
    create_data_distributions( ["./data/HighCountVQA_val.json",
                                "./data/HighCountVQA_test.json",
                                "./data/HighCountVQA_train.json"])
    create_data_distributions( ["./data/tallyqa.json",
                                "./data/HighCountVQA_combined.json"])

    results_df_blip = get_csv("results", "Salesforce/blip-vqa-base")
    bin_df_blip = get_csv("bin", "Salesforce/blip-vqa-base")
    results_df_pali = get_csv("results", "google/paligemma-3b-mix-224")
    bin_df_pali = get_csv("bin", "google/paligemma-3b-mix-224")
    plot_function([bin_df_blip, bin_df_pali], [results_df_blip, results_df_pali], ['BLIP-VQA-Base', 'PaliGemma-3b-mix'], len(results_df_blip))
    create_confusion_matrices(results_df_blip, "data/plots/confusion_matrices_blip.png")
    create_confusion_matrices(results_df_pali, "data/plots/confusion_matrices_pali.png")