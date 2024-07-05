import os
import json
import config
import numpy as np
import pandas as pd
import seaborn as sns
from utils import get_csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def create_data_distribution(dataset_name = config.JSON_FILE):
    with open(dataset_name, 'r') as file:
      data = json.load(file)
    df = pd.DataFrame(data)
    label_counts = df['answer'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    bars = label_counts.plot(kind='bar', color='skyblue')
    plt.title('Count of Each Label in '+ dataset_name.split("/")[2].split(".")[0].replace("/", "_"))
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    for bar in bars.patches:
        bars.annotate(format(bar.get_height(), '.0f'), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha = 'center', va = 'center', size = 12, xytext = (0, 8), textcoords = 'offset points')
    plt.savefig(os.path.join(config.PLOT_DATA_PATH, dataset_name.split("/")[2].split(".")[0]+"_distribution.png"))


def create_data_distributions(filenames):
    combined_label_counts = {}
    for filename in filenames:
        with open(filename, 'r') as file:
            data = json.load(file)
        df = pd.DataFrame(data)
        label_counts = df['answer'].value_counts().sort_index()
        combined_label_counts[os.path.basename(filename)] = label_counts
    combined_df = pd.DataFrame(combined_label_counts).fillna(0)

    ax = combined_df.plot(kind='bar', figsize=(15, 8))
    plt.title('Count of Each Label for Multiple Datasets')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 17), textcoords='offset points',
                    fontsize=10,rotation=90)

    plt.legend(title='Datasets')
    plt.tight_layout()
    processed_filenames = [filename.split("/")[2].split(".")[0].replace("/", "_") for filename in filenames]
    plt.savefig(os.path.join(config.PLOT_DATA_PATH, "distribution_"+"_".join(processed_filenames).replace("/","")+".png"))

def plot_model_accuracies(model_names, dataset_name=config.JSON_FILE):
    dataset_name = dataset_name.split("/")[2].split(".")[0].replace("/","_")
    accuracies = {}
    for model_id in model_names:
        df = get_csv("results", model_id)
        # Calculate the accuracy for each real_answer
        df['accuracy'] = df['real_answer'].astype(str) == df['prediction'].astype(str)
        accuracies[model_id] = df.groupby('real_answer')['accuracy'].mean()
    accuracies_df = pd.DataFrame(accuracies)
    accuracies_df.plot(kind='bar', figsize=(12, 8))
    plt.title('Accuracies for Each Label for '+dataset_name)
    plt.xlabel('Label')
    plt.ylabel('Accuracy')
    plt.legend(title='Model')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DATA_PATH, dataset_name+"_Accuracy_Comparison.png"))

def plot_model_abs_error(model_names, dataset_name=config.JSON_FILE):
    dataset_name = dataset_name.split("/")[2].split(".")[0].replace("/","_")
    abs_errors = {}
    for model_id in model_names:
        df = get_csv("results", model_id)
        # Calculate the absolute error for each formatted_pred
        df['abs_error'] = (df['real_answer'].astype(int) - df['formatted_pred'].astype(int)).abs()
        abs_errors[model_id] = df.groupby('real_answer')['abs_error'].mean()  
    abs_errors_df = pd.DataFrame(abs_errors)
    abs_errors_df.plot(kind='bar', figsize=(12, 8))
    plt.title('Absolute Errors for Each Label for ' + dataset_name)
    plt.xlabel('Label')
    plt.ylabel('Absolute Error')
    plt.legend(title='Model')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DATA_PATH, dataset_name + "_Abs_Error_Comparison.png"))

def plot_model_nan_count(model_names, dataset_name=config.JSON_FILE):
    dataset_name = dataset_name.split("/")[2].split(".")[0].replace("/","_")
    nan_counts = {}
    for model_id in model_names:
        df = get_csv("results", model_id)
        # Count NaNs where formatted_pred is -1 for each real_answer
        df['is_nan'] = df['formatted_pred'] == -1
        nan_counts[model_id] = df.groupby('real_answer')['is_nan'].sum()
    nan_counts_df = pd.DataFrame(nan_counts)
    nan_counts_df.plot(kind='bar', figsize=(12, 8))
    plt.title('NaN Counts for Each Label for ' + dataset_name)
    plt.xlabel('Label')
    plt.ylabel('NaN Count')
    plt.legend(title='Model')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DATA_PATH, dataset_name + "_NaN_Count_Comparison.png"))

def create_confusion_matrices(model_name, dataset_name = config.JSON_FILE):
    dataset_name = dataset_name.split("/")[2].split(".")[0].replace("/","_")
    data = get_csv("results", model_id=model_name)
    # Extract the relevant columns
    real_answers = data['real_answer'].astype(str)
    formatted_predictions = data['formatted_pred'].astype(str)
    non_formatted_predictions = data['prediction'].astype(str)
    count = 25
    # Get the $count$ most common predictions from both formatted and non-formatted predictions
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
    fig, axes = plt.subplots(1, 2, figsize=(40, 10))
    # Plot the confusion matrix for formatted predictions
    plt.figure(figsize=(20, 10))
    sns.heatmap(cm_formatted, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=updated_sorted_predictions_formatted, yticklabels=real_answers_sorted, annot_kws={"size": 15})
    plt.title('Confusion Matrix - Formatted Predictions for '+ model_name + " in " + dataset_name, fontsize=20)
    plt.xlabel('Predicted Labels', fontsize=15)
    plt.ylabel('True Labels', fontsize=15)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DATA_PATH, model_name.replace("/", "_").replace("-", "_") + "_" + dataset_name + "_Confusion_Matrix_Formatted.png"))
    plt.close()
    # Plot the confusion matrix for non-formatted predictions
    plt.figure(figsize=(20, 10))
    sns.heatmap(cm_non_formatted, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=updated_sorted_predictions_non_formatted, yticklabels=real_answers_sorted, annot_kws={"size": 15})
    plt.title('Confusion Matrix - Non-Formatted Predictions for '+ model_name + " in " + dataset_name, fontsize=20)
    plt.xlabel('Predicted Labels', fontsize=15)
    plt.ylabel('True Labels', fontsize=15)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DATA_PATH, model_name.replace("/", "_").replace("-", "_") + "_" + dataset_name + "_Confusion_Matrix_Non_Formatted.png"))
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

    plot_model_accuracies(["google/paligemma-3b-mix-224","Salesforce/blip-vqa-base"])
    plot_model_abs_error(["google/paligemma-3b-mix-224","Salesforce/blip-vqa-base"])
    plot_model_nan_count(["google/paligemma-3b-mix-224","Salesforce/blip-vqa-base"])

    create_confusion_matrices("Salesforce/blip-vqa-base")
    create_confusion_matrices("google/paligemma-3b-mix-224")