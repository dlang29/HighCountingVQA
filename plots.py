import os
import json
import config
import numpy as np
import pandas as pd
import seaborn as sns
from utils import get_csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def create_data_distribution(dataset_name: str = config.JSON_FILE) -> None:
    """
    Generates and saves a bar plot visualizing the distribution of labels in a specified dataset.

    The function reads a JSON file containing data, converts it into a DataFrame, and counts the occurrences of each label. 
    It then plots these counts using a bar chart and saves the plot to a specified directory.

    Args:
    dataset_name (str): The path to the JSON file that contains the dataset. The default is specified by config.JSON_FILE.

    Returns:
    None: This function does not return any value. It saves the resulting plot to 'config.PLOT_DATA_PATH/distributions',
          naming the file after the base name of the dataset JSON file, appended with '_distribution.png'.
    """
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
    plt.savefig(os.path.join(config.PLOT_DATA_PATH + '/distributions', dataset_name.split("/")[2].split(".")[0]+"_distribution.png"))
    plt.close()

def create_data_distributions(filenames: List[str]) -> None:
    """
    Reads multiple JSON files, each containing data for a dataset, and aggregates the label counts across these datasets.
    It generates a bar plot visualizing the distribution of labels across all provided datasets, showing the label counts 
    side by side for comparison.

    The function processes each file, counts the occurrences of each label, combines these counts into a single DataFrame, 
    and then plots this information. It fills missing values with zeros to handle labels that do not appear in some datasets.

    Args:
    filenames (List[str]): A list of strings where each string is the path to a JSON file containing dataset information.

    Returns:
    None: The function does not return any value. It saves the resulting plot to 'config.PLOT_DATA_PATH/distributions',
          naming the file by concatenating the base names of the JSON files, appended with '_distribution.png'.
    """
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
    plt.savefig(os.path.join(config.PLOT_DATA_PATH + '/distributions', "distribution_"+"_".join(processed_filenames).replace("/","")+".png"))
    plt.close()

def plot_model_accuracies(model_names: list[str], dataset_name: str = config.JSON_FILE) -> None:
    """
    Generates and saves a bar plot comparing the accuracy of various models on a specified dataset.

    This function processes each specified model, computes the accuracy for each label within the dataset,
    and then calculates the overall accuracy. These accuracies are then plotted in a bar chart for easy comparison.

    Args:
    model_names (list[str]): A list of identifiers for the models to be evaluated.
    dataset_name (str): The path to the JSON file that contains the dataset. The default is specified by config.JSON_FILE.

    Returns:
    None: This function does not return any value. It saves the resulting plot to 'config.PLOT_DATA_PATH',
          naming the file by appending the dataset base name and model names with '_Accuracy_Comparison.png'.
    """
    dataset_name = dataset_name.split("/")[2].split(".")[0].replace("/","_")
    accuracies = {}
    overall_accuracies = {}
    for model_id in model_names:
        df = get_csv("results", model_id, False)
        # Calculate the overall accuracy for the entire dataset
        df['correct'] = df['real_answer'].astype(str) == df['prediction'].astype(str)
        overall_accuracy = df['correct'].mean()
        overall_accuracies[model_id] = overall_accuracy
        accuracies[model_id] = df.groupby('real_answer')['correct'].mean()
    accuracies_df = pd.DataFrame(accuracies)
    accuracies_df.loc['All'] = overall_accuracies
    accuracies_df.plot(kind='bar', figsize=(12, 8))
    plt.title('Accuracies for Each Label for '+dataset_name)
    plt.xlabel('Label')
    plt.ylabel('Accuracy')
    plt.legend(title='Model')
    plt.grid(axis='y')
    plt.tight_layout()
    processed_model_names = "_".join([model_name.split("/")[1] for model_name in model_names])
    plt.savefig(os.path.join(config.PLOT_DATA_PATH, dataset_name+"_"+processed_model_names+"_Accuracy_Comparison.png"))
    plt.close()

def plot_model_abs_error(model_names: list[str], dataset_name: str = config.JSON_FILE) -> None:
    """
    Generates and saves a bar plot comparing the mean absolute error of various models on a specified dataset.

    This function processes each specified model, calculates the absolute error for each prediction against the true value,
    computes the mean of these errors for each label, and also a total mean absolute error for all labels. These metrics
    are plotted in a bar chart for comparative analysis.

    Args:
    model_names (list[str]): A list of identifiers for the models whose accuracy is being compared.
    dataset_name (str): The path to the JSON file that contains the dataset. Defaults to config.JSON_FILE.

    Returns:
    None: This function does not return any value. It saves the resulting plot to 'config.PLOT_DATA_PATH',
          naming the file by appending the dataset base name and model names with '_Abs_Error_Comparison.png'.
    """
    dataset_name = dataset_name.split("/")[2].split(".")[0].replace("/","_")
    abs_errors = {}
    overall_abs_errors = {}
    for model_id in model_names:
        df = get_csv("results", model_id, False)
        df = df[df['formatted_pred'].astype(int) != -1]
        # Calculate the absolute error for each formatted_pred
        df['abs_error'] = (df['real_answer'].astype(int) - df['formatted_pred'].astype(int)).abs()
        # Calculate overall absolute error for the entire dataset
        overall_abs_error = df['abs_error'].mean()
        overall_abs_errors[model_id] = overall_abs_error
        abs_errors[model_id] = df.groupby('real_answer')['abs_error'].mean()
    abs_errors_df = pd.DataFrame(abs_errors)
    abs_errors_df.loc['All'] = overall_abs_errors
    abs_errors_df.plot(kind='bar', figsize=(12, 8))
    plt.title('Mean of Absolute Errors for ' + dataset_name)
    plt.xlabel('Label')
    plt.ylabel('Absolute Error')
    plt.legend(title='Model')
    plt.grid(axis='y')
    plt.tight_layout()
    processed_model_names = "_".join([model_name.split("/")[1] for model_name in model_names])
    plt.savefig(os.path.join(config.PLOT_DATA_PATH, dataset_name + "_" + processed_model_names + "_Abs_Error_Comparison.png"))
    plt.close()

def plot_model_nan_count(model_names: list[str], dataset_name: str = config.JSON_FILE) -> None:
    """
    Generates and saves a bar plot illustrating the count of NaN values, represented as -1 in 'formatted_pred',
    for each label across various models in a specified dataset.

    The function loads CSV data for each model, counts occurrences of NaN values (as -1) for each label, and
    aggregates these counts across all specified models. The resulting counts are then plotted in a bar chart.

    Args:
    model_names (list[str]): A list of identifiers for the models to be evaluated.
    dataset_name (str): The path to the JSON file that contains the dataset. The default is specified by config.JSON_FILE.

    Returns:
    None: This function does not return any value. It saves the resulting plot to 'config.PLOT_DATA_PATH',
          naming the file by appending the dataset base name and model names with '_NaN_Count_Comparison.png'.
    """
    dataset_name = dataset_name.split("/")[2].split(".")[0].replace("/","_")
    nan_counts = {}
    for model_id in model_names:
        df = get_csv("results", model_id, False)
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
    processed_model_names = "_".join([model_name.split("/")[1] for model_name in model_names])
    plt.savefig(os.path.join(config.PLOT_DATA_PATH, dataset_name+"_"+processed_model_names+"_NaN_Count_Comparison.png"))
    plt.close()

def create_confusion_matrices(model_name: str, dataset_name: str = config.JSON_FILE) -> None:
    """
    Generates and saves confusion matrices for both formatted and non-formatted predictions
    from a model, comparing these predictions to the true labels.

    This function processes the results from a model's output stored in a CSV file, identifies the
    most common predictions, filters the rest under an 'Other' category, and then creates confusion
    matrices for both types of predictions. The matrices are plotted and saved to a specified directory.

    Args:
    model_name (str): The name of the model used to generate predictions. This name is used to
                      retrieve the corresponding results file and in naming the output files.
    dataset_name (str): The path to the JSON file that contains the dataset. The default is specified
                        by config.JSON_FILE. This name is used in the titles of the plots.

    Returns:
    None: This function does not return any value. It saves the resulting plot to 'config.PLOT_DATA_PATH/confusion_matrices',
          naming the file by appending the model name and dataset base name with '_Confusion_Matrix_Non_Formatted.png'.
    """
    dataset_name = dataset_name.split("/")[2].split(".")[0].replace("/","_")
    data = get_csv("results", model_name, False)
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
    plt.savefig(os.path.join(config.PLOT_DATA_PATH + '/confusion_matrices', model_name.replace("/", "_").replace("-", "_") + "_" + dataset_name + "_Confusion_Matrix_Formatted.png"))
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
    plt.savefig(os.path.join(config.PLOT_DATA_PATH + '/confusion_matrices', model_name.replace("/", "_").replace("-", "_") + "_" + dataset_name + "_Confusion_Matrix_Non_Formatted.png"))
    plt.close()

def generate_latex_code(image_folder: str = config.PLOT_DATA_PATH, latex_file: str = 'latex_image_inclusion.tex') -> None:
    """
    Generates a LaTeX file that includes all image files from a specified directory.

    This function walks through the given image folder, finds all images with the extensions .png, .jpg, or .jpeg,
    and writes LaTeX code to include these images in a document. The images are included with a width of 50% of the line width.
    The resulting LaTeX code is written to a specified output file.

    Args:
    image_folder (str): The path to the folder containing the images to be included in the LaTeX document.
    latex_file (str): The file path where the LaTeX document should be saved.

    Returns:
    None: This function does not return any value. It writes directly to the file specified by 'latex_output',
          creating 'latex_image_inclusion.tex' in 'config.PLOT_DATA_PATH'.
    """
    latex_output = os.path.join(image_folder, latex_file)
    with open(latex_output, 'w') as f:
        f.write('\\documentclass{article}\n')
        f.write('\\usepackage{graphicx}\n')
        f.write('\\begin{document}\n')
        f.write('\\section*{Images}\n')

        for dirpath, dirnames, files in os.walk(image_folder):
            for image_file in files:
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    relative_path = os.path.relpath(os.path.join(dirpath, image_file), start=image_folder)
                    f.write(f'\\includegraphics[width=0.5\\linewidth]{{{"plots/"+image_file}}}\n')

        f.write('\\end{document}\n')

if __name__ == "__main__":
    plot_model_accuracies(["Salesforce/blip-vqa-base", "google/paligemma-3b-mix-224", "Salesforce/blip-vqa-base_trained", "Salesforce/blip-vqa-base_trained_lastcheckpoint"], dataset_name = "./data/HighCountVQA_test.json")
    plot_model_abs_error(["Salesforce/blip-vqa-base", "google/paligemma-3b-mix-224", "Salesforce/blip-vqa-base_trained", "Salesforce/blip-vqa-base_trained_lastcheckpoint"], dataset_name = "./data/HighCountVQA_test.json")

    create_confusion_matrices("Salesforce/blip2-opt-2.7b", dataset_name = "./data/HighCountVQA_combined.json")

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

    plot_model_accuracies(["google/paligemma-3b-mix-224", "Salesforce/blip-vqa-base"], dataset_name = "./data/HighCountVQA_test.json")
    plot_model_accuracies(["google/paligemma-3b-mix-224", "Salesforce/blip-vqa-base", "Salesforce/blip2-opt-2.7b"], dataset_name = "./data/HighCountVQA_combined.json")

    plot_model_abs_error(["google/paligemma-3b-mix-224", "Salesforce/blip-vqa-base"], dataset_name = "./data/HighCountVQA_test.json")
    plot_model_abs_error(["google/paligemma-3b-mix-224", "Salesforce/blip-vqa-base", "Salesforce/blip2-opt-2.7b"], dataset_name = "./data/HighCountVQA_combined.json")

    plot_model_accuracies(["google/paligemma-3b-mix-224", "google/paligemma-3b-mix-224_trained", "google/paligemma-3b-mix-224_trained_lastcheckpoint"], dataset_name = "./data/HighCountVQA_test.json")
    plot_model_abs_error(["google/paligemma-3b-mix-224", "google/paligemma-3b-mix-224_trained", "google/paligemma-3b-mix-224_trained_lastcheckpoint"], dataset_name = "./data/HighCountVQA_test.json")
    plot_model_nan_count(["google/paligemma-3b-mix-224", "google/paligemma-3b-mix-224_trained", "google/paligemma-3b-mix-224_trained_lastcheckpoint"], dataset_name = "./data/HighCountVQA_test.json")

    plot_model_accuracies(["Salesforce/blip-vqa-base", "Salesforce/blip-vqa-base_trained", "Salesforce/blip-vqa-base_trained_lastcheckpoint"], dataset_name = "./data/HighCountVQA_test.json")
    plot_model_abs_error(["Salesforce/blip-vqa-base", "Salesforce/blip-vqa-base_trained", "Salesforce/blip-vqa-base_trained_lastcheckpoint"], dataset_name = "./data/HighCountVQA_test.json")
    plot_model_nan_count(["Salesforce/blip-vqa-base", "Salesforce/blip-vqa-base_trained", "Salesforce/blip-vqa-base_trained_lastcheckpoint"], dataset_name = "./data/HighCountVQA_test.json")

    create_confusion_matrices("Salesforce/blip-vqa-base", dataset_name = "./data/HighCountVQA_test.json")
    create_confusion_matrices("Salesforce/blip-vqa-base_trained", dataset_name = "./data/HighCountVQA_test.json")
    create_confusion_matrices("Salesforce/blip-vqa-base_trained_lastcheckpoint", dataset_name = "./data/HighCountVQA_test.json")

    create_confusion_matrices("google/paligemma-3b-mix-224", dataset_name = "./data/HighCountVQA_test.json")
    create_confusion_matrices("google/paligemma-3b-mix-224_trained", dataset_name = "./data/HighCountVQA_test.json")
    create_confusion_matrices("google/paligemma-3b-mix-224_trained_lastcheckpoint", dataset_name = "./data/HighCountVQA_test.json")
    
    generate_latex_code()