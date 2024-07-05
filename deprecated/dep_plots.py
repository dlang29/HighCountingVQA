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
plot_function([bin_df_blip, bin_df_pali], [results_df_blip, results_df_pali], ['BLIP-VQA-Base', 'PaliGemma-3b-mix'], len(results_df_blip))
