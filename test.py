import numpy as np
import config
from plots import create_bar_plots, create_dual_axis_bar_plots
import os

plot_data_path = './data/plots'
files = os.listdir(plot_data_path)
for file in files:
    print(file)

data = np.load('./data/evaluation/blip.npz')

bin_count = data['bin_count']
bin_avg_abs_err = data['bin_avg_abs_err']
nan_count = data['nan_count']
bin_acc = data['bin_acc']


x = np.arange(1, config.MAX_OBJ_NUMBER + 1)

label_dict = {'x_label': 'Number of Objects', 'y_label': 'Total Occurences', 'title': 'Total data distribution', 'subtitles': ['BLIP-VQA-Base', 'BLIP-VQA-Base']}
create_bar_plots(x=x, Y=np.vstack((bin_count, bin_count)), labels=label_dict, output_path=os.path.join(plot_data_path, "distribution.png"))

label_dict.update({'y_label': 'Accuracy', 'title': 'Accuracy over individual numbers of objects'})
create_bar_plots(x=x, Y=np.vstack((bin_acc, bin_acc)), labels=label_dict, output_path=os.path.join(plot_data_path, "accuracy.png"))

label_dict.update({'y_label': 'Average Absolute Error', 'y_label2': 'NaN Outputs', 'title': 'Average Absolute Error and NaN Outputs for every number of objects'})
blip_dual_data = np.vstack((bin_avg_abs_err, nan_count))[None, :, :]
create_dual_axis_bar_plots(x=x, Y=np.vstack((blip_dual_data, blip_dual_data)), labels=label_dict, output_path=os.path.join(plot_data_path, "abs_error.png"))