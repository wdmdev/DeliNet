import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_data_and_plot(file_paths, plot_path):
    plt.style.use('ggplot')  # Setting the style to 'ggplot'

    # Initialize a figure and a set of subplots
    fig, ax1 = plt.subplots()
    fig, ax1 = plt.subplots(figsize=(10, 7))  # Width, Height in inches

    # Initialize lists to store handles and labels for the legend
    handles, labels = [], []

    for file_path in file_paths:
        # Reading the data from TSV file
        data = pd.read_csv(file_path, sep='\t')

        # If the file is for accuracy, plot on the first axis
        if "acc" in file_path:
            line, = ax1.plot(data['step'], data.iloc[:, 1], label=f"{data.columns[1]} (last: {data.iloc[-1, 1]:.3f})")
            handles.append(line)
            labels.append(f"{data.columns[1]} (last: {data.iloc[-1, 1]:.3f})")
        
        # If the file is for loss, normalize and plot on the second axis
        else:
            normalized_loss = data.iloc[:, 1] / np.max(data.iloc[:, 1])
            line, = ax1.plot(data['step'], normalized_loss, label='train loss (normalized)', color='gray')
            handles.append(line)
            labels.append('train loss (normalized) (last: {:.3f})'.format(normalized_loss.iloc[-1]))

    # Set labels and titles
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Accuracy/Normalized Loss')
    ax1.set_title('CLIP Finetuning')

    # Create a single legend
    ax1.legend(handles, labels, loc='best')

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)

# File paths according to the folder structure

if __name__ == "__main__":
    dir = 'CLIP2_903'
    file_paths = [
        dir + '/plots/metrics/top_0_percent_acc.tsv',
        dir + '/plots/metrics/top_10_percent_acc.tsv',
        dir + '/plots/metrics/top_25_percent_acc.tsv',
        dir + '/plots/metrics/train/loss.tsv'
    ]

    plot_path = dir + "/plots/CLIP_train.png"
    read_data_and_plot(file_paths, plot_path)
