# Third Party
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

font = 'Calibri'
dark_red = '#4F1748'
red = '#9D2F92'
dark_blue = '#19779E'
blue = '#199FD3'
title_size = 24
subtitle_size = 16
axes_size = 24


def plot_ecg_1(synth_set, reference_set, x_test):
    # Plotting ECG Signals - first in each array
    plt.plot(x_test[0])
    plt.plot(synth_set[0])
    plt.plot(reference_set[0])
    plt.legend(labels=['x_test first ECG (used in training)', 'synthetic first ECG', 'reference set first ECG'])
    plt.show()


def plot_all_kde(synth_set, reference_set):
    # Plotting KDE of synth and ref set (synth = filled) - Plots every KDE
    # sns.kdeplot(synth_set.transpose(1, 0), fill=True)
    # sns.kdeplot(reference_set.transpose(1, 0))
    # plt.show()
    pass


def plot_average_kde(synth_set, reference_set):
    avg_ref_set = np.mean(reference_set, axis=0)
    avg_synth_set = np.mean(synth_set, axis=0)

    # Plotting KDE of synth and ref set (ref = filled) = IS THE AVERAGE - IS THIS CORRECT
    sns.kdeplot(avg_ref_set, fill=True)
    sns.kdeplot(avg_synth_set)
    plt.legend(labels=['ref set', 'synth set'])
    plt.show()


def plot_average_kde_vs_training_and_non_kde(synth_set, reference_set, x_test, y_test, y_pred, information = 'Blank'):
    """Used to compare the average KDE (which I think is best KDE representation), against the KDE of an ECG signal
    used in training, and one that wasn't. It is then possible to compare the KDE of a training or non-training
    member to the synth and reference KDEs, and compare which is closer and then the assignment of if the program
    thinks it is part of the training data. It will likely show True if it is close to synth, showing the program
    acts as it should. I COULD THEN COMPARE THE PREDICTION to the actual for that one to support how time sensitivity
    is needed (for one that gets it wrong and plot the ECG's with it to show even if KDE is similar, ECG might not be,
    showing need for time sensitivity for accurate prediction, and even more for specific MIA."""

    avg_ref_set = np.mean(reference_set, axis=0)
    avg_synth_set = np.mean(synth_set, axis=0)

    # Plotting the first x_set ECG KDE against the averages to show which it may be closer to, along with printing
    # its classification. Also plotting the last x_test (not from training) and assignment to see if it is closer to ref
    sns.kdeplot(avg_ref_set, fill=True, color=dark_red)
    sns.kdeplot(avg_synth_set, fill=True, color=dark_blue)
    sns.kdeplot(x_test[len(x_test)-1], color=red)
    sns.kdeplot(x_test[0], color=blue)

    plt.suptitle('Probability Densities of ECG Values for Prediction Success Comparison',
              fontname=font,
              fontsize=title_size)
    plt.title(f'Synthesiser: {information[0]} | Training Set Size: {information[1]} | '
              f'Reference Set Size: {information[2]} | '
              f'Synthetic Set Size: {information[3]} | Epochs: {information[4]} | '
              f'Bandwidth: {information[5]}',
                 fontname=font,
                 fontsize=subtitle_size)

    plt.xlabel('ECG Value', fontname=font, fontsize=axes_size)
    plt.ylabel('Density', fontname=font, fontsize=axes_size)

    plt.xlim([0, 1.2])

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)

    # imperial blue [35, 159, 211] #199FD3 , darker 25 119 158 #19779E
    # pink 157 47 146 #9D2F92 darker 79 23 72 #4F1748

    # NOTE Use y_test too in legend if want to show if it got it correct. - dont as we know
    plt.legend(labels=['Population Data',
                       'Synthetic Data',
                       f'Population Sample PDF | Assignment = {y_pred[len(y_pred)-1]}',
                       f'Training Sample PDF | Assignment = {y_pred[0]}'],
               fontsize=20)
    plt.show()


def plot_first_kde_vs_training_and_non_kde(synth_set, reference_set, x_test, y_test, y_pred):
    # Could also plot the first KDE's - and comapre to first and last x test instead of avg
    sns.kdeplot(reference_set[0, :], fill=True)
    sns.kdeplot(synth_set[0, :])
    plt.legend(labels=['first ref set kde',
                       'first synth set kde',
                       f'x_test[0] (from training) | Assignment = {y_pred[0]}',
                       f'x_test[end] (unseen) | Assignment = {y_pred[len(y_pred)-1]}'])
    plt.show()


if __name__ == "__main__":
    # Use here to plot data.
    folder_name = "TVAE_20-04-24-1855_40_40_40_40_kde_mia_domias_evaluation_scott"

    mem_set = pd.read_csv(f'Saved Data/{folder_name}/mem_set.csv').drop(['Unnamed: 0'], axis=1).to_numpy(dtype=np.float32)
    reference_set = pd.read_csv(f'Saved Data/{folder_name}/ref_set.csv').drop(['Unnamed: 0'], axis=1).to_numpy(dtype=np.float32)
    synth_set = pd.read_csv(f'Saved Data/{folder_name}/synth_set.csv').drop(['Unnamed: 0'], axis=1).to_numpy(dtype=np.float32)
    x_test = pd.read_csv(f'Saved Data/{folder_name}/x_test.csv').drop(['Unnamed: 0'], axis=1).to_numpy(dtype=np.float32)
    y_test = pd.read_csv(f'Saved Data/{folder_name}/y_test.csv').drop(['Unnamed: 0'], axis=1).to_numpy(dtype=np.bool_).flatten()
    y_pred = pd.read_csv(f'Saved Data/{folder_name}/y_pred.csv').drop(['Unnamed: 0'], axis=1).to_numpy(dtype=np.bool_).flatten()

    information = ['TVAE', '40', '40', '40', '40', 'Scott']  # TODO future make it take it auto from the text file
    plot_average_kde_vs_training_and_non_kde(synth_set, reference_set, x_test, y_test, y_pred, information)
