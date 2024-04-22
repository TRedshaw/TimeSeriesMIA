import csv
import pandas as pd

filename = "Saved Data/all_test_results.csv"


def update_all_test_results(
        generator,
        date,
        training_size,
        reference_size,
        synthetic_size,
        epochs,
        mia_method,
        bandwidth,
        acc,
        auc
):

    # fields = ['Generator', 'Date', 'Training Size', 'Reference Size', 'Synthetic Size', 'Epochs', 'MIA Method', 'Bandwidth', 'Acc', 'AUC']

    filename = "Saved Data/all_test_results.csv"
    fields = [generator, date, training_size, reference_size, synthetic_size, epochs, mia_method, bandwidth, acc, auc]

    # writing to csv file
    with open(filename, 'a+', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the fields
        csvwriter.writerow(fields)


def save_matrices(folder_name, mem_set, reference_set, synth_set, x_test, y_test, y_pred):
    # Save matrices as CSV
    pd.DataFrame(mem_set).to_csv(f'Saved Data/{folder_name}/mem_set.csv')
    pd.DataFrame(reference_set).to_csv(f'Saved Data/{folder_name}/ref_set.csv')
    pd.DataFrame(synth_set).to_csv(f'Saved Data/{folder_name}/synth_set.csv')
    pd.DataFrame(x_test).to_csv(f'Saved Data/{folder_name}/x_test.csv')
    pd.DataFrame(y_test).to_csv(f'Saved Data/{folder_name}/y_test.csv')
    pd.DataFrame(y_pred).to_csv(f'Saved Data/{folder_name}/y_pred.csv')


def save_info(folder_name,
            generator,
            date,
            training_size,
            reference_size,
            synthetic_size,
            epochs,
            mia_method,
            bandwidth,
            acc,
            auc):

    fields = [f'''Generator: {generator}
Date: {date}
Training Size: {training_size}
Reference Size: {reference_size}
Synthetic Size: {synthetic_size}
Epochs: {epochs}
MIA Method: {mia_method}
Bandwidth: {bandwidth}
Acc: {acc}
AUC: {auc}''']

    with open(f'Saved Data/{folder_name}/info.txt', 'a+', newline='') as info_file:
        info_file.writelines(fields)
