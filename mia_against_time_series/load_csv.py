import pandas as pd
from scipy import signal
from sdv.metadata import SingleTableMetadata
import numpy as np

metadata = SingleTableMetadata()


def csv_to_numpy(file, drop_class=True, drop_patient=True, downsample=True):
    """
    Will convert the CSV file where each record is a time-series signal, into a Numpy array.

    There is an option to remove the Class and Patient columns when CSV is formatted as in this project
    """

    # Setting the path for the all_records.csv file
    record_path = file

    # A dataframe is a 2D data structure with columns that can be of different types with labels
    # .drop removes columns if axis=1. and rows if axis = 0 - here it is removing unnamed:0 column

    patients_dataframe = pd.read_csv(record_path).drop(['Unnamed: 0'], axis=1)

    if drop_class:
        patients_dataframe = patients_dataframe.drop('Class', axis=1)

    if drop_patient:
        patients_dataframe = patients_dataframe.drop('Patient', axis=1)

    if downsample:
        f = signal.decimate(patients_dataframe, 3)
        patients_dataframe = pd.DataFrame(f)
        # Have to convert column names to string from int
        patients_dataframe.columns = patients_dataframe.columns.astype(str)

    # Convert to Numpy array
    all_records_array = patients_dataframe.to_numpy(dtype=np.float32)

    metadata.detect_from_dataframe(patients_dataframe)

    return all_records_array, metadata
