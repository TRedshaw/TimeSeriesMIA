import numpy as np

ecg_sequence = np.tile(np.arange(488)+1, 95082).T
print(len(ecg_sequence))
print(488*95082)