# Third Party
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KernelDensity


def kde_mia_domias_evaluation(synth_set, reference_set, x_test, y_test, bandwidth='scott'):
    """KDE Estimation"""
    print('Constructing PDF using KDE, for the synthetic data and reference set...\n')
    density_gen = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(synth_set)
    density_data = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(reference_set)

    '''KDE Evaluation'''
    # print('Evaluating sample set of training and non training data...\n')
    # Returns a list of log-likelihood that each x_test sample/row is in the synthetic/reference dataset
    # Higher values (closer to 0) indicate a better model, as log(1) will be closer to 0, and log(0) is infinite

    # Makes the PDF of each x_test and compares to the model you're calling the object of
    p_G_evaluated = density_gen.score_samples(x_test)
    p_R_evaluated = density_data.score_samples(x_test)

    # TESTING PRINTS
    # print(p_G_evaluated)
    # print(p_R_evaluated)

    # print('Calculation of ratio...\n')  # From DOMIAS evaluator.py but flipped - their eqn. hence mention in report
    # Therefore, the ratio needs to be reference/synthetic evaluation, where a ratio of larger than 1 will
    # indicate better fit to synthetic data
    p_rel = p_R_evaluated/p_G_evaluated

    '''MIA Success'''  # From DOMIAS baselines.py
    # print('Deduce preducted members above median p_rel value...\n')
    y_pred = p_rel > np.median(p_rel)

    print('Calculating ACC and AUC...')
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, p_rel)
    print('     ACC:', acc, '| AUC:', auc, '\n')

    return acc, auc, y_pred


def time_sensitive_kde():
    """See my Notion for 'my method'. """
    pass


def kde_mia_domias_and_euclidian():
    pass
