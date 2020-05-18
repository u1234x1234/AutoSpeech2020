"""Validation procedures
TODO automatic selection of the best validation procedure conditioned on the data
"""
import re
from sklearn.model_selection import StratifiedShuffleSplit


def _stratified_shuffle_split(match, X, y):
    n_splits, test_size = match.groups()
    n_splits = int(n_splits)
    test_size = float(test_size) / 100.
    assert 0 < n_splits < 100
    assert 0 < test_size < 1.
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    return list(cv.split(X, y))


CV_MATCHERS = [
    (re.compile(r'stratified_shuffle_split_(\d+)_(\d+)'), _stratified_shuffle_split),
]


# TODO Why do we need this?
def validation_procedure(X, y, alg='stratified_shuffle_split_1_30'):
    """
    Returns
        cv: List[train_idx, val_idx]
    """
    for matcher, handler in CV_MATCHERS:
        match = matcher.match(alg)
        if match is not None:
            return handler(match, X, y)

    raise ValueError(f'No such option "{alg}" for alg')
