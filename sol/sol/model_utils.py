import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def make_gaussian_kernel_1d(size: int, sigma: float = 1):
    x = np.linspace(-sigma, sigma, size+1)
    y = np.diff(stats.norm.cdf(x))
    return y


def gaussian_pooling(x):
    "Centered weighted pooling"
    weights = make_gaussian_kernel_1d(x.shape[0])[:, np.newaxis]
    return weights*x


def time_pooling(X, alg, axis=0):
    "Squeeze time dimension. Convert List[np.ndarray2d] -> np.ndarray2d"

    if alg == 'mean':
        X = np.array([x.mean(axis=axis) for x in X])
    elif alg == 'max':
        X = np.array([x.max(axis=axis) for x in X])
    elif alg == 'min':
        X = np.array([x.min(axis=axis) for x in X])
    elif alg == 'first':
        X = np.array([np.take(x, 0, axis=axis) for x in X])
    elif alg == 'last':
        X = np.array([np.take(x, -1, axis=axis) for x in X])
    elif alg == 'gaussian':
        X = np.array([gaussian_pooling(x.squeeze()).mean(axis=axis) for x in X])
    else:
        raise ValueError(f'No such alg: "{alg}"')

    return X.squeeze()


def roc_auc_score_safe(y_true, y_pred):
    assert len(y_true.shape) == 2
    assert len(y_pred.shape) == 2
    assert y_true.shape == y_pred.shape
    scores = []
    for class_idx in range(y_true.shape[1]):
        try:
            class_score = roc_auc_score(y_true[:, class_idx], y_pred[:, class_idx])
        except Exception:
            class_score = 0.5

        scores.append(class_score)
    score = np.array(scores).mean()  # Class averaging
    return score


def balanced_accuracy(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = y_pred.argmax(axis=1)

    C = confusion_matrix(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    return score
