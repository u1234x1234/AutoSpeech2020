from functools import partial

import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from typing import Type


class ClassDesc:
    "Class + arguments"
    def __init__(self, class_: Type, *args, **kwargs):
        self.class_ = class_
        self.args = args
        self.kwargs = kwargs

    def instantiate(self):
        return self.class_(*self.args, **self.kwargs)

    def __repr__(self):
        ar = ', '.join([str(x) for x in self.args])
        kw = ', '.join([f'{k}={v}' for k, v in self.kwargs.items()])
        arg_str = ', '.join([x for x in [ar, kw] if x])
        return f'{self.class_.__name__}({arg_str})'


def winsorized_min_max_scaler(X, a=0, b=1, p1=0.05, p2=0.95):
    p1, p2 = np.percentile(X, [p1, p2])
    X = np.clip(X, p1, p2)
    X = preprocessing.minmax_scale(X, feature_range=(a, b))
    return X


class Processor:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, *args, **kwargs):
        return self

    def transform(self, X, *args, **kwargs):
        raise NotImplementedError()

    def fit_transform(self, X, *args, **kwargs):
        return self.fit(X, *args).transform(X, *args)


class Noop(Processor):
    def transform(self, X):
        return X


class L1Normalizer(Processor):
    def transform(self, X):
        return preprocessing.normalize(X, norm='l1')


class L2Normalizer(Processor):
    def transform(self, X):
        return preprocessing.normalize(X, norm='l2')


class SignSqrt(Processor):
    def transform(self, X):
        return np.sign(X) * np.sqrt(np.abs(X))


class SignLog(Processor):
    def transform(self, X):
        return np.sign(X) * np.log(np.abs(X))


# TODO leaky stateful preprocessors
PREPROCESSORS = {
    # Stateless
    'noop': ClassDesc(Noop),

    'normalize': ClassDesc(L2Normalizer),
    'normalize_l1': ClassDesc(L1Normalizer),
    'normalize_l2': ClassDesc(L2Normalizer),
    'sign_sqrt': ClassDesc(SignSqrt),
    'sign_log': ClassDesc(SignLog),

    # Stateful, do not instantiate
    'standartize': ClassDesc(preprocessing.StandardScaler, with_mean=True, with_std=True),
    'standartize_mean_std': ClassDesc(preprocessing.StandardScaler, with_mean=True, with_std=True),
    'standartize_mean': ClassDesc(preprocessing.StandardScaler, with_mean=True, with_std=False),
    'standartize_std': ClassDesc(preprocessing.StandardScaler, with_mean=False, with_std=True),

    'minmax_scale': ClassDesc(preprocessing.MinMaxScaler),
    'maxabs_scale': ClassDesc(preprocessing.MaxAbsScaler),
}


class Preprocessor:
    def __init__(self, alg):
        self.alg = alg

        p = PREPROCESSORS.get(alg, None)
        if p is None:
            if alg.startswith('pca_'):
                n_components = int(alg[4:])
                p = ClassDesc(TruncatedSVD, n_components=n_components)
            else:
                raise ValueError(f'Unknown alg: {alg}')

        if isinstance(p, ClassDesc):
            self._preprocessor = p.instantiate()
        else:
            self._preprocessor = p

    def fit(self, X):
        if callable(self._preprocessor):
            return self
        else:
            return self._preprocessor.fit(X)

    def transform(self, X):
        if callable(self._preprocessor):
            return self._preprocessor(X)
        else:
            return self._preprocessor.transform(X)

    def fit_transform(self, X):
        if callable(self._preprocessor):
            return self._preprocessor(X)
        else:
            return self._preprocessor.fit_transform(X)


# TODO fixed, mirror
def to_fixed_len_1d(X, n=32000, alg='tile', to_numpy=True, dtype=np.float32):
    "Transform sequence[of list of sequences] of variable len to the fixed len"

    def transform_single(x):
        if len(x) < n:
            if alg == 'tile':
                n_rep = np.math.ceil(n / len(x))
                x = np.tile(x, n_rep)
            elif alg == 'pad':
                x = np.pad(x, (0, n - len(x)))
            else:
                raise ValueError(f'alg "{alg}" is not implemented yet')
        return x[:n]

    def len_depth(x):
        try:
            return len_depth(x[0]) + 1
        except Exception:
            return 0

    d = len_depth(X)
    if d == 1:
        X = transform_single(X)
    elif d == 2:
        X = [transform_single(x) for x in X]
    else:
        raise ValueError('Incorrect shapes')

    if to_numpy:
        X = np.array(X, dtype=dtype)

    return X
