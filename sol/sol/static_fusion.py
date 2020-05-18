"""Fusion of features from the different modalities.
# TODO interactions
"""
import numpy as np

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


# TODO when to augment: on fit, on transform, on both?
def augment(X, alg='gaussian_noise'):

    if alg == 'gaussian_noise':
        X = X + np.random.normal(0, 0.01, size=X.shape)
    elif alg.startswith('data_percent_'):
        percent = float(alg[len('data_percent_'):]) / 100.
        N = int(X.shape[0] * percent)
        assert 2 < N < X.shape[0] - 2
        return np.random.permutation(X)[:N]
    else:
        raise NotImplementedError(f'Alg: {alg} not implemented')

    return X


class StaticFusion:
    def __init__(self, pre_processor: ClassDesc, post_processor: ClassDesc):
        self.pre_processor = pre_processor
        self.post_processor = post_processor

        self._fitted_pre_processors = []
        self._fitted_post_processor = None

    def fit_transform(self, *Xs):
        # Preprocess independently
        nXs = []
        for X in Xs:
            p = self.pre_processor.instantiate()
            p.fit(augment(X))
            nXs.append(p.transform(X))
            self._fitted_pre_processors.append(p)

        # Concat features
        nXs = np.hstack(nXs)

        # Postprocess
        self._fitted_post_processor = self.post_processor.instantiate()
        nXs = self._fitted_post_processor.fit_transform(nXs)

        return nXs

    def transform(self, *Xs):
        assert self._fitted_post_processor, 'call a fit_transform method before transform'

        nXs = [
            self._fitted_pre_processors[idx].transform(X)
            for idx, X in enumerate(Xs)]

        nXs = np.hstack(nXs)
        nXs = self._fitted_post_processor.transform(nXs)

        return nXs
