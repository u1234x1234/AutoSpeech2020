import glob
import importlib
import inspect
import os
import shutil
import sys
import time
import warnings
from functools import partial

import joblib
import numpy as np
import torch
from scipy.stats import gmean
from sklearn.linear_model import LogisticRegression

from .model_utils import balanced_accuracy
from .model_utils import roc_auc_score_safe as balanced_accuracy
from .model_utils import roc_auc_score_safe
from .model_utils import time_pooling
from .networks import PretrainedSpeakerEmbedding
from .parametric_family import ParametricFamilyModel
from .preprocessing import ClassDesc, Preprocessor
from .se_finetune import _load_audio, create_factory_method, search_space
from .static_fusion import StaticFusion
from .system_ext import suppres_all_output
from .torch_ext import dataset_factory, predict_dataset
from .validation import validation_procedure

MODEL_NAME = 'baseline_lite_ap.model'
SAMPLE_RATE = 16000
TIME_BUDGET = 1800


def extract_musicnn_features(data):
    X_train, y, X_test, cv = data
    with suppres_all_output():
        from audio_mlgate.music_genre import MusicGenre
        fe = MusicGenre()
        X_train = fe.extract_features(X_train, SAMPLE_RATE)
        X_test = fe.extract_features(X_test, SAMPLE_RATE)
    return X_train, X_test


def test_apply(x, model):
    if len(x.shape) > 2:
        bs, aug = x.shape[:2]  # Avg TTA (batch_size, aug_dim, features_dims)
        x = x.view((bs*aug, -1))
        out = model(x).view((bs, aug, -1))
    else:
        out = model(x)
    return out


def make_inference(model, X1, X2, device, max_len=3, num_eval=1):
    if max_len > 5 or num_eval > 5:
        batch_size = 2
    else:
        batch_size = 8
    ds = dataset_factory(partial(_load_audio, max_len=max_len, evalmode=True, num_eval=num_eval))
    X1 = predict_dataset(ds(X1), model, device, batch_size=batch_size, test_apply=test_apply)
    X2 = predict_dataset(ds(X2), model, device, batch_size=batch_size, test_apply=test_apply)
    return X1, X2


def lazy_module_import(module_name):
    "Import module with `module_name` into a caller global scope"
    caller_frame = inspect.stack()[1]
    caller_module = inspect.getmodule(caller_frame[0])
    caller_module.__dict__[module_name] = importlib.import_module(module_name)


def fit_single(prep: ClassDesc, model, cv, X, y, X_test, metric=None):
    """
    Returns:
        tuple: (cv_score, test predictions)
    """
    n = len(X) // len(y)
    if n > 1:
        y = np.vstack([y for _ in range(n)])

    score_per_split = []
    for train_idx, val_idx in cv:
        y_train, y_val = y[train_idx], y[val_idx]
        if isinstance(X, np.ndarray):
            X_train, X_val = X[train_idx], X[val_idx]
        else:
            X_train = [X[idx] for idx in train_idx]
            X_val = [X[idx] for idx in val_idx]

        if prep is not None:
            preprocessor = prep.instantiate()
            X_train = preprocessor.fit_transform(X_train)
            X_val = preprocessor.transform(X_val)

        model.fit(X_train, y_train.argmax(axis=1))
        y_pred = model.predict_proba(X_val)
        score = balanced_accuracy(y_val, y_pred)
        score_per_split.append(score)
    score = np.mean(score)

    # refit on the full data
    if prep is not None:
        preprocessor = prep.instantiate()
        X = preprocessor.fit_transform(X)
        X_test = preprocessor.transform(X_test)

    model.fit(X, y.argmax(axis=1))
    y_pred = model.predict_proba(X_test)
    return score, y_pred


def patched_metric(solution, prediction):
    if solution.sum(axis=0).min() == 0:
        return np.nan
    return balanced_accuracy(solution, prediction)


def top1_2019_hazza_cheng(module_path, out_path, n_classes, data, metric):
    # the great solution provided by Hazza Chenh
    # https://github.com/HazzaCheng/AutoSpeech

    data_train, y, data_test, cv = data
    sys.path.append(module_path)

    from model import Model
    m = Model({'class_num': n_classes, 'train_num': len(data_train), 'test_num': len(data_test)})

    # By default, roc-auc is used as a metric
    # Let's just hack into, without modification of the original code
    import model_manager

    def patched_split(self, ratio):
        train_idx, val_idx = cv[0]
        return train_idx.copy(), val_idx.copy()

    from data_manager import DataManager
    PatchedDataManager = DataManager
    PatchedDataManager._train_test_split_index = patched_split
    import model
    model.__dict__['DataManager'] = PatchedDataManager
    model_manager.__dict__['auc_metric'] = patched_metric

    os.makedirs(out_path, exist_ok=True)
    prev_best = 0
    for iteration_idx in range(10000):
        m.train((data_train, y))
        y_pred_test = m.test(data_test)
        score = m.model_manager._k_best_auc[0]
        if iteration_idx % 10 == 0:
            if score > prev_best:
                joblib.dump((score, y_pred_test), f'{out_path}/{iteration_idx}.pkl.lzma')
                prev_best = score


def top3_2019_kon(module_path, out_path, n_classes, data, metric):
    data_train, y, data_test, cv = data
    sys.path.append(module_path)

    from model import Model
    meta = {
        'class_num': n_classes, 'train_num': len(data_train),
        'test_num': len(data_test), 'time_budget': TIME_BUDGET}
    m = Model(meta, cv, metric)

    os.makedirs(out_path, exist_ok=True)
    prev_best = 0
    start_time = time.time()
    for iteration_idx in range(10000):
        m.train((data_train, y), TIME_BUDGET - (time.time() - start_time))
        y_pred_test = m.test(data_test)
        score = m.max_score
        if score > prev_best:
            joblib.dump((score, y_pred_test), f'{out_path}/{iteration_idx}.pkl.lzma')
            prev_best = score


class MetaModel:
    def __init__(self, n_classes, weights_path, root_path):
        self.n_classes = n_classes
        self.iteration = 0
        self.root_path = root_path
        self.metric = roc_auc_score_safe

        self.X = None
        self.y = None
        self.cv = None
        self.out_path_top1_2019 = '/tmp/top1_2019'
        self.out_path_top3_2019 = '/tmp/top3_2019'
        shutil.rmtree(self.out_path_top1_2019, ignore_errors=True)
        shutil.rmtree(self.out_path_top3_2019, ignore_errors=True)

        se_path = os.path.join(weights_path, MODEL_NAME)
        self.device = torch.device('cuda')
        self.se_extractor = PretrainedSpeakerEmbedding(se_path)
        self.se_extractor.to(self.device)
        self.pm = ParametricFamilyModel(
            create_factory_method(n_classes, se_path), search_space, balanced_accuracy,
            n_cpus=2, n_gpus=0.2, gpu_per_trial=0.1, cpu_per_trial=1)

        self.predictions = []  # (name, score, probs)
        lazy_module_import('ray')
        ray.init(
            num_gpus=1, num_cpus=4, memory=10e9, object_store_memory=10e9,
            configure_logging=False, ignore_reinit_error=True,
            log_to_driver=False,
            include_webui=False)

        self.features = {}
        self.features_completed = set()
        warnings.simplefilter("ignore")

    def train_for_budget(self, X, y):
        self.iteration += 1
        if self.X is None:
            self.X = X
            self.y = y
            if len(X) < 400:
                self.cv = validation_procedure(X, y, 'stratified_shuffle_split_3_30')
            else:
                self.cv = validation_procedure(X, y, 'stratified_shuffle_split_1_30')

        if self.iteration > 10000:
            return True
        else:
            return False

    def predict(self, X_test):
        try:
            self.step(X_test, self.iteration)

            predictions = []
            top_predictions = list(sorted(self.predictions, key=lambda x: -x[0]))

            for i, (score, name, y_pred) in enumerate(top_predictions[:40]):
                if i and score == top_predictions[i-1][0]:  # same score
                    continue
                if score < top_predictions[0][0] * 0.92:  # relaxed condition on the later iterations
                    continue

                print(score, name)
                predictions.append(y_pred)
                if len(predictions) > 25:
                    break

            print('##############')

            predictions = np.stack(predictions)
            predictions = gmean(predictions, axis=0)
        except Exception as e:
            print(e)
            predictions = np.zeros((len(X_test), self.n_classes))
        return predictions

    def step(self, X_test, iteration):

####################################### Hardcoded sequences of actions ####################################################
        if iteration == 1:  # as fast as possible
            X_train, X_test = make_inference(
                self.se_extractor, self.X, X_test, self.device, max_len=3, num_eval=1)

            model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=20)
            model.fit(X_train, self.y.argmax(axis=1))
            y_pred = model.predict_proba(X_test)
            self.predictions.append((1, 'first', y_pred))
            self.features['se_3_1'] = (X_train, X_test)
            return

        if iteration == 2:
            # schedule a non-blocking op
            self.data_id = ray.put((self.X, self.y, X_test, self.cv))
            self.X_music = ray.remote(num_gpus=0.25, num_cpus=1, max_calls=1)(
                extract_musicnn_features).remote(self.data_id)

            self.predictions.clear()
            X_tr, X_t = self.features['se_3_1']
            model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=100)
            score, y_pred = fit_single(None, model, self.cv, X_tr, self.y, X_t, self.metric)
            self.predictions.append((score, 'first', y_pred))

            module_path = f'{self.root_path}/3rdparty/autospeech19/'
            self.solution3_2019 = ray.remote(num_gpus=0.25, num_cpus=1, max_calls=1)(
                top3_2019_kon).remote(module_path, self.out_path_top3_2019, self.n_classes, self.data_id, self.metric)

            module_path = f'{self.root_path}/3rdparty/AutoSpeech/code_submission'
            self.solution1_2019 = ray.remote(num_gpus=0.25, num_cpus=1, max_calls=1)(
                top1_2019_hazza_cheng).remote(module_path, self.out_path_top1_2019, self.n_classes, self.data_id, self.metric)

        if iteration == 3:
            X_train, X_test = make_inference(
                self.se_extractor, self.X, X_test, self.device, max_len=5, num_eval=5)

            self.features['se_5_5'] = (X_train.mean(axis=1), X_test.mean(axis=1))
            X_train = X_train.transpose(1, 0, 2).reshape((-1, X_train.shape[-1]))
            self.features['se_5_5_expand'] = (X_train, X_test.mean(axis=1))

        if iteration == 10:
            X_train, X_test = make_inference(
                self.se_extractor, self.X, X_test, self.device, max_len=10, num_eval=5)
            self.features['se_10_5'] = (X_train.mean(axis=1), X_test.mean(axis=1))

        if iteration == 15:
            X_train, X_test = make_inference(
                self.se_extractor, self.X, X_test, self.device, max_len=15, num_eval=5)
            self.features['se_15_5'] = (X_train.mean(axis=1), X_test.mean(axis=1))

        if iteration == 20:
            X_train, X_test = make_inference(
                self.se_extractor, self.X, X_test, self.device, max_len=10, num_eval=10)
            self.features['se_10_10'] = (X_train.mean(axis=1), X_test.mean(axis=1))

##################################### Check for external results ############################################################

        try:  # TODO add interprocess filelock (concurrent read/write)
            for root_path in [self.out_path_top1_2019, self.out_path_top3_2019]:

                ext_paths = glob.glob(root_path + '/*.pkl.lzma')
                for path in ext_paths:
                    score, y_pred = joblib.load(path)
                    name = os.path.basename(path).split('.')[0]
                    name = f'{os.path.dirname(path)}_{name}'
                    names = set([x[1] for x in self.predictions])
                    if name in names:
                        continue
                    print('extresult', score, name)
                    self.predictions.append((score, name, y_pred))
        except:
            pass

########################################### Train a model #########################################################

        # Check whether the futures are ready
        if 'mg' not in self.features:
            ready, nready = ray.wait([self.X_music], timeout=0.1)
            if ready:
                X_tr, X_te = ray.get(self.X_music)
                X_tr = time_pooling(X_tr, 'mean')
                X_te = time_pooling(X_te, 'mean')
                assert len(X_tr.shape) == 2 and len(X_te.shape) == 2
                print('>' * 400, 'MG ready')
                self.features['mg'] = (X_tr, X_te)
                self.pm.start(self.data_id, time_budget=TIME_BUDGET, seconds_per_step=10, max_t=5, reduction_factor=4)

        # Select candidates to train
        candidates = []
        for model_suf in ['standartize', 'normalize', 'noop', 'sign_sqrt']:
            for fname in self.features:
                out_name = f'{fname}_{model_suf}'
                if out_name in self.features_completed:
                    continue
                candidates.append((model_suf, fname, out_name))

        # Train random candidate
        if candidates and np.random.rand() < 0.8:  # sometimes skip this step
            r_idx = np.random.randint(0, len(candidates))
            model_suf, fname, out_name = candidates[r_idx]

            prep = ClassDesc(Preprocessor, model_suf)
            X_train, X_test = self.features[fname]
            max_iter = 100 if len(X_train) < 1500 else 30
            model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=max_iter)
            score, y_pred = fit_single(prep, model, self.cv, X_train, self.y, X_test)
            self.features_completed.add(out_name)
            self.predictions.append((score, out_name, y_pred))
            return  # one model per iteration

        if self.pm.executor is not None:
            pm_results = self.pm.executor.get_results()[0]
            if pm_results:
                pm_results = list(sorted(pm_results, key=lambda x: -x['score']))
                pm_top_score = pm_results[0]['score']

                top_predictions = list(sorted(self.predictions, key=lambda x: -x[0]))
                if pm_top_score > top_predictions[0][0] - 0.03:
                    refit_seconds = 10 if self.iteration < 30 else 20
                    y_pred = self.pm.predict(X_test, refit_seconds=refit_seconds)
                    name = f'pm_{pm_top_score}'
                    if len(pm_results) > 1:
                        pm_top2_score = pm_results[1]['score']
                        name = f'pm_{pm_top_score}_{pm_top2_score}'

                    self.predictions.append((pm_top_score, name, y_pred))

        if 'mg' in self.features:
            X1_train, X1_test = self.features['mg']
            for model_suf in ['standartize', 'normalize', 'noop', 'sign_sqrt']:
                for fname in self.features:
                    if 'expand' in fname:
                        continue
                    out_name = f'mg_{fname}_fus_{model_suf}'
                    if out_name in self.features_completed:
                        continue

                    X2_train, X2_test = self.features[fname]
                    pre = ClassDesc(Preprocessor, model_suf)
                    post = ClassDesc(Preprocessor, model_suf)
                    fus = StaticFusion(pre, post)
                    X_train = fus.fit_transform(X1_train, X2_train)
                    X_test = fus.transform(X1_test, X2_test)
                    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')

                    score, y_pred = fit_single(None, model, self.cv, X_train, self.y, X_test)
                    self.features_completed.add(out_name)
                    self.predictions.append((score, out_name, y_pred))
                    return

    def __del__(self):
        self.pm.executor.stop()
        import ray
        ray.shutdown()
