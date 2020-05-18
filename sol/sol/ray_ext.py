import atexit
import copy
import inspect
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Callable, List

import numpy as np
import pandas as pd
import ray
# import tabulate
from ray import tune
from ray.tune import Trainable, track
from ray.tune.experiment import Experiment
from ray.tune.logger import CSVLogger, JsonLogger
from ray.tune.ray_trial_executor import RayTrialExecutor
from ray.tune.schedulers import AsyncHyperBandScheduler, FIFOScheduler
from ray.tune.suggest import BasicVariantGenerator
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.syncer import wait_for_sync
from ray.tune.trial import Trial
from ray.tune.trial_runner import TrialRunner
from ray.tune.web_server import TuneServer

from .system_ext import suppres_all_output

SCORE_NAME = 'score'
TRAINING_STEP_NAME = 'training_iteration'


def _create_trainable(model_cls, metric, data_getter, seconds_per_step):

    class TimeBudgetedTrainable(Trainable):
        def _setup(self, config):
            self.model = model_cls(**config)
            self.metric = metric
            self.config = config
            self.data_getter = data_getter
            self.timestep = 0

        def _train(self):
            self.timestep += 1

            X, y, cv = self.data_getter()
            cv_scores = []
            # TODO separate model per cv split
            for train_idx, val_idx in cv:
                y_train, y_val = y[train_idx], y[val_idx]
                if isinstance(X, np.ndarray):
                    X_train, X_val = X[train_idx], X[val_idx]
                else:
                    X_train = [X[idx] for idx in train_idx]
                    X_val = [X[idx] for idx in val_idx]

                self.model.fit(X_train, y_train.argmax(axis=1), n_seconds=seconds_per_step)
                y_pred = self.model.predict(X_val)
                score = self.metric(y_val, y_pred)
                cv_scores.append(score)

            score = np.mean(cv_scores)  # TODO why only mean, return predictions?

            return {SCORE_NAME: score}

        def _save(self, checkpoint_dir):
            m_path = os.path.join(checkpoint_dir, "model")
            os.makedirs(m_path, exist_ok=True)
            self.model.save(m_path)
            return m_path

        def _restore(self, checkpoint_path):
            self.model.restore(checkpoint_path)

    return TimeBudgetedTrainable


@contextmanager
def ray_context(num_cpus=8, num_gpus=0, memory_mb=20000, force_reinit=True):
    "Ray server context manager"
    start_ray(num_cpus=num_cpus, num_gpus=num_gpus, memory_mb=memory_mb, force_reinit=force_reinit)
    yield ray
    ray.shutdown()


def start_ray(num_cpus=8, num_gpus=0, memory_mb=8000, force_reinit=True):
    memory = int(memory_mb*1e6)
    assert 1e6 < memory < 1e12

    if force_reinit:
        if ray.is_initialized():
            ray.shutdown()

    ray.init(
        ignore_reinit_error=True,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        configure_logging=False,
        include_webui=False,
        memory=memory,
        object_store_memory=memory,
    )


@contextmanager
def provide_ray(num_cpus=8, num_gpus=0, memory_mb=8000):
    """Ensure that ray is running == if True:return else: ray.init.
    TODO check that there are enough resources
    """
    if ray.is_initialized():
        yield ray
    else:
        with ray_context(num_cpus=num_cpus, num_gpus=num_gpus, memory_mb=memory_mb) as r:
            yield r


def optimize_class(trainable_class: Callable, search_space: dict, metric: Callable, data,
                   time_budget=30, seconds_per_step=10, max_t=4, reduction_factor=3,
                   num_gpus=0, num_cpus=4, gpu_per_trial=0, cpu_per_trial=1, show_progress=True,
                   checkpoint_freq=1, checkpoint_at_end=False):
    "Search for the best hyperparameters with Successive halving-type optimization"

    with ray_context(num_gpus=num_gpus, num_cpus=num_cpus):
        data_id = ray.put(data)

        def data_getter():  # TODO check speed up
            return ray.get(data_id)

        trainable = _create_trainable(trainable_class, metric, data_getter, seconds_per_step=seconds_per_step)

        search_alg = BasicVariantGenerator(
            shuffle=True
        )

        scheduler = AsyncHyperBandScheduler(metric='score', max_t=max_t, reduction_factor=reduction_factor)

        results = run_async_executor(
            trainable=trainable,
            config=search_space,
            time_budget=time_budget,
            scheduler=scheduler,
            search_alg=search_alg,
            gpu_per_trial=gpu_per_trial,
            cpu_per_trial=cpu_per_trial,
            show_progress=show_progress,
            checkpoint_at_end=checkpoint_at_end,
            checkpoint_freq=checkpoint_freq,
        )

    return results


def wrap_function(train_func):

    sign = inspect.signature(train_func).parameters
    names = list(sign.keys())

    def fn(config):
        if names == ['config']:
            output = train_func(config)
        else:
            output = train_func(**config)

        output = float(output)
        track.log(**{SCORE_NAME: output})

    return fn


def _transform_config(config):
    new_config = {}
    for k, v in config.items():
        if isinstance(v, list):
            new_config[k] = tune.grid_search(v)
        elif callable(v):
            new_config[k] = tune.sample_from(v)
        else:
            raise ValueError(f'Incorrect config: {config}')

    return new_config


class AsyncExecutor:
    "Async version of tune.run(...)"
    def __init__(
            self,
            run_or_experiment,
            name=None,
            stop=None,
            config=None,
            resources_per_trial=None,
            num_samples=1,
            local_dir=None,
            upload_dir=None,
            trial_name_creator=None,
            loggers=None,
            sync_to_cloud=None,
            sync_to_driver=False,
            checkpoint_freq=0,
            checkpoint_at_end=False,
            sync_on_checkpoint=True,
            keep_checkpoints_num=None,
            checkpoint_score_attr=None,
            global_checkpoint_period=10,
            export_formats=None,
            max_failures=0,
            fail_fast=True,
            restore=None,
            search_alg=None,
            scheduler=None,
            with_server=False,
            server_port=TuneServer.DEFAULT_PORT,
            verbose=0,
            progress_reporter=None,
            resume=False,
            queue_trials=False,
            reuse_actors=False,
            trial_executor=None,
            raise_on_failed_trial=True,
            return_trials=False,
            ray_auto_init=True,
            shuffle=False):

        if loggers is None:
            loggers = [JsonLogger, CSVLogger]
        config = _transform_config(config)

        is_trainable = False
        try:
            if issubclass(run_or_experiment, Trainable):
                is_trainable = True
        except TypeError:
            pass

        if not is_trainable:
            run_or_experiment = wrap_function(run_or_experiment)

        self.trial_executor = trial_executor or RayTrialExecutor(
            queue_trials=queue_trials,
            reuse_actors=reuse_actors,
            ray_auto_init=ray_auto_init)

        experiments = [run_or_experiment]
        self.logger = logging.getLogger(__name__)

        for i, exp in enumerate(experiments):
            if not isinstance(exp, Experiment):
                run_identifier = Experiment.register_if_needed(exp)
                experiments[i] = Experiment(
                    name=name,
                    run=run_identifier,
                    stop=stop,
                    config=config,
                    resources_per_trial=resources_per_trial,
                    num_samples=num_samples,
                    local_dir=local_dir,
                    upload_dir=upload_dir,
                    sync_to_driver=sync_to_driver,
                    trial_name_creator=trial_name_creator,
                    loggers=loggers,
                    checkpoint_freq=checkpoint_freq,
                    checkpoint_at_end=checkpoint_at_end,
                    sync_on_checkpoint=sync_on_checkpoint,
                    keep_checkpoints_num=keep_checkpoints_num,
                    checkpoint_score_attr=checkpoint_score_attr,
                    export_formats=export_formats,
                    max_failures=max_failures,
                    restore=restore)

        if fail_fast and max_failures != 0:
            raise ValueError("max_failures must be 0 if fail_fast=True.")

        self.runner = TrialRunner(
            search_alg=search_alg or BasicVariantGenerator(shuffle=shuffle),
            scheduler=scheduler or FIFOScheduler(),
            local_checkpoint_dir=experiments[0].checkpoint_dir,
            remote_checkpoint_dir=experiments[0].remote_checkpoint_dir,
            sync_to_cloud=sync_to_cloud,
            stopper=experiments[0].stopper,
            checkpoint_period=global_checkpoint_period,
            resume=resume,
            launch_web_server=with_server,
            server_port=server_port,
            verbose=bool(verbose > 1),
            fail_fast=fail_fast,
            trial_executor=self.trial_executor)

        for exp in experiments:
            self.runner.add_experiment(exp)

        self._is_worker_stopped = threading.Event()
        self._worker_exc = None
        self._worker = threading.Thread(target=self.step_worker, daemon=True)
        self._worker.start()

        atexit.register(self.stop)

    def step_worker(self):
        while not self._is_worker_stopped.is_set() and not self.runner.is_finished():
            try:
                self.runner.step()  # blocking call!
            except Exception:
                self._is_worker_stopped.set()
                self._worker_exc = sys.exc_info()

    def stop(self, timeout=5):

        self.runner.request_stop_experiment()
        self._is_worker_stopped.set()

        # FORCE KILL, mute all the errors from the dying subprocesses
        for t in self.trial_executor.get_running_trials():
            try:  # TODO ?? ValueError: ray.kill() only supported for actors. Got: .
                ray.kill(t.runner)
            except Exception:
                pass
            self.trial_executor.stop_trial(t, True)
            time.sleep(0.5)  # wait for stdio sync

        self._worker.join(timeout=timeout)
        assert self._worker.is_alive() is False

    def get_trials(self):
        return self.runner.get_trials()

    def get_results(self):
        # Reraise from the worker thread
        if self._worker_exc:
            raise self._worker_exc[1].with_traceback(self._worker_exc[2])

        trials = self.runner.get_trials()
        try:
            self.runner.checkpoint(force=True)
        except Exception:
            self.logger.exception("Trial Runner checkpointing failed.")
        wait_for_sync()

        completed_results = []
        n_incompleted = 0
        for trial in trials:

            if len(trial.metric_analysis) > 0:
                score = trial.metric_analysis[SCORE_NAME]['last']
                it = trial.metric_analysis[TRAINING_STEP_NAME]['last']

                result = {
                    SCORE_NAME: score,
                    TRAINING_STEP_NAME: it,
                    'logdir': trial.logdir,
                    'config': trial.config,
                }
                completed_results.append(result)

            if trial.status != Trial.TERMINATED:
                n_incompleted += 1
                continue

        return completed_results, n_incompleted


def run_async_executor(trainable, config, time_budget,
                       cpu_per_trial, gpu_per_trial,
                       num_samples=1, show_progress=False,
                       scheduler=None, search_alg=None, checkpoint_at_end=False, checkpoint_freq=0):

    results = []

    resources_per_trial = {'cpu': cpu_per_trial, 'gpu': gpu_per_trial}
    executor = AsyncExecutor(
        trainable,
        config=config,
        num_samples=num_samples,
        resources_per_trial=resources_per_trial,
        shuffle=True,
        scheduler=scheduler,
        search_alg=search_alg,
        checkpoint_at_end=checkpoint_at_end,
        checkpoint_freq=checkpoint_freq,
    )

    start_time = time.time()

    # Wait for the ray initialization
    for _ in range(min(30, time_budget)):
        time.sleep(1)
        n_trials = len(executor.get_trials())
        if n_trials:
            break

    while (time.time() - start_time) < time_budget:
        time.sleep(1)
        results, n_incompleted = executor.get_results()

        if n_incompleted == 0:
            break

    executor.stop()

    return results
