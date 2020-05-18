import inspect
import os
import sys
import warnings
from contextlib import contextmanager


def is_in_ipython():
    try:
        return __IPYTHON__
    except NameError:
        return False


class RedirectStdStreams:
    def __init__(self, stdout=None, stderr=None):
        devnull = open(os.devnull, 'w')
        if stdout is None:
            stdout = devnull
        if stderr is None:
            stderr = devnull

        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


# Adapted from https://stackoverflow.com/a/57677370
class SuppressNativeIOStream:
    def __init__(self, stream):
        self.original_fileno = stream.fileno()

    def __enter__(self):
        self.original_stream = os.dup(self.original_fileno)
        self.devnull = open(os.devnull, 'w')
        os.dup2(self.devnull.fileno(), self.original_fileno)

    def __exit__(self, *args):
        os.close(self.original_fileno)
        os.dup2(self.original_stream, self.original_fileno)
        os.close(self.original_stream)
        self.devnull.close()


@contextmanager
def ignore_tensorflow_logger():
    initial_verbosity = None
    try:
        import tensorflow as tf
        initial_verbosity = tf.compat.v1.logging.get_verbosity()
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except NameError:
        pass

    yield

    # Recover initial value
    try:
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(initial_verbosity)
    except NameError:
        pass


@contextmanager
def ignore_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@contextmanager
def suppres_all_output():
    "Ignore stdout, stderr from the underlying python or native code"

    with ignore_warnings():

        # ipython could patch streams => suppress original __stdou__
        if is_in_ipython():
            with SuppressNativeIOStream(sys.__stdout__), SuppressNativeIOStream(sys.__stderr__):
                yield
        else:
            with SuppressNativeIOStream(sys.stdout), SuppressNativeIOStream(sys.stderr):
                yield


def disable_gpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def enable_gpu(devices: str='0'):
    os.environ['CUDA_VISIBLE_DEVICES'] = devices


@contextmanager
def modify_python_path(path):
    sys.path.insert(0, path)
    yield
    # remove the first occurence => does not break the original order
    sys.path.remove(path)
