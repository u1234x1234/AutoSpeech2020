import importlib
import inspect
import logging
import os
import sys
from contextlib import contextmanager

import appdirs
import requests

APPLICATION_NAME = 'audio_mlgate'

# TODO model backups
DOWNLOAD_URLS = {
    'baseline_lite_ap.model': 'http://www.robots.ox.ac.uk/~vgg/data/voxceleb/models/baseline_lite_ap.model',
}


def download_file(url, out_path):
    response = requests.get(url, stream=True)
    with open(out_path, "wb") as out_file:
        for data in response.iter_content():
            out_file.write(data)


@contextmanager
def load_external(name):
    dn = os.path.dirname(os.path.abspath(__file__))
    ext_path = f'{dn}/../3rdparty/{name}/'
    sys.path.append(ext_path)
    yield
    sys.path.remove(ext_path)


def get_model_path(model_name):
    assert model_name in DOWNLOAD_URLS

    root_model_dir = appdirs.user_data_dir(APPLICATION_NAME)
    os.makedirs(root_model_dir, exist_ok=True)

    model_path = os.path.join(root_model_dir, model_name)
    if not os.path.exists(model_path):
        logging.info(f'Model {model_name} not found. Downloadin...')
        download_file(DOWNLOAD_URLS[model_name], model_path)

    # TODO assert hash

    return model_path


def lazy_module_import(module_name):
    "Import module with `module_name` into a caller global scope"
    caller_frame = inspect.stack()[1]
    caller_module = inspect.getmodule(caller_frame[0])
    caller_module.__dict__[module_name] = importlib.import_module(module_name)
