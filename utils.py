import os
import sys
from subprocess import check_call


def install_pip_package(name, upgrade: bool = True):
    # options = '--upgrade' if upgrade else ''
    command = f'{sys.executable} -m pip install {name}'
    retcode = check_call([command], shell=True)
    return retcode


def prepare_env():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    dn = os.path.dirname(__file__)

    # Local wheels
    # install_pip_package(f'--no-index --find-link {dn}/pypi/ ray[tune]')
    # Download torch==1.5.0, 700+MB
    install_pip_package(f'ray[tune]')
    install_pip_package(f'torchaudio==0.5.0')
    install_pip_package(f'kapre==0.1.4')
    install_pip_package(f'requests')
    install_pip_package(f'{dn}/sol')

    sys.path.append(f'{dn}/sol')

    install_pip_package(f'{dn}/3rdparty/musicnn/')
    install_pip_package(f'{dn}/3rdparty/audio-mlgate/')

    return os.path.abspath(dn), os.path.abspath(os.path.join(dn, 'models'))
