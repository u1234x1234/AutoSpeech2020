import os
import time
from contextlib import contextmanager
from typing import List, Tuple

import numpy as np
import torch
from torch import cat, cuda
from torch import device as torch_device
from torch import enable_grad, from_numpy, nn, optim
from torch.utils.data import DataLoader, Dataset

ACTIVATIONS = {
    'tanh': nn.Tanh,
    'celu': nn.CELU,
    # 'gelu': nn.GELU,
    'relu': nn.ReLU,
    'selu': nn.SELU,
}

NORMALIZATIONS = {
    'batch': nn.BatchNorm1d,
    'layer': nn.LayerNorm,
}


def is_subclass(obj, classinfo):
    try:
        return issubclass(obj, classinfo)
    except Exception:
        pass
    return False


def init_activation(activation):
    if activation is None:
        return None
    if isinstance(activation, str) and activation in ACTIVATIONS:
        return ACTIVATIONS[activation]()

    if is_subclass(activation, nn.Module):
        return activation()

    raise ValueError('No such activation: "{}"'.format(activation))


def init_normalization(normalization):
    if normalization is None:
        return None
    if isinstance(normalization, str) and normalization in NORMALIZATIONS:
        return NORMALIZATIONS[normalization]
    if is_subclass(normalization, nn.Module):
        return normalization
    raise ValueError('No such normalization: "{}"'.format(normalization))


def device_of_model(model):
    return next(model.parameters()).device


# Following snippet is licensed under MIT license
# Christoph Heindl
# https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
@contextmanager
def eval_mode(net):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


@contextmanager
def test_mode(net):
    "torch.no_grad + model.eval()"
    with torch.no_grad(), eval_mode(net):
        yield


def to_torch_tensors(*args, dtype=torch.long, cuda=True):
    tensors = []
    for x in args:
        if isinstance(x, list):
            x = np.array(x)
        x = torch.from_numpy(x).to(dtype).contiguous()
        if cuda and not x.is_cuda:
            x = x.cuda()
        tensors.append(x)

    return tensors


def number_of_learnable_parameters(model):
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


@contextmanager
def clear_up():
    yield
    torch.cuda.empty_cache()


# TODO 2D, 3D, inputs
def infer_shape(model, shape=(1000,), device='cpu'):
    batch_size = 1
    # Does not work when the model parameters spanned across multiple devices
    try:
        model_device = next(model.parameters()).device
    except StopIteration:  # No parameters
        model_device = torch.device(device)
    x = torch.randn(size=(batch_size, *shape)).to(model_device)
    out = model(x)
    return tuple(out.shape[1:])


OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
}


def dataset_factory(preprocessor=None) -> Dataset:

    class TrainTestDataset(Dataset):
        def __init__(self, X, y=None, preprocessor=preprocessor):
            self.X = X
            self.y = torch.from_numpy(y) if y is not None else None
            self.preprocessor = preprocessor

        def __getitem__(self, index):
            x = self.X[index]
            if self.preprocessor:
                x = self.preprocessor(x)

            if self.y is not None:
                return x, self.y[index]
            else:
                return x

        def __len__(self):
            return len(self.X)

    return TrainTestDataset


class Restorable:
    def save(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
        torch.save(self.optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))

    def restore(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
        self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pt')))


class TransferClassifier(nn.Module):
    """
                    X
                    |
        feature_extractor(trainable or not)
                    |
            classification head

    Classifier on top of the passed feature extractor (encoder, embedder)
    """

    def __init__(
            self, feature_extractor: nn.Module, layers: List[int], freeze=0,
            activation=None, out_dropout=None, in_norm=None):
        """
        Args:
            feature_extractor (nn.Module): anything that returns fixed size vectors
            n_classes (int): number of
            freeze (int, optional): percentage of freezed layers. Defaults to 0.
            activation (nn.Module, optional): activation. Defaults to None.
            out_dropout (int, optional): dropout. Defaults to None.
            in_norm ([type], optional): [description]. Defaults to None.
        """
        super().__init__()

        assert len(layers) > 0
        # Freeze first `freeze`% of weights
        perc_to_freeze = int(len(list(feature_extractor.parameters())) * freeze)
        for idx, p in enumerate(feature_extractor.parameters()):
            if idx < perc_to_freeze:
                p.requires_grad = False

        emb_size = infer_shape(feature_extractor)[0]

        modules = [feature_extractor]
        in_norm = init_normalization(in_norm)
        if in_norm:
            modules.append(in_norm(emb_size))

        size = emb_size
        for hidden_size in layers[:-1]:
            modules.append(nn.Linear(size, hidden_size))
            size = hidden_size
            if activation:
                modules.append(ACTIVATIONS[activation]())

        if out_dropout:
            modules.append(nn.Dropout(out_dropout))

        modules.append(nn.Linear(size, layers[-1]))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class ModelRunner(Restorable):
    "Combines the model, training process and data"

    def __init__(self, model, trainset_cls,
                 testset_cls=None, test_apply=None,
                 batch_size=8, lr=0.01, clip_grad=0, weight_decay=0,
                 optimizer='sgd', criterion=nn.CrossEntropyLoss,
                 device='cuda', dataset_caching=True):

        assert isinstance(model, nn.Module)

        self.trainset_cls = trainset_cls
        self.testset_cls = testset_cls or trainset_cls
        self.test_apply = test_apply

        self.batch_size = batch_size
        self.clip_grad = clip_grad
        self.device = torch_device(device)

        self.model = model
        self.model.to(device)
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = OPTIMIZERS[optimizer](parameters, lr=lr, weight_decay=weight_decay)
        self.criterion = criterion()

        self._dataset_caching = dataset_caching
        self._dataset_cache = {}
        self._dataset_hash = {}

    def get_dataset(self, *data, key: str) -> Dataset:
        if self._dataset_caching:
            hs = id(data[0])
            if self._dataset_hash.get(key) != hs:
                self._dataset_hash[key] = hs
                ds_cls = self.trainset_cls if key == 'fit' else self.testset_cls
                self._dataset_cache[key] = ds_cls(*data)
            return self._dataset_cache[key]
        else:
            ds_cls = self.trainset_cls if key == 'fit' else self.testset_cls
            return ds_cls(*data)

    def fit(self, X, y, n_seconds=None, n_epochs=None):
        dataset = self.get_dataset(X, y, key='fit')

        training_loop(
            model=self.model,
            dataset=dataset,
            device=self.device,
            criterion=self.criterion,
            optimizer=self.optimizer,
            clip_grad=self.clip_grad,
            batch_size=self.batch_size,
            max_time=n_seconds, max_epoch=n_epochs)

    def predict(self, X):
        dataset = self.get_dataset(X, key='predict')
        predictions = predict_dataset(dataset, self.model, self.device, self.test_apply)
        return predictions


def predict_dataset(dataset, model, device, test_apply=None, batch_size=4):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=cuda.is_available()
    )
    y_pred = []

    with test_mode(model):
        for x in loader:
            x = x.to(device)

            if test_apply is not None:
                out = test_apply(x, model)
            else:
                out = model(x)

            y_pred.append(out.to(torch_device('cpu')).numpy())

    y_pred = np.vstack(y_pred)
    return y_pred


def training_loop(
        model, dataset, device, criterion, optimizer,
        clip_grad, batch_size, max_epoch=None, max_time=None):
    """Looping over the data_loader until a certain number of epochs or time is reached.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=cuda.is_available()
    )

    if max_epoch is None:
        max_epoch = np.inf
    if max_time is None:
        max_time = np.inf

    start_time = time.time()

    def time_is_over():
        return (time.time() - start_time) > max_time

    epoch_idx = -1
    model.train()
    with enable_grad():

        while epoch_idx < max_epoch and not time_is_over():
            epoch_idx += 1

            for data, target in loader:
                if isinstance(data, list):
                    data = [x.to(device) for x in data]
                else:
                    data = data.to(device)
                target = target.to(device)

                model.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                if clip_grad > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                optimizer.step()

                if time_is_over():
                    break


def load_state(path: str, model: nn.Module):
    loaded = torch.load(path)
    self_state = model.state_dict()
    for k, v in loaded.items():
        k = k[2:]  # TODO detect common prefix
        if k in self_state:
            self_state[k].copy_(v)
        else:
            print('skipped', k)
