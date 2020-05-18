import numpy as np
from .networks import PretrainedSpeakerEmbedding
from .torch_ext import ModelRunner, dataset_factory, TransferClassifier
from functools import partial

SAMPLE_RATE = 16000


def _load_audio(audio, sample_rate=SAMPLE_RATE, max_len=3, evalmode=False, num_eval=10):
    """Modified version from
    https://github.com/clovaai/voxceleb_trainer/blob/master/DatasetLoader.py#L18
    """
    max_len = max_len * SAMPLE_RATE
    audiosize = audio.shape[0]

    if audiosize <= max_len:
        n_rep = np.math.ceil(max_len / audiosize)
        audio = np.tile(audio, n_rep)
        audiosize = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0, audiosize-max_len, num=num_eval).astype(np.int)
        seqs = [audio[x:x+max_len] for x in startframe]
    else:
        startframe = np.random.randint(0, audiosize-max_len+1)
        seqs = [audio[startframe:startframe+max_len]]

    seqs = np.array(seqs).squeeze()
    return seqs


def test_apply(x, model):
    if len(x.shape) > 2:
        bs, aug = x.shape[:2]  # Avg TTA (batch_size, aug_dim, features_dims)
        x = x.view((bs*aug, -1))
        out = model(x).view((bs, aug, -1)).mean(axis=1)
    else:
        out = model(x)
    return out


search_space = {
    'batch_size': [8, 16],
    'optimizer': [('sgd', 0.1), ('adam', 0.001)],
    'num_eval': [3],
    'max_len': [3, 7],
    'clip_grad': [1],
    'activation': ['tanh', None],
    'in_norm': [None, 'batch'],
    'freeze': [0, 0.5, 0.8],
    'out_dropout': [0, 0.5],
    'layers': [[], [128], [128, 64]]
}


def create_factory_method(n_classes, weights_path):

    def model_factory_method(**config):
        feature_extractor = PretrainedSpeakerEmbedding(weights_path)

        trainset_factory = dataset_factory(partial(_load_audio, max_len=config['max_len'], evalmode=False))
        testset_factory = dataset_factory(partial(_load_audio, max_len=config['max_len'], evalmode=True, num_eval=config['num_eval']))

        model = TransferClassifier(feature_extractor, [*config['layers'], n_classes], freeze=config['freeze'], activation=config['activation'])
        optimizer, lr = config['optimizer']
        runner = ModelRunner(
            model, trainset_factory, testset_factory,
            test_apply=test_apply, batch_size=config['batch_size'], lr=lr,
            optimizer=optimizer, clip_grad=config['clip_grad'])

        return runner

    return model_factory_method
