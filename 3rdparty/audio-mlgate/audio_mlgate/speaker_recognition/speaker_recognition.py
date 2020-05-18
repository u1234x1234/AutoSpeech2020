"Pre-trained speaker recognition models."
from typing import List, Union

import numpy as np

from .pyannote_adapter import PyannoteSpeakerEmbeddingExtractor
from .voxceleb_trainer_adapter import ClovaAISpeakerEmbeddingExtractor

MODELS = {
    'clovaai_voxceleb_resnetse34l': ClovaAISpeakerEmbeddingExtractor,
    'pyannote_emb_voxceleb': PyannoteSpeakerEmbeddingExtractor,
}


class SpeakerEmbeddingExtractor:
    def __init__(self, model_name=None):
        if model_name not in MODELS.keys():
            raise ValueError(
                'Please, specify the "model_name" argument. Available options:\n'
                f'{list(MODELS.keys())}')

        self.model = MODELS[model_name]()

    def extract_from_files(self, X: Union[str, List[str]]):
        features = self.model.extract_from_files(X if isinstance(X, list) else [X])
        return features if isinstance(X, list) else features[0]

    def extract(self, X: Union[np.ndarray, List[np.ndarray]], sample_rate: int):
        features = self.model.extract(X if isinstance(X, list) else [X], sample_rate)
        return features if isinstance(X, list) else features[0]
