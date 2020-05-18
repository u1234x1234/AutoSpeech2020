import numpy as np

from .musicnn_adapter import MusicnnAdatper


class MusicGenre:
    def __init__(self, model='MSD_musicnn', layer='mean_pool'):
        self._layer = layer
        self.model = MusicnnAdatper(model, input_overlap=False)

    def extract_features(self, X, sample_rate):
        features = []

        for audio_data in X:
            x_t, x_f = self.model.extract(audio_data, sample_rate)
            x_f = x_f[self._layer]
            features.append(x_f.reshape((-1, x_f.shape[-1])))

        return np.array(features)
