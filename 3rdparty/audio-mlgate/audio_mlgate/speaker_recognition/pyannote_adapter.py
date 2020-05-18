"Adapter to https://github.com/pyannote/pyannote-audio"
import os
import sys
import tempfile
from contextlib import contextmanager

import numpy as np
from scipy.io.wavfile import write

from ..utils import lazy_module_import


@contextmanager
def patch_stdout():
    back = sys.stdout
    void = open(os.devnull, 'w')
    sys.stdout = void
    yield
    void.close()
    sys.stdout = back


class PyannoteSpeakerEmbeddingExtractor:
    def __init__(self):
        lazy_module_import('torch')

        # pyannote messes with stdout, so temporary disable
        with patch_stdout():
            self.model = torch.hub.load('pyannote/pyannote-audio', 'emb_voxceleb', verbose=False)
        self.model.model_.eval()
        torch.set_grad_enabled(False)

    def extract(self, X, sample_rate):
        features = []
        for x in X:
            assert isinstance(x, np.ndarray)
            # TODO pyannote does not allow to pass the data, create a temporary file
            with tempfile.NamedTemporaryFile(mode='wb', buffering=0) as out_file:
                write(out_file, sample_rate, x)
                features.append(self._extract(out_file.name))

        return np.array(features)

    def extract_from_files(self, paths):
        features = []
        for path in paths:
            features.append(self._extract(path))
        return features

    def _extract(self, path):
        embedding = self.model({'audio': path})
        results = []
        # Window per second
        for _, emb in embedding:
            results.append(emb)

        return np.array(results)
