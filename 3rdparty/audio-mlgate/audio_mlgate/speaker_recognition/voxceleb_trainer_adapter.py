from contextlib import redirect_stdout

import librosa
import numpy as np
import soundfile

from ..utils import get_model_path, load_external, lazy_module_import

SAMPLE_RATE = 16000
MODEL_NAME = 'baseline_lite_ap.model'


def _load_audio(audio, sample_rate, max_frames, evalmode=True, num_eval=10):
    """Modified version from
    https://github.com/clovaai/voxceleb_trainer/blob/master/DatasetLoader.py#L18
    """
    if sample_rate != SAMPLE_RATE:
        audio = librosa.core.resample(audio, sample_rate, SAMPLE_RATE)

    max_len = max_frames * 160 + 240
    audiosize = audio.shape[0]

    if audiosize <= max_len:
        n_rep = np.math.ceil(max_len / audiosize)
        audio = np.tile(audio, n_rep)
        audiosize = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0, audiosize-max_len, num=num_eval)

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_len])

    feat = np.stack(feats, axis=0)

    return feat


class ClovaAISpeakerEmbeddingExtractor:
    def __init__(self):
        lazy_module_import('torch')

        with load_external('voxceleb_trainer'), redirect_stdout(None):
            from SpeakerNet import SpeakerNet

            self.net = SpeakerNet(
                max_frames=300,
                model='ResNetSE34L',
                nOut=512,
                trainfunc='angleproto',
                nSpeakers=6200,
            )

            self.net.loadParameters(get_model_path(MODEL_NAME))
            self.net.eval()
            torch.set_grad_enabled(False)

    def extract_from_files(self, path):
        audio, sample_rate = soundfile.read(path)
        audio_data = _load_audio(audio, sample_rate, 300, evalmode=True, num_eval=10)
        return self.transform(audio_data, sample_rate)

    def extract(self, X, sample_rate: int):
        features = []
        for x in X:
            audio_data = _load_audio(x, sample_rate, 300, evalmode=True, num_eval=10)

            if len(audio_data.shape) == 1:
                audio_data = audio_data[np.newaxis, :]

            audio_data = torch.FloatTensor(audio_data)
            audio_data = audio_data.cuda()
            x_f = self.net.__S__.forward(audio_data).detach().cpu().numpy()
            features.append(x_f)

        return np.array(features)
