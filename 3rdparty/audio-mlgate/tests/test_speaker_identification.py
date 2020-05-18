from audio_mlgate.speaker_identification import FeatureExtractor


def test_speaker_features():
    fe = FeatureExtractor()
    path = '/home/u1234x1234/voxceleb/voxceleb1/id10270/5r0dWxy17C8/00001.wav'
    features = fe.extract_features_from_file(path)
