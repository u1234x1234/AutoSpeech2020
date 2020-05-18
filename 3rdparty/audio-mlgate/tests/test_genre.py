from audio_mlgate.genre_classification import GenreFeatureExtractor


def test_genre():
    fe = GenreFeatureExtractor()
    path = '/home/u1234x1234/voxceleb/voxceleb1/id10270/5r0dWxy17C8/00001.wav'
    features = fe.extract_features_from_file(path)


test_genre()
