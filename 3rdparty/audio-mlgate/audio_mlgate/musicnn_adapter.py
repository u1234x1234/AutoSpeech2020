import os

import librosa
import numpy as np
import tensorflow as tf
from musicnn import configuration as config
from musicnn import models
from collections import defaultdict


def batch_data(audio, sr, n_frames, overlap):
    # compute the log-mel spectrogram with librosa
    if sr != config.SR:
        audio = librosa.core.resample(audio, sr, config.SR)

    if audio.shape[0] < sr*10:
        audio = np.repeat(audio, np.math.ceil(sr*10 / audio.shape[0]))

    audio_rep = librosa.feature.melspectrogram(y=audio,
                                               sr=sr,
                                               hop_length=config.FFT_HOP,
                                               n_fft=config.FFT_SIZE,
                                               n_mels=config.N_MELS).T
    audio_rep = audio_rep.astype(np.float16)
    audio_rep = np.log10(10000 * audio_rep + 1)

    # batch it for an efficient computing
    first = True
    last_frame = audio_rep.shape[0] - n_frames + 1
    # +1 is to include the last frame that range would not include

    for time_stamp in range(0, last_frame, overlap):
        patch = np.expand_dims(audio_rep[time_stamp: time_stamp + n_frames, :], axis=0)
        if first:
            batch = patch
            first = False
        else:
            batch = np.concatenate((batch, patch), axis=0)

    return batch, audio_rep


class MusicnnAdatper:
    def __init__(self, model='MTT_musicnn', input_length=10, input_overlap=False, extract_features=True):
        tf.compat.v1.disable_eager_execution()

        self.extract_features = extract_features
        self.model = model

        # select model
        if 'MTT' in model:
            labels = config.MTT_LABELS
        elif 'MSD' in model:
            labels = config.MSD_LABELS
        num_classes = len(labels)
        if 'vgg' in model and input_length != 3:
            raise ValueError('Set input_length=3, the VGG models cannot handle different input lengths.')

        # convert seconds to frames
        self.n_frames = librosa.time_to_frames(input_length, sr=config.SR,
                                               n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP) + 1
        if not input_overlap:
            self.overlap = self.n_frames
        else:
            self.overlap = librosa.time_to_frames(input_overlap, sr=config.SR,
                                                  n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP)

        tf.compat.v1.reset_default_graph()
        with tf.name_scope('model'):
            self.x = tf.compat.v1.placeholder(tf.float32, [None, self.n_frames, config.N_MELS])
            self.is_training = tf.compat.v1.placeholder(tf.bool)
            if 'vgg' in model:
                self.y, self.pool1, self.pool2, self.pool3, self.pool4, self.pool5 = models.define_model(
                    self.x, self.is_training, model, num_classes)
            else:
                self.y, _, _, _, _, _, self.mean_pool, self.max_pool, self.penultimate = models.define_model(
                    self.x, self.is_training, model, num_classes)
            self.normalized_y = tf.nn.sigmoid(self.y)

        cfg = tf.compat.v1.ConfigProto()
        cfg.gpu_options.per_process_gpu_memory_fraction = 0.5
        cfg.gpu_options.allow_growth = True
        cfg.log_device_placement = False
        self.sess = tf.compat.v1.Session(config=cfg)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()

        try:
            saver.restore(self.sess, os.path.dirname(models.__file__)+'/'+model+'/')
        except Exception:
            raise ValueError('MSD_musicnn_big model is only available if you install from source: python setup.py install')

    def extract(self, audio, sample_rate):
        batch, spectrogram = batch_data(audio, sample_rate, self.n_frames, self.overlap)

        if self.extract_features:
            if 'vgg' in self.model:
                extract_vector = [self.normalized_y, self.pool1, self.pool2, self.pool3, self.pool4, self.pool5]
            else:
                extract_vector = [self.normalized_y, self.mean_pool, self.max_pool, self.penultimate]
        else:
            extract_vector = [self.normalized_y]

        tf_out = self.sess.run(extract_vector,
                               feed_dict={self.x: batch[:config.BATCH_SIZE],
                                          self.is_training: False})

        if self.extract_features:
            if 'vgg' in self.model:
                predicted_tags, pool1_, pool2_, pool3_, pool4_, pool5_ = tf_out
                features = defaultdict(list)
                features['pool1'].append(np.squeeze(pool1_))
                features['pool2'].append(np.squeeze(pool2_))
                features['pool3'].append(np.squeeze(pool3_))
                features['pool4'].append(np.squeeze(pool4_))
                features['pool5'].append(np.squeeze(pool5_))
            else:
                predicted_tags, mean_pool_, max_pool_, penultimate_ = tf_out
                features = defaultdict(list)
                features['mean_pool'].append(mean_pool_)
                features['max_pool'].append(max_pool_)
                features['penultimate'].append(penultimate_)
        else:
            predicted_tags = tf_out[0]

        taggram = np.array(predicted_tags)

        for id_pointer in range(config.BATCH_SIZE, batch.shape[0], config.BATCH_SIZE):

            tf_out = self.sess.run(extract_vector,
                                   feed_dict={self.x: batch[id_pointer:id_pointer+config.BATCH_SIZE],
                                              self.is_training: False})

            if self.extract_features:
                if 'vgg' in self.model:
                    predicted_tags, pool1_, pool2_, pool3_, pool4_, pool5_ = tf_out
                    features['pool1'].append(np.squeeze(pool1_))
                    features['pool2'].append(np.squeeze(pool2_))
                    features['pool3'].append(np.squeeze(pool3_))
                    features['pool4'].append(np.squeeze(pool4_))
                    features['pool5'].append(np.squeeze(pool5_))
                else:
                    predicted_tags, mean_pool_, max_pool_, backend_ = tf_out
                    features['mean_pool'].append(mean_pool_)
                    features['max_pool'].append(max_pool_)
                    features['penultimate'].append(penultimate_)
            else:
                predicted_tags = tf_out[0]

            taggram = np.concatenate((taggram, np.array(predicted_tags)), axis=0)

        if self.extract_features:
            r_features = {}
            for name, values in features.items():
                r_features[name] = np.array(values)
            return taggram, r_features
        else:
            return taggram

    # def __del__(self):
    #     if hasattr(self, 'sess'):
    #         self.sess.close()
