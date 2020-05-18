import logging
import os
import time

import keras
import numpy as np
import tensorflow as tf

from kapre.time_frequency import Melspectrogram
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import safe_indexing
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()

logger.addHandler(handler)
logger.setLevel(logging.INFO)

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True
config.log_device_placement = False

sess = tf.compat.v1.Session(config=config)

set_session(sess)

# parameters
SAMPLING_RATE = 16_000
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 1_024  # 0.064 sec
FMIN = 20
FMAX = SAMPLING_RATE // 2


def make_extractor(input_shape, sr=SAMPLING_RATE):
    model = keras.models.Sequential()

    model.add(
        Melspectrogram(
            fmax=FMAX,
            fmin=FMIN,
            n_dft=N_FFT,
            n_hop=HOP_LENGTH,
            n_mels=N_MELS,
            name='melgram',
            image_data_format='channels_last',
            input_shape=input_shape,
            return_decibel_melgram=True,
            power_melgram=2.0,
            sr=sr,
            trainable_kernel=False
        )
    )

    return model


def get_fixed_array(X_list, len_sample=5, sr=SAMPLING_RATE):
    for i in range(len(X_list)):
        if len(X_list[i]) < len_sample * sr:
            n_repeat = np.ceil(
                sr * len_sample / X_list[i].shape[0]
            ).astype(np.int32)
            X_list[i] = np.tile(X_list[i], n_repeat)

        X_list[i] = X_list[i][:len_sample * sr]

    X = np.asarray(X_list)
    X = X[:, :, np.newaxis]
    X = X.transpose(0, 2, 1)

    return X


def extract_features(X_list, model, len_sample=5, sr=SAMPLING_RATE):
    X = get_fixed_array(X_list, len_sample=len_sample, sr=sr)
    X = model.predict(X)
    X = X.transpose(0, 2, 1, 3)

    return X


def crop_image(image):
    h, w, _ = image.shape
    h0 = np.random.randint(0, h - w)
    image = image[h0:h0 + w]

    return image


def load_pretrained_model(input_shape, n_classes, max_layer_num=5):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(base_dir, 'ckpt/ckpt01/data01.ckpt')
    model = make_model(input_shape, 100, max_layer_num=max_layer_num)

    model.load_weights(filename)
    model.pop()
    model.pop()
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    return model


def make_model(input_shape, n_classes, max_layer_num=5):
    model = Sequential()
    min_size = min(input_shape[:2])
    optimizer = tf.keras.optimizers.SGD(decay=1e-06, momentum=0.9)

    for i in range(max_layer_num):
        if i == 0:
            model.add(Conv2D(64, 3, input_shape=input_shape, padding='same'))
        else:
            model.add(Conv2D(64, 3, padding='same'))

        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        min_size //= 2

        if min_size < 2:
            break

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(rate=0.5))
    model.add(Activation('relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    model.compile(optimizer, 'categorical_crossentropy')

    return model


def frequency_masking(image, p=0.5, F=0.2):
    _, w, _ = image.shape
    p_1 = np.random.rand()

    if p_1 > p:
        return image

    f = np.random.randint(0, int(w * F))
    f0 = np.random.randint(0, w - f)

    image[:, f0:f0 + f, :] = 0.0

    return image


class TTAGenerator(object):
    def __init__(self, X, batch_size):
        self.X = X
        self.batch_size = batch_size

        self.n_samples, _, _, _ = X.shape

    def __call__(self):
        while True:
            for start in range(0, self.n_samples, self.batch_size):
                end = min(start + self.batch_size, self.n_samples)
                X_batch = self.X[start:end]

                yield self.__data_generation(X_batch)

    def __data_generation(self, X_batch):
        n, _, w, _ = X_batch.shape
        X = np.zeros((n, w, w, 1))

        for i in range(n):
            X[i] = crop_image(X_batch[i])

        return X, None


class MixupGenerator(object):
    def __init__(
        self,
        X,
        y,
        alpha=0.2,
        batch_size=32,
        datagen=None,
        shuffle=True
    ):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.batch_size = batch_size
        self.datagen = datagen
        self.shuffle = shuffle

    def __call__(self):
        while True:
            indices = self.__get_exploration_order()
            n_samples, _, _, _ = self.X.shape
            itr_num = int(n_samples // (2 * self.batch_size))

            for i in range(itr_num):
                indices_head = indices[
                    2 * i * self.batch_size:(2 * i + 1) * self.batch_size
                ]
                indices_tail = indices[
                    (2 * i + 1) * self.batch_size:(2 * i + 2) * self.batch_size
                ]

                yield self.__data_generation(indices_head, indices_tail)

    def __get_exploration_order(self):
        n_samples = len(self.X)
        indices = np.arange(n_samples)

        if self.shuffle:
            np.random.shuffle(indices)

        return indices

    def __data_generation(self, indices_head, indices_tail):
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1_tmp = safe_indexing(self.X, indices_head)
        X2_tmp = safe_indexing(self.X, indices_tail)
        n, _, w, _ = X1_tmp.shape
        X1 = np.zeros((n, w, w, 1))
        X2 = np.zeros((n, w, w, 1))

        for i in range(self.batch_size):
            X1[i] = crop_image(X1_tmp[i])
            X2[i] = crop_image(X2_tmp[i])

        X = X1 * X_l + X2 * (1.0 - X_l)

        y1 = safe_indexing(self.y, indices_head)
        y2 = safe_indexing(self.y, indices_tail)
        y = y1 * y_l + y2 * (1.0 - y_l)

        if self.datagen is not None:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        return X, y


class Model(object):
    def __init__(
        self,
        metadata,
        cv,
        metric,
        batch_size=32,
        crop_sec=5,
        n_predictions=10,
        patience=100,
        random_state=0,
        sr=SAMPLING_RATE
    ):
        self.metadata = metadata
        self.batch_size = batch_size
        self.crop_sec = crop_sec
        self.n_predictions = n_predictions
        self.patience = patience
        self.random_state = random_state
        self.sr = sr

        self.done_training = False
        self.max_score = 0
        self.n_iter = 0
        self.cv = cv[0]
        self.metric = metric

    def train(self, train_dataset, remaining_time_budget=None):
        start_time = time.perf_counter()

        if not hasattr(self, 'X_train'):
            self.extractor = make_extractor((1, self.crop_sec * self.sr))

            X, y = train_dataset
            X = extract_features(
                X,
                self.extractor,
                len_sample=self.crop_sec,
                sr=self.sr
            )
            X = (
                X - np.mean(
                    X,
                    axis=(1, 2, 3),
                    keepdims=True
                )
            ) / np.std(X, axis=(1, 2, 3), keepdims=True)

            # self.X_train, self.X_valid, \
            #     self.y_train, self.y_valid = train_test_split(
            #         X,
            #         y,
            #         random_state=self.random_state,
            #         shuffle=True,
            #         stratify=y,
            #         train_size=0.9
            #     )
            train_idx, val_idx = self.cv
            self.X_train, self.X_valid = X[train_idx], X[val_idx]
            self.y_train, self.y_valid = y[train_idx], y[val_idx]

            self.train_size, _, w, _ = self.X_train.shape
            self.valid_size, _, _, _ = self.X_valid.shape

            self.model = load_pretrained_model(
                (w, w, 1),
                self.metadata['class_num']
            )

        while True:
            elapsed_time = time.perf_counter() - start_time
            remaining_time = remaining_time_budget - elapsed_time

            if remaining_time <= 0.125 * self.metadata['time_budget']:
                self.done_training = True

                break

            datagen = ImageDataGenerator(
                preprocessing_function=frequency_masking
            )
            training_generator = MixupGenerator(
                self.X_train,
                self.y_train,
                batch_size=self.batch_size,
                datagen=datagen
            )()
            valid_generator = TTAGenerator(
                self.X_valid,
                batch_size=self.batch_size
            )()

            self.model.fit_generator(
                training_generator,
                steps_per_epoch=self.train_size // self.batch_size,
                epochs=self.n_iter + 5,
                initial_epoch=self.n_iter,
                shuffle=True,
                verbose=1
            )

            probas = self.model.predict_generator(
                valid_generator,
                steps=np.ceil(self.valid_size / self.batch_size)
            )
            # valid_score = roc_auc_score(self.y_valid, probas, average='macro')
            valid_score = self.metric(self.y_valid, probas)

            self.n_iter += 5

            logger.info(
                f'valid_auc={valid_score:.3f}, '
                f'max_valid_auc={self.max_score:.3f}'
            )

            if self.max_score < valid_score:
                self.max_score = valid_score

                break

    def test(self, X, remaining_time_budget=None):
        if not hasattr(self, 'X_test'):
            self.X_test = extract_features(
                X,
                len_sample=self.crop_sec,
                model=self.extractor
            )
            self.X_test = (
                self.X_test - np.mean(
                    self.X_test,
                    axis=(1, 2, 3),
                    keepdims=True
                )
            ) / np.std(self.X_test, axis=(1, 2, 3), keepdims=True)
            self.test_size, _, _, _ = self.X_test.shape

        probas = np.zeros(
            (self.metadata['test_num'], self.metadata['class_num'])
        )

        for _ in range(self.n_predictions):
            test_generator = TTAGenerator(
                self.X_test,
                batch_size=self.batch_size
            )()

            probas += self.model.predict_generator(
                test_generator,
                steps=np.ceil(self.test_size / self.batch_size)
            )

        probas /= self.n_predictions

        return probas
