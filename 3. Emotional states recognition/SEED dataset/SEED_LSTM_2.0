"""
- Python 3.7
- tenforflow 2.0.0
"""
#
import os
import sys
from scipy import io
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.model_selection import cross_val_score
from tensorflow.python.ops import control_flow_ops

from tensorflow.keras.layers import LSTM, GRU, Dense, Activation, LeakyReLU, Dropout
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix

class EmotionRecognition():
    def __init__(self, **kwargs):
        """

        :param kwargs:
        ==========================================================
            Model params

        ==========================================================
            Training params

        """

        self.train_type = kwargs.get("train_type", 'DEPN')
        self.num_sub = 1
        self.faeture_type = kwargs.get("feature_type", "Entropy")

        self.batch_size = kwargs.get("batch_size", 45)
        self.epochs = kwargs.get("epochs", 100)
        self.k = kwargs.get("k", 10)
        self.n_classes = kwargs.get("n_classes", 3)

        self.n_rnn_layers = kwargs.get("n_rnn_layers", 2)
        self.n_dense_layers = kwargs.get("n_dense_layers", 2)
        self.rnn_units = kwargs.get("rnn_units", 128)
        self.dense_units = kwargs.get("dense_units", 128)
        self.cell = kwargs.get("cell", LSTM)

        # list of dropouts of each layer
        # must be len(dropouts) = n_rnn_layers + n_dense_layers
        self.dropout = kwargs.get("dropout", 0.3)
        self.dropout = self.dropout if isinstance(self.dropout, list) else [self.dropout] * (
                    self.n_rnn_layers + self.n_dense_layers)

        self.optimizer = kwargs.get("optimizer", "adam")
        self.loss = kwargs.get("loss", "categorical_crossentropy")

        self.data_loaded = False
        self.model_created = False
        self.model_trained = False

        self.model_path = 'models'
        if not os.path.isdir("models"):
            os.mkdir("models")

        if self.train_type == 'DEPN':
            self.model_name = f"{self.cell.__name__}-layers-{self.n_rnn_layers}-{self.n_dense_layers}-units-{self.rnn_units}-{self.dense_units}-dropout-{self.dropout}-{self.train_type}-sub-{self.num_sub}.h5"
        else:
            self.model_name = f"{self.cell.__name__}-layers-{self.n_rnn_layers}-{self.n_dense_layers}-units-{self.rnn_units}-{self.dense_units}-dropout-{self.dropout}-{self.train_type}.h5"

    def model_exists(self):
        """
        Check if model already exists
        """
        filename = f'model/{self.model_name}'
        return filename if os.path.isfile(filename) else None

    def load_data(self):
        """
        temp_data = [samples x time_window x values]
        """
        if not self.data_loaded:
            input_data = []
            input_label = []

            for i in range(1, 16):
                data_path = f'./dataset/{self.faeture_type}/sub{str(i)}_data.mat'
                label_path = f'./dataset/{self.faeture_type}/sub{str(i)}_label.mat'
                temp_data = io.loadmat(data_path)['data'].astype(np.float32)
                temp_label = io.loadmat(label_path)['label'].astype(np.float32)
                input_data.append(temp_data)
                input_label.append(temp_label)

            self.X = np.array(input_data)
            self.y = np.array(input_label)
            self.input_length = self.X.shape[-1]
            self.time_win = self.X.shape[2]
            self.data_loaded = True

    def create_model(self):
        if self.model_created:
            # model already created, why call twice
            return

        if not self.data_loaded:
            # if data isn't loaded yet, load it
            self.load_data()

        model = Sequential()

        # rnn layers
        for i in range(self.n_rnn_layers):
            if i == 0:
                # first layer
                model.add(self.cell(self.rnn_units, return_sequences=True, input_shape=(None, self.input_length)))
                model.add(Dropout(self.dropout[i]))
            elif i == self.n_rnn_layers-1:
                # last layer
                model.add(self.cell(self.rnn_units, return_sequences=False))
                model.add(Dropout(self.dropout[i]))
            else:
                # middle layers
                model.add(self.cell(self.rnn_units, return_sequences=True))
                model.add(Dropout(self.dropout[i]))

        # dense layers
        for j in range(self.n_dense_layers):
            # if n_rnn_layers = 0, only dense
            if self.n_rnn_layers == 0 and j == 0:
                model.add(Dense(self.dense_units, input_shape=(None, self.input_length)))
                model.add(Dropout(self.dropout[i + j]))
            else:
                model.add(Dense(self.dense_units))
                model.add(Dropout(self.dropout[i + j]))

        model.add(Dense(self.n_classes, activation="softmax"))
        model.compile(loss=self.loss, metrics=["accuracy"], optimizer=self.optimizer)

        model.summary()

        self.model = model
        model.save_weights(f'{self.model_path}/{self.model_name}')
        self.model_created = True

    def train(self):
        pass

    def evaluate_perf(self):
        """
        Evaluate performance according to the type of train
        :return: 
        """
        self.create_model()
        accuracy = []

        if self.train_type == 'DEPN':
            cv = KFold(self.k, shuffle=True, random_state=None)
            X_sub = self.X[self.num_sub]
            y_sub = self.y[self.num_sub]

            for i, (idx_train, idx_test) in enumerate(cv.split(y_sub)):
                print(i + 1, "fold")
                X_train = X_sub[idx_train]
                y_train = y_sub[idx_train]
                X_test = X_sub[idx_test]
                y_test = y_sub[idx_test]

                self.model.load_weights(f'{self.model_path}/{self.model_name}')
                self.model.fit(X_train, y_train,
                               batch_size=self.batch_size,
                               epochs=self.epochs,
                               validation_data=(X_test, y_test))

                y_pred = self.model.predict_classes(X_test)[0]
                y_test = [np.argmax(y, out=None, axis=None) for y in y_test]
                acc = accuracy_score(y_true=y_test, y_pred=y_pred)
                print(i+1, "fold, test score:", acc)
                accuracy.append(acc)
        else:
            loo = LeaveOneOut()
            for i, (idx_train, idx_test) in enumerate(loo.split(self.y)):
                print("Leave subject ", i+1)
                n_train = len(idx_train) * self.X.shape[1]
                n_test = self.X.shape[1]

                X_train = self.X[idx_train].reshape((n_train, self.time_win, self.input_length))
                y_train = self.y[idx_train].reshape((n_train, self.n_classes))
                X_test = self.X[idx_test].reshape((n_test, self.time_win, self.input_length))
                y_test = self.y[idx_test].reshape((n_test, self.n_classes))

                self.model.load_weights(f'{self.model_path}/{self.model_name}')
                self.model.fit(X_train, y_train,
                               batch_size=self.batch_size,
                               epochs=self.epochs,
                               validation_data=(X_test, y_test))

                y_pred = self.model.predict_classes(X_test)
                y_test = [np.argmax(y, out=None, axis=None) for y in y_test]
                acc = accuracy_score(y_true=y_test, y_pred=y_pred)
                print(i + 1, "subject leave, test score:", accuracy_score(y_true=y_test, y_pred=y_pred))
                accuracy.append(acc)
        print(f'Total accuracy: {accuracy} \n Average accuracy: {np.mean(accuracy)}')
        return accuracy


if __name__ == "__main__":
    ER = EmotionRecognition(rnn_units=256, epochs=50, train_type='INDEPN')
    ER.evaluate_perf()
