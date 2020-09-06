"""
- Python 3.7
- tenforflow 2.0.0
"""

import os
from scipy import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, LeaveOneOut

from tensorflow.keras.layers import LSTM, GRU, Dense, Activation, LeakyReLU, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization

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

        self.train_type = kwargs.get("train_type", 'LOO')
        self.faeture_type = kwargs.get("feature_type", "Entropy")
        self.freq_band = kwargs.get("freq_band", "beta")

        self.batch_size = kwargs.get("batch_size", 40)
        self.epochs = kwargs.get("epochs", 100)
        self.k = kwargs.get("k", 10)
        self.n_classes = kwargs.get("n_classes", 2)

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
        self.loss = kwargs.get("loss", "binary_crossentropy")

        self.data_loaded = False
        self.model_created = False
        self.model_trained = False

        self.model_path = 'models'
        if not os.path.isdir("models"):
            os.mkdir("models")

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

            for i in range(1, 33):
                s = 's0'+str(i) if i<10 else 's'+str(i)
                data_path = f'./dataset/{self.faeture_type}/{self.freq_band}/{s}_data.mat'
                label_path = f'./dataset/{self.faeture_type}/{self.freq_band}/{s}_label.mat'

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
                model.add(self.cell(self.rnn_units, return_sequences=True, input_shape=(None, self.input_length), kernel_regularizer=l2(0.001)))
            elif i == self.n_rnn_layers-1:
                # last layer
                model.add(self.cell(self.rnn_units, return_sequences=False, kernel_regularizer=l2(0.001)))
            else:
                # middle layers
                model.add(self.cell(self.rnn_units, return_sequences=True, kernel_regularizer=l2(0.001)))

            model.add(BatchNormalization())
            model.add(Dropout(self.dropout[i]))

        # dense layers
        for j in range(self.n_dense_layers):
            # if n_rnn_layers = 0, only dense
            if self.n_rnn_layers == 0 and j == 0:
                model.add(Dense(self.dense_units, input_shape=(None, self.input_length)))
            else:
                model.add(Dense(self.dense_units))

            model.add(BatchNormalization())
            model.add(Dropout(self.dropout[i]))

        model.add(Dense(self.n_classes, activation="softmax"))
        # model.compile(loss=self.loss, metrics=["accuracy"], optimizer=self.optimizer)
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(loss=self.loss, metrics=["accuracy"], optimizer=opt)

        model.summary()

        self.model = model
        # model.build()
        model.save_weights(f'{self.model_path}/{self.model_name}')
        self.model_created = True

    def evaluate_perf(self):
        """
        Evaluate performance according to the type of train
        :return: classification accuracy
        """
        self.create_model()
        accuracy = []

        if self.train_type == 'KFOLD':
            cv = KFold(self.k, shuffle=True, random_state=None)

            X_ = np.reshape(self.X, (32 * 40, self.time_win, self.input_length))
            y_ = np.reshape(self.y, (32*40, -1))
            s = np.arange(X_.shape[0])
            np.random.shuffle(s)
            X_ = X_[s]
            y_ = y_[s]

            for i, (idx_train, idx_test) in enumerate(cv.split(y_)):
                print(i + 1, "fold")
                X_train = X_[idx_train]
                y_train = y_[idx_train]
                X_test = X_[idx_test]
                y_test = y_[idx_test]

                callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

                self.model.load_weights(f'{self.model_path}/{self.model_name}')
                history = self.model.fit(X_train, y_train,
                                         batch_size=self.batch_size,
                                         epochs=self.epochs,
                                         validation_data=(X_test, y_test),
                                         callbacks=[callback])

                # plt.plot(history.history['loss'])
                # plt.plot(history.history['val_loss'])
                # plt.show()

                y_pred = self.model.predict_classes(X_test)
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

                s = np.arange(X_train.shape[0])
                np.random.shuffle(s)
                X_train = X_train[s]
                y_train = y_train[s]

                callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

                self.model.load_weights(f'{self.model_path}/{self.model_name}')
                history = self.model.fit(X_train, y_train,
                                         batch_size=self.batch_size,
                                         epochs=self.epochs,
                                         validation_data=(X_test, y_test),
                                         callbacks=[callback])

                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.show()

                y_pred = self.model.predict_classes(X_test)
                y_test = [np.argmax(y, out=None, axis=None) for y in y_test]
                acc = accuracy_score(y_true=y_test, y_pred=y_pred)
                print(i + 1, "subject leave, test score:", accuracy_score(y_true=y_test, y_pred=y_pred))
                accuracy.append(acc)
        print(f'Total accuracy: {accuracy} \n Average accuracy: {np.mean(accuracy): .2f}')
        return accuracy


if __name__ == "__main__":
    ER = EmotionRecognition(dropout=0.3, n_dense_layers=1, rnn_units=128, epochs=50, train_type='KFOLD', feature_type='Entropy')
    accuracy = ER.evaluate_perf()
    
    if(ER.train_type == 'KFOLD'):
        acc = pd.DataFrame(accuracy, index=[f'{i}fold' for i in range(1, ER.k+1)], columns=['accuracy'])
    else:
        acc = pd.DataFrame(accuracy, index=[f'{i}subject_test' for i in range(1, ER.X.shape[0] + 1)], columns=['accuracy'])
        
    acc.to_csv(f'accuracy-{ER.model_name}.csv')
