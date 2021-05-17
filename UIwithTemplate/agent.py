import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import os, re, glob
import cv2
import numpy as np
from model import CNNAutoEncoder_28, CNNAutoEncoder_96, MakeAutoencoderModel
from PIL import Image
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

class Autoencoder_Agent(object):

    def __init__(self, model_size, optimizer,learning_rate, activation_function="sigmoid", loss_function="mse", dimension=32):
        self.model_size = model_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.dimension = dimension
        if self.model_size == 28:
            self.model = CNNAutoEncoder_28(optimizer = optimizer, learning_rate = learning_rate, activation=activation_function, loss=loss_function)
            self.compressed_layer = 5
        elif self.model_size == 96: 
            self.model = CNNAutoEncoder_96(optimizer = optimizer,dimension=dimension,
                                           learning_rate = learning_rate, activation=activation_function, loss=loss_function)
            self.compressed_layer = 8
        else:
            self.model = MakeAutoencoderModel(image_size = model_size, dimension=dimension,optimizer = optimizer, learning_rate = learning_rate,
                                              activation=activation_function, loss=loss_function)
            if model_size == 1024:
                self.compressed_layer = 17
            elif model_size == 256:
                self.compressed_layer = 11

    def train(self, X_train, batch_size, epochs, validation_data, early_stopping=False):
        if early_stopping:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100,  min_delta=0.0001)
            mc = ModelCheckpoint(f'insectWing_dimension_{self.dimension}.h5', monitor='val_loss', verbose=1, save_best_only=True)
            hist = self.model.fit(X_train,X_train,batch_size = batch_size,epochs=epochs,validation_data=(validation_data,validation_data), callbacks=[es, mc])
        else:
            hist = self.model.fit(X_train,X_train,batch_size = batch_size,epochs=epochs,validation_data=(validation_data,validation_data))
        return hist


    def feature_extract(self,X_data):
        get_5th_layer_output = K.function([self.model.layers[0].input],[self.model.layers[self.compressed_layer].output])
        compressed = get_5th_layer_output([X_data])[0]
        compressed = compressed.reshape(compressed.shape[0], compressed.shape[1]*compressed.shape[2]*compressed.shape[3])
        return compressed


    def load(self, name):
        self.model.load_weights(name)


    def save(self, name):
        self.model.save_weights(name)