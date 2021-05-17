from keras import optimizers

def optimizer_set(optimizer, learning_rate):
    if optimizer == 'SGD':
        optimizer = optimizers.SGD(lr=learning_rate, clipnorm=1.)
        
    if optimizer == 'RMSprop':
        optimizer = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
        
    if optimizer == 'Adam':
        optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        
    if optimizer == 'Nadam':
        optimizer = optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        
    if optimizer == 'Adagrad':
        optimizer = optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.0)
        
    if optimizer == 'Adadelta':
        optimizer = optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=None, decay=0.0)
    return optimizer

from sklearn.model_selection import train_test_split
import numpy as np

def split_data(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
    # xy = (X_train, X_test, Y_train, Y_test)
    # np.save("./img_data.npy", xy)
    # X_train, X_test, Y_train, Y_test = np.load('./img_data.npy', allow_pickle = True)
    
    return X_train, X_test, Y_train, Y_test

def normalization_tool(data):
    data_nor = data.astype('float32') / 255
    return data_nor