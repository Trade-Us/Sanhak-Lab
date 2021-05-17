from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
from keras.models import load_model
#from keras import optimizers
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from utils import optimizer_set

def CNNAutoEncoder_28(optimizer,learning_rate,activation='sigmoid', loss='mse'):
    optimizer = optimizer_set(optimizer, learning_rate)
    model = Sequential()
 
    #1st convolution layer
    model.add(Conv2D(16, (3, 3) #16 is number of filters and (3, 3) is the size of the filter.
    , padding='same', input_shape=(28,28,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    #2nd convolution layer
    model.add(Conv2D(2,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    #here compressed version

    #3rd convolution layer
    model.add(Conv2D(2,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))

    #4rd convolution layer
    model.add(Conv2D(16,(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(1,(3, 3), padding='same'))
    model.add(Activation('sigmoid'))
    model.summary()
    
    
    model.compile(optimizer=optimizer, loss=loss) #사용자 지정 파라미터(optimizer, loss)
    return model

def CNNAutoEncoder_96(optimizer, dimension, learning_rate, activation='sigmoid', loss='mse'):
    filter_size = 0
    channel = 0
    if dimension == 4:
        filter_size = 12; channel = 1
    elif dimension == 8:
        filter_size = 12; channel = 2
    elif dimension == 16:
        filter_size = 6; channel = 1
    elif dimension == 32:
        filter_size = 6; channel = 2
        
    
    optimizer = optimizer_set(optimizer, learning_rate)
    model = Sequential()

    #1st convolution layer
    model.add(Conv2D(32, (3, 3) #16 is number of filters and (3, 3) is the size of the filter.
    , padding='same', input_shape=(96,96,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    #2nd convolution layer
    model.add(Conv2D(16,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    #3rd convolution layer
    model.add(Conv2D(channel,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(filter_size,filter_size), padding='same'))

    #here compressed version

    #4th convolution layer
    model.add(Conv2D(channel,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(UpSampling2D((filter_size, filter_size)))

    #5th convolution layer
    model.add(Conv2D(16,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))

    #6th convolution layer
    model.add(Conv2D(32,(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(1,(3, 3), padding='same'))
    model.add(Activation(activation))
    model.summary()
    
    #학습(약 7000번 정도 진행)
    model.compile(optimizer=optimizer, loss=loss) #사용자 지정 파라미터(optimizer, loss)
    return model

def decide_filter_channel(image_size, dimension):
    if image_size == 1024:
        if dimension == 128:
            return 16, 2
        elif dimension == 64:
            return 16, 1
        elif dimension == 32:
            return 32, 2
        elif dimension == 16:
            return 32, 1
        else:
            print("Not good Value")
    if image_size == 256:
        if dimension == 32:
            return 8, 2
        elif dimension == 16:
            return 8, 1
        elif dimension == 8:
            return 16, 2
        elif dimension == 4:
            return 16, 1
        else:
            print("Not good Value")
def MakeAutoencoderModel(image_size, dimension, optimizer, learning_rate, activation='sigmoid', loss='binary_crossentropy'):
    assert image_size == 256 or image_size == 1024, "Wrong image Size"
    it = 2
    if image_size == 1028:
        it = 4
    other_channel = 128
    filter_size, channel = decide_filter_channel(image_size, dimension)
    
    
    optimizer = optimizer_set(optimizer, learning_rate)
    model = Sequential()

    #1st convolution layer
    model.add(Conv2D(other_channel, (3, 3), padding='same', input_shape=(image_size, image_size,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
      
    for i in range(it):
        other_channel = other_channel / 2
        # #2nd convoluti
        model.add(Conv2D(other_channel, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    #4rd convolution layer
    model.add(Conv2D(channel,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(filter_size,filter_size), padding='same'))
    
    #here compressed version
    #5th convolution layer
    model.add(Conv2D(channel,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(UpSampling2D((filter_size, filter_size)))
    
    for i in range(it+1):
        #6th convolution layer
        model.add(Conv2D(other_channel, (3, 3), padding='same')) # apply 2 filters sized of (3x3)
        model.add(Activation('relu'))
        model.add(UpSampling2D((2, 2)))
        
        other_channel = other_channel * 2

    model.add(Conv2D(1,(3, 3), padding='same'))
    model.add(Activation(activation))
    model.summary()
    model.compile(optimizer=optimizer, loss=loss) #사용자 지정 파라미터(optimizer, loss)
    return model