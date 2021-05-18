from tslearn.preprocessing import TimeSeriesResampler
from sklearn.model_selection import train_test_split
from utils.agent import Autoencoder_Agent
import pywt
import numpy as np

## tslearn resampling
def exec_ts_resampler(X, size):
    data = TimeSeriesResampler(sz=size).fit_transform(X)
    return data

## Autoeocoder
def fit_autoencoder(X, image_size, dimension, optimizer, learning_rate, activation, loss_func, batch_size, epochs):
    # input
    # 이미지 데이터, 이미지 사이즈, 추출 될 차원 크기, 최적화기, 학습률, 활성화함수, 손실함수, 배치 크기, 에포크

    #데이터 분리
    X_train, X_test, Y_train, Y_test = train_test_split(X, X) 

    # Autoencoder model Object 생성
    autoencoder = Autoencoder_Agent(model_size=image_size,
                                    optimizer=optimizer,
                                    learning_rate=learning_rate, 
                                    activation_function=activation,
                                    loss_function=loss_func,
                                    dimension=dimension)

    # fitting
    hist = autoencoder.train(X_train, batch_size, epochs, X_test, early_stopping=False)
    all_feature = np.array(autoencoder.feature_extract(X))
    return all_feature
    
## wavelet
def exec_wavelet(X, func, iter):
    for i in range(iter):
        data,trash = pywt.dwt(X, func)
    return data

# PCA

