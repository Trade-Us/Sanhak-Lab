from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import numpy as np

def MinMax(data):
    result_list = data.values.tolist()
    result_T = [list(x) for x in zip(*result_list)]
    MMS = MinMaxScaler().fit(result_T)
    scaled = MMS.transform(result_T)
    result_scaled = [list(x) for x in zip(*scaled)]
    return result_scaled

def Standard(data):
    result_list = data.values.tolist()
    result_T = [list(x) for x in zip(*result_list)]
    SS = StandardScaler().fit(result_T)
    scaled = SS.transform(result_T)
    result_scaled = [list(x) for x in zip(*scaled)]
    return result_scaled

def Robust(data):
    RS = RobustScaler().fit(data)
    scaled = RS.transform(data)
    return scaled

def MaxAbsScaler(data):
    MAS = MaxAbsScaler().fit(data)
    scaled = MAS.transform(data)
    return scaled

def tsleanr_scaler(data):
    TSS = TimeSeriesScalerMeanVariance().fit(data)
    scaled = TSS.transform(data)