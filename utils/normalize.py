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
    result_list = data.values.tolist()
    result_T = [list(x) for x in zip(*result_list)]
    RS = RobustScaler().fit(result_T)
    scaled = RS.transform(result_T)
    result_scaled = [list(x) for x in zip(*scaled)]
    return result_scaled

def MaxAbsScaler(data):
    result_list = data.values.tolist()
    result_T = [list(x) for x in zip(*result_list)]
    MAS = MaxAbsScaler().fit(result_T)
    scaled = MAS.transform(result_T)
    result_scaled = [list(x) for x in zip(*scaled)]
    return result_scaled

def tsleanr_scaler(data):
    result_list = data.values.tolist()
    TSS = TimeSeriesScalerMeanVariance().fit(result_list)
    scaled = TSS.transform(result_list)
    return scaled