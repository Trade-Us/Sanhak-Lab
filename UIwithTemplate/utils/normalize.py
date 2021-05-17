from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobusterScaler, MaxAbsScaler
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import numpy as np

def MinMax(data):
    MMS = MinMaxScaler().fit(data)
    scaled = MMS.transform(data)
    return scaled

def Standard(data):
    SS = StandardScaler().fit(data)
    scaled = SS.transform(data)
    return scaled

def Robust(data):
    RS = RobusterScaler().fit(data)
    scaled = RS.transform(data)
    return scaled

def MaxAbsScaler(data):
    MAS = MaxAbsScaler().fit(data)
    scaled = MAS.transform(data)
    return scaled

def tsleanr_scaler(data):
    TSS = TimeSeriesScalerMeanVariance().fit(data)
    scaled = TSS.transform(data)