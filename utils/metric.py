# 평가 지표 함수
import numpy as np
from sklearn.metrics import silhouette_score
# Accuracy를 가져오는 함수
# data_y: 정답 데이터
# pred_y: 예측 데이터
def getAccuracy(data_y, pred_y):
    count = 0
    bool_array = (data_y == pred_y)
    for correct in bool_array:
        if(correct):
            count += 1
    return count / pred_y.size

def plotSilhouette(X, labels):
    if max(labels) > 0:
        return silhouette_score(X, labels ,metric='euclidean')
    else:
        return "군집 개수 2미만으로 측정 불가"
