# 평가 지표 함수 

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