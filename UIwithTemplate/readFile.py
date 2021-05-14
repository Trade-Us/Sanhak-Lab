import pandas as pd
import numpy as np
from itertools import product


def split_into_values(df, columns):
    # Columns 사용 Parameter 종류 추출
    params_list = []
    for param in columns:
        df_params = df.loc[:, [param]].drop_duplicates().reset_index().drop(columns='index').transpose()
        params_list.append(df_params.values[0].tolist())

    # Parameter Columns 가 가질 수 있는 모든 조합 생성
    parameter_combination_sets = list(product(*params_list))

    result_df = pd.DataFrame() # 결과 DataFrame
    for param_set in parameter_combination_sets:
        part_df = df.copy()
        for i, param in enumerate(param_set):
            # Parameter 각 조합을 반복문을 통해 파싱한다.
            part_df = part_df.loc[part_df[columns[i]] == param, :]
        # 값들만 결과에 저장
        part_df = part_df.reset_index().drop(columns=['index']+columns).transpose()
        result_df = result_df.append(part_df, ignore_index=True)
    # 반환
    return result_df[pd.notnull(result_df[0])]
