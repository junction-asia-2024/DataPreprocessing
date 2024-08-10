import pandas as pd
import numpy as np
from numpy.typing import NDArray
from datetime import datetime


# 데이터 로드
data = pd.read_csv("total_pohang.csv")

# 주소에서 필요한 부분 추출
data["do"] = data["address"].str.split(" ").apply(lambda x: x[0])
data["si"] = data["address"].str.split(" ").apply(lambda x: x[1])
data["gu"] = data["address"].str.split(" ").apply(lambda x: x[2])
data["dong"] = data["address"].str.split(" ").apply(lambda x: x[3])

def ranking_danger_combined() -> dict:
    # 유니크 값 추출
    unique_classnames: NDArray = np.unique(data["classname"])
    unique_gu: NDArray = np.unique(data["gu"])

    # 'time' 열을 datetime 형식으로 변환
    data['time'] = pd.to_datetime(data['time'])
    
    # 최저 날짜와 최고 날짜 계산
    min_date: datetime = data['time'].min()
    max_date: datetime = data['time'].max()
    
    # 결과를 문자열로 포맷
    min_date_str: str = min_date.strftime("%Y-%m-%d %H:%M:%S")
    max_date_str: str = max_date.strftime("%Y-%m-%d %H:%M:%S")
    
    # 결과를 저장할 딕셔너리
    result: dict = {}
    
    # 각 gu별로 city와 classname별 사건 수 계산
    for gu in unique_gu:
        gu_data = data[data["gu"] == gu]
        gu_data_n: dict = {}
        total_count = 0  # 총 사건 수

        for classname in unique_classnames:
            class_data = gu_data[gu_data["classname"] == classname]
            crack_counts = class_data.groupby("dong").size()
            gu_data_n[classname] = crack_counts.sort_values(ascending=False).to_dict()
            total_count += crack_counts.sum()
        
        # 각 구에 대한 데이터 저장
        result[gu] = {
            "total": int(total_count),
            **gu_data_n  # 각 클래스별 동별 사건 수 추가
        }
    
    # 최저 및 최고 날짜 추가
    result["date_min_time"] = min_date_str
    result["date_max_time"] = max_date_str

    return result