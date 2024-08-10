import pandas as pd
import numpy as np

# 데이터 로드
data = pd.read_csv('total_pohang.csv')

# 타임스탬프 변환
data['time'] = pd.to_datetime(data['time'])
data['hour'] = data['time'].dt.hour
data['day'] = data['time'].dt.day
data['month'] = data['time'].dt.month
data['year'] = data['time'].dt.year

# 결측값 처리 (예: 위도와 경도)
data = data.dropna(subset=['latitude', 'longitude'])

# 문제 클래스 인코딩
data['classname'] = data['classname'].astype('category').cat.codes


print(data)