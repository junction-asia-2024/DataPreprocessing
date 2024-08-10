import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

data = pd.read_csv("total_pohang.csv")

# 타임스탬프 변환
data['time'] = pd.to_datetime(data['time'])
data['hour'] = data['time'].dt.hour
data['day'] = data['time'].dt.day
data['month'] = data['time'].dt.month
data['year'] = data['time'].dt.year

# 문제 발생 시간대 분석
hourly_counts = data.groupby('hour').size()
plt.figure(figsize=(12, 6))
hourly_counts.plot(kind='bar')
plt.title('Pothole Incidents by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Incidents')
plt.show()

# 계절성 분석
monthly_counts = data.groupby('month').size()
plt.figure(figsize=(12, 6))
monthly_counts.plot(kind='bar')
plt.title('Pothole Incidents by Month')
plt.xlabel('Month')
plt.ylabel('Number of Incidents')
plt.show()



# 예측을 위한 특성 선택 및 데이터 준비
X = data[['hour', 'day', 'month', 'year', 'latitude', 'longitude']]
y = data['classname']  # 이진 분류를 위한 클래스

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 훈련
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

