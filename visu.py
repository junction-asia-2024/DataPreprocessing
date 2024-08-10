import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("total_pohang.csv")

# 타임스탬프 변환
data['time'] = pd.to_datetime(data['time'])
data['hour'] = data['time'].dt.hour
data['day'] = data['time'].dt.day
data['month'] = data['time'].dt.month
data['year'] = data['time'].dt.year

# # 문제 발생 시간대 분석
# hourly_counts = data.groupby('hour').size()
# plt.figure(figsize=(12, 6))
# hourly_counts.plot(kind='bar')
# plt.title('Pothole Incidents by Hour of Day')
# plt.xlabel('Hour of Day')
# plt.ylabel('Number of Incidents')
# plt.show()

# 계절성 분석
monthly_counts = data.groupby("month").size()
plt.figure(figsize=(12, 6))
monthly_counts.plot(kind='bar')
plt.title('Pothole Incidents by Month')
plt.xlabel('Month')
plt.ylabel('Number of Incidents')
plt.show()



