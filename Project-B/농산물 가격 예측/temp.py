import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
display(train.head())

train.info()

train['date'] = pd.______(train['date'])
display(train['date'].info())

import matplotlib.pyplot as plt

item_selected = '배추'

train_last = train.tail(100)

plt.figure(figsize=(12, 4))

plt.subplot(1,2,1)
plt.plot(train[_____], train[_______])
plt.title(f"{item_selected}_가격(원/kg) 시계열 그래프")
plt.xlabel('날짜')
plt.ylabel(f"{item_selected}_가격(원/kg)")

plt.subplot(1,2,2)
plt.plot(____[____], ____[_____])
plt.title(f"{item_selected}_가격(원/kg) 시계열 그래프 : {train_last['date'].min().date()} ~ {train_last['date'].max().date()}")
plt.xlabel('날짜')
plt.ylabel(f"{item_selected}_가격(원/kg)")

plt.tight_layout()
plt.show()

train_prep = train.copy()

price_columns = [col for col in train_prep._____ if '____' in ____]

for col in price_columns:
    train_prep.____(0, method = '____', ____=True)

display(train.head())
display(train_prep.head())

import matplotlib.pyplot as plt

plt.figure(figsize=(16, 20))

for i, feature in enumerate(train_prep.columns[1:]):
    plt.subplot(8,2,i+1)
    plt.plot(train_prep['date'], train_prep[feature])
    plt.title(feature + '시계열 그래프')
    plt.xlabel('날짜')
    plt.ylabel(feature)

plt.tight_layout()
plt.grid(True)
plt.show()

import matplotlib.dates as mdates

item_selected = '배추'

# x_start, x_end를 날짜 형식으로 설정
start_date = '2020-04-10'
end_date = '2020-07-10'

train_selected = train_prep[(train_prep['date'] >= start_date) & (train_prep['date'] < end_date) ].copy()

# 다양한 윈도우 크기로 MA와 EMA 계산
window_size = 7

train_selected[f'{item_selected}_가격(원/kg)_MA'] = train_selected[f'{item_selected}_가격(원/kg)']._____(____=window_size).____()
train_selected[f'{item_selected}_가격(원/kg)_EMA'] = train_selected[f'{item_selected}_가격(원/kg)'].____(____=window_size, adjust=False).____()

# y축 범위를 각 그래프의 최대, 최소값으로 설정
y_min = min(train_selected[f'{item_selected}_가격(원/kg)'].min(), train_selected[f'{item_selected}_가격(원/kg)_MA'].min(), train_selected[f'{item_selected}_가격(원/kg)_EMA'].min())
y_max = max(train_selected[f'{item_selected}_가격(원/kg)'].max(), train_selected[f'{item_selected}_가격(원/kg)_MA'].max(), train_selected[f'{item_selected}_가격(원/kg)_EMA'].max())

plt.figure(figsize=(12, 4))

# 가격, MA, EMA 그래프 그리기
plt.plot(train_selected['date'], train_selected[f'{item_selected}_가격(원/kg)'], label='Original Price', alpha=0.5)
plt.plot(train_selected['date'],  train_selected[f'{item_selected}_가격(원/kg)_MA'], label=f'MA-{window_size} days')
plt.plot(train_selected['date'],  train_selected[f'{item_selected}_가격(원/kg)_EMA'], label=f'EMA-{window_size} days')

# x축과 y축의 범위를 설정
plt.xlim([train_selected['date'].min(), train_selected['date'].max()])
plt.ylim([y_min, y_max])

# x축을 날짜 형식으로 설정
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gcf().autofmt_xdate()

# 그래프의 제목과 레이블을 설정
plt.title(f'Moving Averages and Exponential Moving Averages of {item_selected}')
plt.xlabel('Date')
plt.ylabel('Price')

# 범례를 표시
plt.legend()

# 그래프를 출력
plt.show()

item_selected = '무'
ax = train_prep.____.____(f"{item_selected}_거래량(kg)",f"{item_selected}_가격(원/kg)", s=1)

plt.show()

# 모든 품목 이름 추출
item_names = [col.___('_')[0] for col in ____ if _____]

# 4x2 그리드로 그림을 초기화
fig, axes = plt.subplots(3, 3, figsize=(12, 8))

# axes 배열을 평평하게 만들기 (쉬운 인덱싱을 위해)
axes = axes.flatten()

# 각 품목 이름을 반복하며 scatter plot 그리기
for i, name in enumerate(item_names):
    ax = axes[i]
    volume_col = f"{name}_거래량(kg)"
    price_col = f"{name}_가격(원/kg)"

    # 거래량, 가격 컬러만을 추출
    data = train_prep[[volume_col, price_col]].copy()

    # 개별 스케일로 scatter plot 생성
    ax._____ (data[volume_col], data[price_col],  s=1)
    ax.set_title(f" {name} 거래량과 가격")
    ax.set_xlabel(" 거래량 (kg)")
    ax.set_ylabel("가격 (원/kg)")

    # 스케일 자동 조정
    ax.autoscale_view()

plt.tight_layout()
plt.show()

import seaborn as sns

# 가격에 대한 컬럼만 선택하여 상관 관계를 계산
price_columns = [col for col in train_prep.columns if '가격' in col]
correlation_matrix = train[price_columns].____()

# 퍼센트 변화를 계산
train_pct_change = train_prep[price_columns].____(7)

# 상관 관계 행렬 계산
correlation_matrix_pct_change = _____._____()

# 상관 관계 행렬을 시각화
plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1,2,1)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('품목 가격 간의 상관 관계')

ax2 = plt.subplot(1,2,2)
sns.______(correlation_matrix_pct_change, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('품목 가격의 퍼센트 변화 간의 상관 관계')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from pandas.plotting import lag_plot

# Lag 값과 타이틀 설정
lags = [1, 7, 14, 28]
titles = ['Lag 1', 'Lag 7', 'Lag 14', 'Lag 28']

# 그래프 그리기 설정
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

# 각 Lag 값에 따른 Scatter Plot 그리기
for ax, lag, title in zip(axes.flatten(), lags, titles):
    _____(train_prep['무_가격(원/kg)'], lag=____, s=1, ax=ax)
    ax.set_title(title)

# 전체 타이틀 및 레이아웃 설정
plt.suptitle('Lag Scatter Plots', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

import seaborn as sns

# 주(week)와 월(month) 정보를 새로운 컬럼에 저장
train_prep['Week'] = train_prep['date'].dt.isocalendar().week.astype(np.int32)
train_prep['month'] = train_prep['date'].dt.month

item_selected = '배추'

# 주와 월별 Box Plot 그리기
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plt.suptitle('Box Plot by Week and Month', fontsize=16)


# 주별 Box Plot
sns.____(x='____', y=_____, data=____, ax=axes[0])
axes[0].set_title('Box Plot by Week', fontsize=14)
axes[0].set_xlabel('week', fontsize=12)
axes[0].set_ylabel(f'{item_selected} 가격 (원/kg)', fontsize=12)

# 월별 Box Plot
sns.boxplot(x='_____', y=______, data=_____, ax=axes[1])
axes[1].set_title('Box Plot by Month', fontsize=14)
axes[1].set_xlabel('month', fontsize=12)
axes[1].set_ylabel(f'{item_selected} 가격 (원/kg)', fontsize=12)

plt.tight_layout()
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

item_selected = '배추'
start_date_train = '2021-01'

train_selected = train_prep[train_prep['date'] >= start_date_train].copy()

# 배추 가격 추출
train_selected_item = train_selected[['date', f'{item_selected}_가격(원/kg)']].copy()

# 계절성 분해 수행 (연별 계절성 가정) 및 그래프 그리기
plt.figure(figsize=(12, 8))
decomposition = _______(train_selected_item[f'{item_selected}_가격(원/kg)'], model='additive', period=30)

components = [decomposition.observed, decomposition.____, decomposition.____, decomposition.____]
titles = ['Original Series', 'Trend Component', 'Seasonal Component', 'Residual Component']

for idx, component in enumerate(____):
    plt.subplot(4, 1, idx + 1)
    plt.plot(train_selected_item['date'], ____)  # x축에 날짜 사용, y축에 성분 사용
    plt.title(titles[idx])
    plt.grid(True)

plt.tight_layout()

#for checkcode only
axes = plt.gcf().axes 

plt.show()
