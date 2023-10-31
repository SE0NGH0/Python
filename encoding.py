# 필요한 라이브러리 임포트
from sklearn.preprocessing import LabelEncoder  # 라벨 인코딩을 위한 라이브러리 임포트
from sklearn.preprocessing import OneHotEncoder  # 원핫 인코딩을 위한 라이브러리 임포트
import numpy as np  # 배열 조작을 위한 넘파이 라이브러리 임포트
import pandas as pd  # 데이터프레임 생성 및 가공을 위한 판다스 라이브러리 임포트
from sklearn.datasets import load_iris  # 아이리스 데이터셋 로드
from sklearn.preprocessing import StandardScaler  # 표준화를 위한 스케일러 라이브러리 임포트
from sklearn.preprocessing import MinMaxScaler  # 최소-최대 스케일링을 위한 스케일러 라이브러리 임포트

# 라벨 인코딩을 수행하는 부분
items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']
encoder = LabelEncoder()  # LabelEncoder 객체 생성
encoder.fit(items)  # 데이터를 기반으로 라벨 인코더 학습
print(encoder.classes_)  # 학습된 클래스(라벨) 출력

items_trans = encoder.transform(items)  # 데이터를 라벨로 변환
print(items_trans)  # 라벨로 변환된 데이터 출력

encoder.inverse_transform(items_trans)  # 라벨을 다시 원래 데이터로 변환

# 원핫 인코딩을 수행하는 부분
encoder = LabelEncoder()  # 새로운 LabelEncoder 객체 생성
encoder.fit(items)  # 데이터를 기반으로 라벨 인코더 학습
items_trans = encoder.transform(items)  # 데이터를 라벨로 변환
print(items_trans.reshape(-1, 1))  # 2D 배열로 변환

oh_encoder = OneHotEncoder(sparse=False)  # OneHotEncoder 객체 생성 (희소 행렬 사용 안 함)
oh_encoder.fit(items_trans.reshape(-1, 1))  # 데이터를 기반으로 원핫 인코더 학습
print(oh_encoder.transform(items_trans.reshape(-1, 1))) # 원핫 인코딩 수행

# 판다스 데이터프레임 생성
df = pd.DataFrame({'item': items})
print(df)

pd.get_dummies(df, dtype=int)  # 판다스를 사용한 원핫 인코딩 수행

# 아이리스 데이터셋 로드 및 표준화 수행
iris = load_iris()  # 아이리스 데이터셋 로드
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)  # 데이터프레임 생성

s_scaler = StandardScaler()  # 표준화 스케일러 객체 생성
iris_df_s = s_scaler.fit_transform(iris_df)  # 데이터 표준화 수행
print(iris_df_s.var())  # 분산 출력
print(iris_df_s.std())  # 표준 편차 출력

# 최소-최대 스케일링 수행
m_scaler = MinMaxScaler()  # 최소-최대 스케일러 객체 생성
iris_df_m = m_scaler.fit_transform(iris_df)  # 데이터 최소-최대 스케일링 수행
print(iris_df_m.min())  # 최소값 출력
print(iris_df_m.max())  # 최대값 출력

# 최소-최대 스케일링 예제
train = np.arange(0, 11).reshape(-1, 1)  # 0에서 10까지의 값으로 이루어진 훈련 데이터 생성
test = np.arange(0, 12).reshape(-1, 1)  # 0에서 11까지의 값으로 이루어진 테스트 데이터 생성

scaler_m = MinMaxScaler()  # 최소-최대 스케일러 객체 생성
scaler_m.fit(train)  # 훈련 데이터를 기반으로 스케일러 학습
scaler_m.transform(train).reshape(-1)  # 훈련 데이터 스케일링 및 1D 배열로 변환
scaler_m.transform(test).reshape(-1)  # 테스트 데이터 스케일링 및 1D 배열로 변환
print(scaler_m.fit_transform(test).reshape(-1))  # 테스트 데이터 스케일링 및 1D 배열로 변환하여 출력