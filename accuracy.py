from sklearn.base import BaseEstimator
import numpy as np

class MyDummyClassifier(BaseEstimator):
    def fit(self, x, y=None):
        pass
    
    def predict(self, x):
        pred = np.zeros((x.shape[0], 1))  # 0으로 채워진 배열 생성
        for i in range(x.shape[0]):
            if x['Sex'].iloc[i] == 1:  # 'Sex' 열의 값이 1인 경우 0으로 예측
                pred[i] = 0
            else:
                pred[i] = 1  # 'Sex' 열의 값이 1이 아닌 경우 1로 예측
        return pred  # 예측 결과 반환

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 전처리 함수들 정의
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)  # 'Age' 열의 누락된 값 평균값으로 대체
    df['Cabin'].fillna('N', inplace=True)  # 'Cabin' 열의 누락된 값 'N'으로 대체
    df['Embarked'].fillna('N', inplace=True)  # 'Embarked' 열의 누락된 값 'N'으로 대체
    df['Fare'].fillna(0, inplace=True)  # 'Fare' 열의 누락된 값 0으로 대체
    return df

def drop_features(df):
    df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)  # 불필요한 열 삭제
    return df

def format_features(df):
    from sklearn.preprocessing import LabelEncoder
    df['Cabin'] = df.Cabin.str[0]  # 'Cabin' 열의 첫 글자만 남김
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])  # 레이블 인코딩 수행
        print(le.classes_)  # 인코딩된 클래스(라벨) 출력
    return df

def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df

# 'titanic_train.csv' 파일을 읽고 데이터 전처리 및 분리
df = pd.read_csv('titanic_train.csv')
y = df.Survived  # 종속변수
x = df.drop(columns=['Survived'])  # 독립변수
x = transform_features(x)

# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

myclf = MyDummyClassifier()  # 사용자 정의 더미 분류기 객체 생성
myclf.fit(X_train, y_train)  # 더미 분류기 학습
pred = myclf.predict(X_test)  # 테스트 데이터에 대한 예측 수행
print(accuracy_score(y_test, pred))  # 정확도 출력