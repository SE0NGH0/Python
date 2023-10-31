import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import Binarizer
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 데이터 전처리 함수 정의
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)  # 'Age' 열의 누락된 값들을 평균값으로 대체
    df['Cabin'].fillna('N', inplace=True)  # 'Cabin' 열의 누락된 값들을 'N'으로 대체
    df['Embarked'].fillna('N', inplace=True)  # 'Embarked' 열의 누락된 값들을 'N'으로 대체
    df['Fare'].fillna(0, inplace=True)  # 'Fare' 열의 누락된 값들을 0으로 대체
    return df

def drop_features(df):
    df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)  # 불필요한 열 삭제
    return df

def format_features(df):
    from sklearn.preprocessing import LabelEncoder
    df['Cabin'] = df.Cabin.str[0]  # 'Cabin' 열의 값을 첫 글자로 변환
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

# 분류 모델의 평가 지표 출력 함수 정의
def get_clf_eval(y_test, pred, pred_proba=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score
    # 정확도, 정밀도, 재현율, confusion matrix 출력
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba) if pred_proba is not None else 0  # ROC AUC 점수 (예측 확률이 있는 경우)
    print('오차행렬')
    print(confusion)
    print(f'정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}')

# 'titanic_train.csv' 파일 읽고 데이터 전처리
df = pd.read_csv('titanic_train.csv')
y = df.Survived
x = df.drop(columns=['Survived'])
x = transform_features(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)

# 로지스틱 회귀 모델 학습
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(max_iter=150)
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
get_clf_eval(y_test, pred)  # 분류 모델 평가 출력

# 로지스틱 회귀 모델의 예측 확률 출력
pred_proba = lr_clf.predict_proba(X_test)
print(pred_proba)

# Binarizer를 사용한 임계값 변환 예제
X = [[1, -1, 2], [2, 0, 0], [0, 1.1, 1.2]]
binarizer = Binarizer(threshold=0.0)  # 임계값을 0.0으로 설정 (0.0보다 크면 1, 작으면 0)
binarizer.fit_transform(X)

# 사용자 정의 임계값을 적용한 예제
custom_threshold = 0.5
pred_proba_1 = pred_proba[:, 1].reshape(-1, 1)
print(pred_proba_1)

binarizer = Binarizer(threshold=custom_threshold)
custom_pred = binarizer.fit_transform(pred_proba_1)
print(custom_pred)

get_clf_eval(y_test, custom_pred)  # 사용자 정의 임계값에 따른 분류 모델 평가

# 다양한 사용자 정의 임계값에 따른 분류 모델 평가
custom_threshold = 0.4
pred_proba_1 = pred_proba[:, 1].reshape(-1, 1)
binarizer = Binarizer(threshold=custom_threshold)
custom_pred = binarizer.fit_transform(pred_proba_1)
get_clf_eval(y_test, custom_pred)

custom_thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]

# 다양한 사용자 정의 임계값에 따른 분류 모델 평가 함수
def get_eval_by_threshold(y_test, pred_proba_c1, custom_thresholds):
    for custom_threshold in custom_thresholds:
        binarizer = Binarizer(threshold=custom_threshold)
        custom_pred = binarizer.fit_transform(pred_proba_c1)
        print('임계값:', custom_threshold)
        get_clf_eval(y_test, custom_pred)

get_eval_by_threshold(y_test, pred_proba[:, 1].reshape(-1, 1), custom_thresholds)

# 정밀도-재현율 곡선 및 임계값 변화 시각화 함수
def precision_recall_curve_plot(y_test, pred_proba_c1):
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)
    threshold_boundary = thresholds.shape[0]
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, precisions[:-1], linestyle='--', label='정밀도')
    plt.plot(thresholds, recalls[:-1], label='재현율')
    start, end = plt.xlim()
    print(start, end)
    plt.xticks(np.round(np.arange(0, 1, 0.1), 2))
    plt.grid()
    plt.legend()
    plt.xlabel('임계값의 변화')
    plt.ylabel('정밀도, 재현율')

precision_recall_curve_plot(y_test, lr_clf.predict_proba(X_test)[:, 1])

custom_thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]
get_eval_by_threshold(y_test, pred_proba[:, 1].reshape(-1, 1), custom_thresholds)