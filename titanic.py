import pandas as pd
# 판다스를 사용하여 'titanic_train.csv' 파일을 읽어 데이터프레임을 생성
df = pd.read_csv('titanic_train.csv')
print(df)  # 데이터프레임 출력

# 'Age' 열의 누락된 값을 해당 열의 평균값으로 대체
df['Age'].fillna(df['Age'].mean(), inplace=True)

# 'Cabin'과 'Embarked' 열의 누락된 값을 각각 'N'으로 대체
df['Cabin'].fillna('N', inplace=True)
df['Embarked'].fillna('N', inplace=True)

# 'Sex'와 'Survived' 열을 기반으로 그룹화하고, 각 그룹별로 'Survived' 값의 개수를 세어 출력
print(df.groupby(['Sex', 'Survived'])['Survived'].count())

import seaborn as sns
# seaborn을 사용하여 성별에 따른 생존율을 막대 그래프로 표시
sns.barplot(data=df, x='Sex', y='Survived')

# seaborn을 사용하여 객실 등급(Pclass)에 따른 생존율을 막대 그래프로 표시
sns.barplot(data=df, y='Survived', x='Pclass')

# seaborn을 사용하여 객실 등급(Pclass) 및 성별(Sex)에 따른 생존율을 막대 그래프로 표시
sns.barplot(data=df, y='Survived', x='Pclass', hue='Sex')

# seaborn을 사용하여 연령(Age)에 따른 생존율을 막대 그래프로 표시
sns.barplot(data=df, y='Survived', x='Age')

# 연령 카테고리를 반환하는 함수 정의
def get_category(age):
    cat = ''
    if age <= -1:
        cat = 'Unknown'
    elif age <= 5:
        cat = 'Baby'
    elif age <= 12:
        cat = 'Child'
    elif age <= 18:
        cat = 'Teenager'
    elif age <= 25:
        cat = 'Student'
    elif age <= 35:
        cat = 'Young Adult'
    elif age <= 60:
        cat = 'Adult'
    else:
        cat = 'Elderly'
    return cat

# 'Age_cat' 열을 생성하고 연령 카테고리로 채움
df['Age_cat'] = df.Age.apply(lambda x: get_category(x))

# seaborn을 사용하여 연령 카테고리(Age_cat)에 따른 생존율을 막대 그래프로 표시
sns.barplot(data=df, y='Survived', x='Age_cat')

# seaborn을 사용하여 연령 카테고리(Age_cat) 및 성별(Sex)에 따른 생존율을 막대 그래프로 표시
sns.barplot(data=df, y='Survived', x='Age_cat', hue='Sex')

# 그래프 순서를 지정하기 위한 카테고리 순서 리스트 정의
order_name = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Elderly']

# seaborn을 사용하여 연령 카테고리(Age_cat) 및 성별(Sex)에 따른 생존율을 막대 그래프로 표시하고, 그래프 순서를 order_name에 지정된 순서로 설정
sns.barplot(data=df, y='Survived', x='Age_cat', hue='Sex', order=order_name)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))  # 그래프의 사이즈를 조절

# 다양한 데이터 전처리 함수 정의
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)
    return df

def drop_features(df):
    df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
    return df

def format_features(df):
    from sklearn.preprocessing import LabelEncoder
    df['Cabin'] = df.Cabin.str[0]
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        print(le.classes_)
    return df

def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df

# 'titanic_train.csv' 파일을 다시 읽고, 종속변수와 독립변수로 데이터를 분리하고 데이터 전처리를 수행
df = pd.read_csv('titanic_train.csv')
y = df.Survived
x = df.drop(columns=['Survived'])
x = transform_features(x)

# 학습 데이터와 테스트 데이터로 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)

# 의사결정 트리, 랜덤 포레스트, 로지스틱 회귀를 사용하여 학습하고 예측
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  # 회귀모델을 이용한 이진 분류
from sklearn.metrics import accuracy_score

dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression(max_iter=200)

dt_clf.fit(X_train, y_train)
pred_dt = dt_clf.predict(X_test)
print(accuracy_score(y_test, pred_dt))

rf_clf.fit(X_train, y_train)
pred_rf = rf_clf.predict(X_test)
print(accuracy_score(y_test, pred_rf))

lr_clf.fit(X_train, y_train)
pred_lr = lr_clf.predict(X_test)
print(accuracy_score(y_test, pred_lr))

# 하이퍼파라미터 튜닝을 위한 그리드 서치 수행
from sklearn.model_selection import GridSearchCV

param = {
    'max_depth': [2, 3, 5, 10],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 5, 8]
}
grid_dt = GridSearchCV(dt_clf, param_grid=param, scoring='accuracy', cv=5)
grid_rf = GridSearchCV(rf_clf, param_grid=param, scoring='accuracy', cv=5)
grid_dt.fit(X_train, y_train)
grid_rf.fit(X_train, y_train)

pred_dt = grid_dt.best_estimator_.predict(X_test)
pred_rf = grid_rf.best_estimator_.predict(X_test)

print(accuracy_score(y_test, pred_dt))
print(accuracy_score(y_test, pred_rf))