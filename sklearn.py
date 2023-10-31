import sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Iris 데이터셋을 로드합니다.
iris = load_iris()
# iris는 Bunch 객체로, 다양한 정보를 포함하고 있습니다.

# 데이터의 타입을 확인합니다.
type(iris)  # bunch는 사전(dictionary)과 비슷한 데이터 타입입니다.

# Iris 데이터의 특성(Features)을 출력합니다.
print(iris['data'])

# Iris 데이터에 포함된 키(필드)들을 출력합니다.
print(iris.keys())

# Iris의 대상 클래스(타겟) 이름을 출력합니다.
print(iris.target_names)

# Iris 데이터셋의 자세한 정보를 출력합니다.
print(iris.DESCR)

# Iris 데이터를 DataFrame으로 변환합니다.
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# 'label' 열을 추가하여 대상 클래스(타겟) 정보를 추가합니다.
iris_df['label'] = iris.target

# 데이터와 대상 클래스(타겟) 정보를 학습 데이터와 테스트 데이터로 분할합니다.
X_train, X_test, y_train, y_test = train_test_split(iris.data,
                 iris.target,
                 test_size=0.2, 
                 random_state=11
)

# 의사결정 트리(Decision Tree) 분류 모델을 생성하고 학습합니다.
dt_clf = DecisionTreeClassifier(random_state=11)
dt_clf.fit(X_train, y_train)
# 테스트 데이터를 사용하여 예측합니다.
pred = dt_clf.predict(X_test)

# 실제 타겟과 예측 결과를 출력합니다.
print(y_test, pred)

# 정확도를 평가하기 위해 accuracy_score를 사용합니다.
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)

# K-Fold 교차 검증(K-Fold Cross Validation)을 수행합니다.
from sklearn.model_selection import KFold
import numpy as np

# Iris 데이터의 크기를 출력합니다.
iris = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)
print('붓꽃 데이터 세트 크기:', iris.data.shape[0])

# K-Fold Cross Validation을 설정합니다.
kfold = KFold(n_splits=5, shuffle=True, random_state=None)
n_iter = 0
cv_accuracy = []
for train_index, test_index in kfold.split(iris.data):
    X_train, X_test = iris.data[train_index], iris.data[test_index]
    y_train, y_test = iris.target[train_index], iris.target[test_index]
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    n_iter += 1
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    cv_accuracy.append(accuracy)

# 각 폴드별 정확도와 평균 정확도를 출력합니다.
print('정확도:', cv_accuracy)
print('평균 정확도:', np.mean(cv_accuracy))

# Stratified K-Fold 교차 검증을 수행합니다.
from sklearn.model_selection import StratifiedKFold

# Stratified K-Fold Cross Validation을 설정합니다.
kfold = StratifiedKFold(n_splits=3, shuffle=False)
n_iter = 0
for train_index, test_index in kfold.split(iris.data, iris_df['label']):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print('## 교차 검증:{0}'.format(n_iter))
    print('학습 레이블 데이터 분포:\n', label_train.value_counts())
    print('검증 레이블 데이터 분포:\n', label_test.value_counts())

# 정확도를 기록하기 위한 리스트를 설정합니다.
cv_accuracy = []
kfold = StratifiedKFold(n_splits=5, shuffle=False)
for train_index, test_index in kfold.split(iris.data, iris_df['label']):
    X_train, X_test = iris.data[train_index], iris.data[test_index]
    y_train, y_test = iris.target[train_index], iris.target[test_index]
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    cv_accuracy.append(accuracy)

# 각 폴드별 정확도와 평균 정확도를 출력합니다.
print('정확도:', cv_accuracy)
print('평균 정확도:', np.mean(cv_accuracy))

# 교차 검증을 사용하여 모델을 평가합니다.
from sklearn.model_selection import cross_val_score

# 의사결정 트리 모델을 사용하여 교차 검증을 수행합니다.
iris = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)
scores = cross_val_score(dt_clf, iris.data, iris.target, cv=3, scoring='accuracy')
# 각 교차 검증별 정확도와 평균 정확도를 출력합니다.
print('교차 검증별 정확도:', np.round(scores, 4))
print('평균 검증 정확도:', np.round(np.mean(scores), 4))

# 그리드 서치(Grid Search)를 사용하여 하이퍼파라미터 튜닝을 수행합니다.
from sklearn.model_selection import GridSearchCV

iris = load_iris()
dtree = DecisionTreeClassifier()
grid_param = {
    'max_depth': [1, 2, 3, 4],
    'min_samples_split': [2, 3, 4],
}
grid = GridSearchCV(estimator=dtree, param_grid=grid_param, cv=3, refit=True)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=121)
grid.fit(X_train, y_train)

# 그리드 서치의 결과를 DataFrame으로 변환하여 출력합니다.
scores_df = pd.DataFrame(grid.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score']]

# 최적의 하이퍼파라미터로 학습한 모델을 사용하여 예측합니다.
pred = grid.best_estimator_.predict(X_test)

# 테스트 데이터의 실제 결과와 예측 결과를 비교하고 정확도를 출력합니다.
print(accuracy_score(y_test, pred))