# pandas 라이브러리 가져오기
import pandas as pd

# CSV 파일을 읽어서 DataFrame 생성
df = pd.read_csv('data/scientists.csv')

# DataFrame 출력
print(df)

# DataFrame에서 'Name'과 'Age' 열 선택 및 출력
print(df['Name', 'Age'])

# 'Age' 열의 평균값 출력
print(df['Age'].mean())

# 'Age' 열의 값이 평균보다 큰지 확인하고 True/False 출력
print(df['Age'] > df['Age'].mean())

# 'Age' 열의 값이 평균보다 큰지 확인하고 결과를 리스트로 변환하여 출력
print(df['Age'].mean() < df['Age'].to_list())

# 'Age' 열을 자신과 더한 결과 출력
print(df['Age'] + df['Age'])

# 'Age' 열의 값에 100을 더한 결과 출력
print(df['Age'] + 100)

# 'Age' 열의 값에 Series([100, 1])을 더한 결과 출력
print(df['Age'] + pd.Series([100, 1]))

# DataFrame의 기본 정보 출력
print(df.info())

# 'Born' 열의 데이터 타입 출력
print(df['Born'].dtype)

# 'Born' 열의 데이터를 날짜 형식으로 변환
born_datetime = pd.to_datetime(df['Born'], format='%Y-%m-%d')
died_datetime = pd.to_datetime(df['Died'], format='%Y-%m-%d')

# 변환된 날짜 열 출력
print(born_datetime)

# DataFrame에 'Born_dt'와 'Died_dt' 열 추가
df['Born_dt'], df['Died_dt'] = (born_datetime, died_datetime)

# DataFrame의 정보 출력
print(df.info())

# 'Born_dt'와 'Died_dt' 열의 날짜 차이 계산 및 출력
print(df['Born_dt'] - df['Died_dt'])

# 'Born_dt' 열의 값과 365일을 더한 결과 출력
print(df['Born_dt'], df['Born_dt'] + pd.to_timedelta(365, 'day'))

# 'Born_dt'와 'Died_dt' 열을 삭제하고 결과 출력
print(df.drop(columns=['Born_dt', 'Died_dt']))

# 'Born_dt'와 'Died_dt' 열을 삭제하고 원본 DataFrame을 수정
df.drop(columns=['Born_dt', 'Died_dt'], inplace=True)

# datetime 라이브러리를 사용하여 현재 날짜 및 시간을 가져와서 출력
import datetime
now = datetime.datetime.now()
print(now)

# 현재 날짜를 원하는 형식으로 문자열로 변환하고 출력
date = now.strftime('%d일%m월%년')
print(date)

# 현재 시간을 원하는 형식으로 문자열로 변환하고 출력
time = now.strftime('%H:%M:%S')
print(time)

# 현재 날짜와 시간을 원하는 형식으로 문자열로 변환하고 출력
datetime = now.strftime('%Y-%m-%d %H:%M:%S')
print(datetime)
# CSV 파일을 DataFrame으로 읽어옵니다
df1 = pd.read_csv('data/concat_1.csv')
df2 = pd.read_csv('data/concat_2.csv')
df3 = pd.read_csv('data/concat_3.csv')

# df1의 내용을 출력합니다
print(df1)

# DataFrame을 가로로 합칩니다 (열을 따라)
pd.concat([df1, df2, df3], axis=1)

# DataFrame을 세로로 합칩니다 (행을 따라) 내부 조인(inner join)을 수행하며 인덱스를 무시합니다
pd.concat([df1, df2, df3], axis=0, join='inner', ignore_index=True)

# CSV 파일로부터 두 개의 DataFrame을 읽어옵니다: 'person'과 'survey'

person = pd.read_csv('data/survey_person.csv')
survey = pd.read_csv('data/survey_survey.csv')

# 'person'과 'survey' DataFrame을 지정된 열을 기반으로 외부 조인(outer join)하여 병합합니다
data = pd.merge(person, survey, left_on='idnet', right_on='person', how='outer')

# DataFrame 또는 Series에서 누락된 값(NaN 또는 None)을 확인합니다
pd.isnull('')  # 빈 문자열이 누락된 값인지 확인
pd.isna('')    # 빈 문자열이 NA인지 확인
pd.isnull(nan)  # 'nan' 변수가 누락된 값인지 확인
pd.notnull(nan)  # 'nan' 변수가 누락되지 않은 값인지 확인

# 'data' DataFrame의 각 열에 누락된 값의 수를 세어봅니다
pd.isnull(data).sum()

# 'data' DataFrame 전체에서 누락된 값의 총 수를 계산합니다
pd.isnull(data).sum().sum()

# CSV 파일에서 'ebola' DataFrame을 읽어옵니다
ebola = pd.read_csv('data/country_timeseries.csv')

# 'ebola' DataFrame의 각 열에 누락된 값의 수를 세어봅니다
ebola.isnull().sum()

# 'ebola' DataFrame의 누락된 값을 0으로 채웁니다 (원본 DataFrame을 수정하지 않음, inplace=False)
ebola.fillna(0, inplace=False)

# 뒤쪽 값으로 누락된 값을 채웁니다 (bfill)
ebola.fillna(method='bfill')

# 'ebola' DataFrame에서 누락된 값을 보간(interpolate)합니다
ebola.interpolate()

# 'ebola' DataFrame에서 누락된 행을 삭제합니다
ebola.dropna(axis=0)
# 'pew.csv' 파일을 읽어와 'pew' DataFrame을 생성합니다
pew = pd.read_csv('data/pew.csv')

# 'pew' DataFrame을 Melt하여 'income' 열을 새로 생성하고, 'count' 열의 값들을 새로운 열로 이동합니다
pd.melt(pew, id_vars='religion', var_name='income', value_name='count')

# 'billboard.csv' 파일을 읽어온 후, DataFrame의 컬럼 이름들을 출력합니다
billboard = pd.read_csv('data/billboard.csv')
print(billboard.columns)

# 'billboard' DataFrame을 Melt하여 'week' 열을 새로 생성하고, 'rating' 열의 값들을 새로운 열로 이동합니다
pd.melt(billboard, id_vars=['year', 'artist', 'track', 'time', 'date.entered'], var_name='week', value_name='rating')

# '인천광역시_소방장비 현황_20191231.csv' 파일을 cp949 인코딩으로 읽어옵니다
pd.read_csv('data/인천광역시_소방장비 현황_20191231.csv', encoding='cp949')

# 'country_timeseries.csv' 파일을 읽어온 후, 'Date'와 'Day' 열을 기준으로 Melt하여 'variable' 열을 생성합니다
ebola = pd.read_csv('data/country_timeseries.csv')
ebola_long = pd.melt(ebola, id_vars=['Date', 'Day'])

# 'variable' 열에서 'status'와 'country' 열을 추출하고 'variable' 열을 삭제합니다
ebola_long['variable'].str.split('_').str.get(1)
ebola_long['status'] = ebola_long['variable'].str.split('_').str.get(0)
ebola_long['country'] = ebola_long.variable.str.split('_').str[1]
ebola_long.drop(columns='variable', inplace=True)

# 'weather.csv' 파일을 읽어온 후, Melt하여 'day'와 'temp' 열을 생성합니다
weather = pd.read_csv('data/weather.csv')
pd.melt(weather, id_vars=['id', 'year', 'month', 'element'], var_name='day', value_name='temp')

# Melt한 DataFrame을 기반으로 Pivot하여 'element' 열을 컬럼으로 옮깁니다
melt_df = pd.melt(weather, id_vars=['id', 'year', 'month', 'element'], var_name='day', value_name='temp')
pivot_df = melt_df.pivot_table(index=['id', 'year', 'month', 'element', 'day'], columns='element', values='temp')

# 'pivot_df'를 기초로 만든 DataFrame을 리셋합니다 (인덱스 열을 열로 변환)
df1 = pivot_df.reset_index()

# 인덱스를 리셋하면서, 기존 인덱스를 삭제합니다
df2 = pivot_df.reset_index(drop=True)

# 'billboard.csv' 파일을 읽어온 후, Melt하여 'week' 열을 생성하고 'rating' 열의 값을 새로운 열로 이동합니다
billboard = pd.read_csv('data/billboard.csv')
billboard_melt = pd.melt(billboard, id_vars=['year', 'artist', 'track', 'time', 'date.entered'], var_name='week', value_name='rating')

# 'track' 열이 'Loser'인 행만 필터링합니다
billboard_melt[billboard_melt['track'] == 'Loser']

# 'year', 'artist', 'track', 'time' 열만 선택한 새로운 DataFrame을 생성하고, 중복 행을 제거합니다
billboard_songs = billboard_melt[['year', 'artist', 'track', 'time']]
billboard_songs.drop_duplicates(inplace=True)

# 인덱스를 리셋하면서, 기존 인덱스를 삭제합니다
billboard_songs.reset_index(drop=True)

# 'billboard_melt' DataFrame과 'billboard_songs' DataFrame을 'year', 'artist', 'track', 'time' 열을 기준으로 병합합니다
billboard_ratings = pd.merge(billboard_melt, billboard_songs, on=['year', 'artist', 'track', 'time'])

# 불필요한 열을 제거합니다
billboard_ratings.drop(columns=['year', 'artist', 'track', 'time'], inplace=True)

# 'billboard_ratings'와 'billboard_songs'를 'index' 열을 기준으로 병합합니다
pd.merge(billboard_ratings, billboard_songs, on='index')

# 'billboard_songs'와 'billboard_ratings'를 'index' 열을 기준으로 병합합니다
pd.merge(billboard_songs, billboard_ratings, on='index')
import seaborn as sns  # 데이터 시각화를 위한 Seaborn 라이브러리를 불러옵니다.
import pandas as pd   # 데이터 조작을 위한 Pandas 라이브러리를 불러옵니다.

# Seaborn의 'tips' 데이터셋을 불러와 'tips' 변수에 저장합니다.
# 이 데이터셋은 레스토랑 팁에 관한 정보를 포함하고 있습니다.
sns.load_dataset('tips')
tips = sns.load_dataset('tips')

# 'tips' DataFrame에 대한 정보를 .info() 메서드를 사용하여 출력합니다.
# 이는 DataFrame의 열, 데이터 유형 및 누락된 값에 관한 자세한 정보를 표시합니다.
print(tips.info())

# 'sex' 열의 데이터 유형을 문자열로 변경합니다. 이를 .astype() 메서드를 사용하여 수행합니다.
tips['sex'] = tips['sex'].astype(str)

# 'size' 열의 데이터 유형을 'int8'로 변환하려고 하지만, 이 코드는 실제로 DataFrame을 변경하지 않습니다.
# 변경을 적용하려면 결과를 다시 DataFrame에 할당해야 합니다.
tips['size'].astype('int8')

# 데이터 유형 변경이 적용된 'tips' DataFrame에 대한 정보를 출력합니다.
print(tips.info())

# 초기 데이터로 'tips' 데이터셋을 다시 불러옵니다.
tips = sns.load_dataset('tips')

# 'total_bill' 열의 특정 셀을 빈 문자열(' ')로 설정합니다.
tips.loc[[1, 3, 5, 7], 'total_bill'] = ' '

# 변경 사항을 확인하기 위해 'tips' DataFrame에 대한 정보를 출력합니다.
print(tips.info())

# 'total_bill' 열을 숫자 데이터 유형으로 변환하는데, 오류가 발생하면 NaN으로 처리합니다.
print(pd.to_numeric(tips['total_bill'], errors='coerce'))

# 'sex' 열의 값을 오름차순으로 정렬합니다.
print(tips['sex'].sort_values())

# 'sex' 열이 범주형 데이터 유형이라면 해당 범주를 출력합니다.
# 그러나 이 코드에서 'sex'는 이전에 문자열로 변환되었으므로 범주가 아니므로 유용한 정보를 제공하지 않을 수 있습니다.
print(tips['sex'].cat.categories)

# 'my_sq' 함수를 정의하여 숫자 'x'의 제곱을 계산합니다.
def my_sq(x):
    return x**2

# 'my_exp' 함수를 정의하여 'x'를 'n' 제곱 계산합니다.
def my_exp(x, n):
    return x**n

# 'a'와 'b' 두 열을 가진 DataFrame 'df'를 생성합니다.
df = pd.DataFrame({'a': [10, 20, 30], 'b': [2, 3, 2]})

# DataFrame 'df'를 출력합니다.
print(df)

# 'a' 열에 'my_sq' 함수를 .apply() 메서드를 사용하여 적용합니다.
# 이로써 'a' 열의 각 값을 제곱합니다.
print(df.a.apply(my_sq))

# 'a' 열에 'my_exp' 함수를 적용하며 'n'을 3으로 설정합니다.
# 이로써 'a' 열의 각 값을 3제곱 계산합니다.
print(df.a.apply(my_exp, n=3))

# DataFrame의 각 행에 대해 행 데이터를 출력하고 첫 번째 원소를 두 번째 원소의 거듭제곱 결과로 반환하는 'df_print' 함수를 정의합니다.
def df_print(x):
    print(x)
    return x[0] ** x[1]

# .apply() 메서드를 사용하여 'df_print' 함수를 DataFrame 'df'의 각 행에 적용합니다.
# 'axis=1'을 사용하여 각 행에 함수를 적용하며 각 행은 Series로 처리됩니다.
# 결과적으로 각 행을 출력하고 각 행에 대한 함수 결과를 반환합니다.
df.apply(df_print, axis=1)
import pandas as pd

# 2013년 9월부터 2015년 8월까지의 평균 분양가격 데이터를 불러옵니다.
first_df = pd.read_csv('data/전국_평균_분양가격(2013년_9월부터_2015년_8월까지).csv', encoding='cp949')

# 2019년 12월의 평균 분양가격 데이터를 불러옵니다.
last_df = pd.read_csv('data/주택도시보증공사_전국_평균_분양가격(2019년_12월).csv', encoding='cp949')

# 결측값(누락된 값)의 수를 출력합니다.
print(last_df.isna().sum())

# '분양가격(㎡)' 열에서 문제가 있는 값을 NaN(누락된 값)으로 변경합니다.
last_df['분양가격'] = pd.to_numeric(last_df['분양가격(㎡)'], errors='coerce')

# '분양가격' 열의 데이터 유형을 확인합니다.
print(last_df['분양가격'].dtype)

# '분양가격' 열의 합을 출력합니다.
print(last_df['분양가격'].sum(skipna=True))

# '분양가격' 열의 평균을 출력합니다.
print(last_df['분양가격'].mean(skipna=True))

# '평당분양가격' 열을 생성하고, 값에 3.3을 곱합니다.
last_df['평당분양가격'] = last_df['분양가격'] * 3.3

# '규모구분' 열의 고유한 값 개수를 출력합니다.
print(last_df['규모구분'].nunique())

# '규모구분' 열의 고유한 값 목록을 출력합니다.
print(last_df['규모구분'].unique())

# '전용면적' 열을 정리하고, 필요한 문자를 제거합니다.
last_df['전용면적'] = last_df['규모구분'].str.replace('이하', '').str.replace('초과', '~').str.replace('전용면적', '').str.replace(' ', '').str.strip()

# '규모구분' 및 '분양가격(㎡)' 열을 삭제합니다.
last_df.drop(columns=['규모구분', '분양가격(㎡)'], inplace=True)

# '지역명'을 기준으로 '평당분양가격'의 평균을 계산합니다.
last_df.groupby(['지역명'])['평당분양가격'].mean()

# '전용면적'이 '전체'인 경우를 필터링하고 '지역명'을 기준으로 '평당분양가격'의 평균을 계산합니다.
last_df[last_df['전용면적'] == '전체'].groupby(['지역명'])['평당분양가격'].mean()

# '전용면적'을 기준으로 '평당분양가격'의 평균을 계산합니다.
last_df.groupby(['전용면적'])['평당분양가격'].mean()

# '전용면적'과 '지역명'을 기준으로 '평당분양가격'의 평균을 계산합니다.
last_df.groupby(['전용면적', '지역명'])['평당분양가격'].mean()

# '지역명'과 '전용면적'을 기준으로 '평당분양가격'의 평균을 계산하고, 표 형태로 변환합니다.
last_df.groupby(['지역명', '전용면적'])['평당분양가격'].mean()

# '지역명'과 '전용면적'을 기준으로 '평당분양가격'의 평균을 계산하고, 표를 둥근 십의 자리로 반올림합니다.
last_df.groupby(['지역명', '전용면적'])['평당분양가격'].mean().unstack().round(-1)

# '연도'와 '전용면적'을 기준으로 '평당분양가격'의 평균을 계산하고, 표를 전치합니다.
last_df.groupby(['연도', '전용면적'])['평당분양가격'].mean().unstack().T

# '지역명'을 기준으로 '평당분양가격'의 평균을 계산하고 표로 표시합니다.
last_df.pivot_table(index=['지역명'], values=['평당분양가격'], aggfunc='mean')

# '전용면적'을 기준으로 '평당분양가격'의 평균을 계산하고 반올림합니다.
last_df.pivot_table(index=['전용면적'], values=['평당분양가격'], aggfunc='mean').round()

# '전용면적'과 '지역명'을 기준으로 '평당분양가격'의 평균을 계산하고 표로 표시합니다.
last_df.pivot_table(index=['전용면적', '지역명'], values=['평당분양가격'], aggfunc='mean').round()

# '전용면적'과 '지역명'을 기준으로 '평당분양가격'의 평균을 계산하고 표로 표시합니다.
last_df.pivot_table(index=['전용면적', '지역명'], values=['평당분양가격'], aggfunc='mean').round().reset_index()

# '전용면적'을 기준으로 '지역명'을 열로, '평당분양가격'을 값으로 하는 표를 만들고 반올림합니다.
last_df.pivot_table(index=['전용면적'], columns=['지역명'], values=['평당분양가격'], aggfunc='mean').round()

# '전용면적'을 기준으로 '지역명'을 열로, '평당분양가격'을 값으로 하는 표를 만들고 반올림합니다.
last_df.pivot_table(index=['전용면적'], columns=['지역명'], values=['평당분양가격'], aggfunc='mean').round().reset_index()

# '연도'와 '지역명'을 기준으로 '평당분양가격'의 평균을 계산하고 표로 표시합니다.
last_df.pivot_table(index=['연도', '지역명'], values=['평당분양가격'], aggfunc='mean').round().reset_index()

# '연도'와 '지역명'을 기준으로 '평당분양가격'의 평균을 계산하고 표로 표시합니다. 그리고 '연도'를 인덱스로 설정합니다.
last_df.pivot_table(index=['연도', '지역명'], values=['평당분양가격'], aggfunc='mean').round().reset_index().set_index('연도')

# '지역명'을 기준으로 '평당분양가격'의 평균을 계산하고, 결과를 내림차순으로 정렬하여 DataFrame으로 저장합니다.
g = last_df.groupby(['지역명'])['평당분양가격'].mean().sort_values(ascending=False).to_frame()

# 선 그래프로 평당분양가격을 시각화합니다.
g.plot()

# 막대 그래프로 평당분양가격을 시각화합니다.
g.plot(kind='bar')

# 원 그래프로 평당분양가격을 시각화합니다.
g['평당분양가격'].plot(kind='pie')

# '지역명'과 '연도'를 기준으로 피벗 테이블을 생성합니다.
g = last_df.pivot_table(index='지역명', columns='연도', values='평당분양가격')

# 상자 그림 (박스 플롯)으로 데이터 분포를 시각화합니다.
g.plot(kind='box')

# '지역'을 기준으로 피벗 테이블을 생성합니다.
first_df_melt = first_df.melt(id_vars=['지역'], var_name='기간', value_name='평당분양가격')

# '기간' 열을 나누어 '연도'와 '월' 열을 생성합니다.
first_df_melt['연도'] = first_df_melt['기간'].str.split('년').str[0].astype(int)
first_df_melt['월'] = first_df_melt['기간'].str.split('년').str[1].str.replace('월', '').astype(int)

# '전용면적'이 '전체'인 경우를 필터링하고 '지역명'을 '지역'으로 변경합니다.
last_df_result = last_df[last_df['전용면적'] == '전체']
last_df_result.rename(columns={'지역명': '지역'}, inplace=True)

# 경고 메시지를 무시합니다.
import warnings
warnings.filterwarnings('ignore')

# 필터링된 데이터프레임을 '지역' 및 '연도'를 기준으로 선택하여 '평당분양가격' 데이터를 추출합니다.
first = first_df_melt[['지역', '연도', '월', '평당분양가격']]
last = last_df_result[['지역', '연도', '월', '평당분양가격']]

# 두 데이터프레임을 연결하여 하나의 데이터프레임으로 만듭니다.
df = pd.concat([first, last])