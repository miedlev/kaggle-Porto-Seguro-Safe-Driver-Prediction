#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.style.use("ggplot")

import matplotlib
matplotlib.rc("font", family = "AppleGothic")
matplotlib.rc("axes", unicode_minus = False)

from IPython.display import set_matplotlib_formats
set_matplotlib_formats("retina")

import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as os

from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.options.display.max_columns = 5000
pd.options.display.max_rows = 1000


# In[2]:


train = pd.read_csv("Desktop/phthon/Kaggle/porto/train.csv", index_col = "id")
print(train.shape)
train.head()


# In[3]:


test = pd.read_csv("Desktop/phthon/Kaggle/porto/test.csv", index_col = "id")
print(test.shape)
test.head()


# ## Pre - Preprocessing

# ### 1. columns check

# In[4]:


print(len(train.columns))
print(len(test.columns))
train.columns, test.columns


# In this competition, you will predict the probability that an auto insurance policy holder files a claim.
# 
# In the train and test data, features that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features. Features without these designations are either continuous or ordinal. Values of -1 indicate that the feature was missing from the observation. The target columns signifies whether or not a claim was filed for that policy holder.

# ### 2. Finding Data_type

# In[5]:


# front = ind, reg, car, calc
# last = cat, bin, continuous, ordinal

# columns management
# make role, level, dtype
hyperparameter = []

for columns in train.columns:
    
    if "target" in columns:
        role = "target"
    else:
        role = "input"

    if train[columns].dtype == float:
        level = "continuous"
    if train[columns].dtype == int:
        level = "ordinal"
    if "bin" in columns or columns == "target":
        level = "binary"
    if "cat" in columns or columns == "id":
        level = "categorical"

    dtype = train[columns].dtype

    dictionary = {"name" : columns, "role" : role, "level" : level, "dtype" : dtype}
    hyperparameter.append(dictionary)


data = pd.DataFrame(hyperparameter).set_index("name")
pd.pivot_table(index = ["role", "level"], data = data, aggfunc = len)


# ### 3. Missing Data

# In[6]:


# 이번에는 missing data를 구해본다
# 여기서 중요한 것은 데이터 미스가 얼마나 많이 났느냐 이다 이걸 파악한담에 어떻게 처리를 할지 파악해야 한다

hyperparameter = []
for i in train.columns:
    
    counts = train[train[i] == -1][i].count()    
    
    if counts == 0 :
        calculation = 0      
    if counts > 0:
        calculation = round(counts / train.shape[0], 2)
        
    data_miss = {"name" : i, "counts" : counts, "calculation" : calculation}
    hyperparameter.append(data_miss)
    
pd.DataFrame(hyperparameter)[pd.DataFrame(hyperparameter)["counts"] > 0].sort_values(by = "counts", 
                                                                                     ascending = False)


# <위의 missing data수를 파악해보면 다음과 같다>
# 
# 1. ps_car_03_cat, ps_car_05_cat는 결측치가 매우 많다. 따라서 관련 컬럼 데이터는 머신러닝을 사용하는데 좋은 결과를 주지 않을 것이기 때문에 사용하지 않는다.
# 
# 2. ps_car_11 / ps_car_12 / ps_reg_03 / ps_car_14, 관련 4 컬럼은 continuous value이기 때문에 평균으로 대체할 수 있다.
# 
# 3. 이외 나머지 categorical value는 범주형이기 때문에 일단은 놔둔 상태로 진행하고 추후에 관련 업무를 분석할 때 사용하도록 한다

# #### ps_car_11, ps_car_12, ps_reg_03, ps_car_14

# In[7]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = -1, strategy = "mean")

# train
train["ps_car_12"] = imputer.fit_transform(train[["ps_car_12"]])
# train["ps_car_11"] = imputer.fit_transform(train[["ps_car_11"]])
train["ps_reg_03"] = imputer.fit_transform(train[["ps_reg_03"]])
train["ps_car_14"] = imputer.fit_transform(train[["ps_car_14"]])

# train
test["ps_car_12"] = imputer.fit_transform(test[["ps_car_12"]])
# test["ps_car_11"] = imputer.fit_transform(test[["ps_car_11"]])
test["ps_reg_03"] = imputer.fit_transform(test[["ps_reg_03"]])
test["ps_car_14"] = imputer.fit_transform(test[["ps_car_14"]])


# final sum for -1(np.nan)
print(train[["ps_car_14", "ps_reg_03", "ps_car_11", "ps_car_12"]].isnull().sum())
print(test[["ps_car_14", "ps_reg_03", "ps_car_11", "ps_car_12"]].isnull().sum())


# ### 4. VarianceThreshold

# In[8]:


# 특징데이터의 값 자체가 표본에 따라 그다지 변하지 않는다면 종속데이터 예측에도 도움이 되지 않을 가능성이 높다. 
# 따라서 표본 변화에 따른 데이터 값의 변화 즉, 분산이 기준치보다 낮은 특징 데이터는 사용하지 않는 방법이 
# 분산에 의한 선택 방법(VarianceThreshold)이다.

from sklearn.feature_selection import VarianceThreshold
vector = np.vectorize(lambda x : not x)

# 임계치를 0.1보다 작은 것을 추출해본다 / 작은것은 false, 큰것은 ture로 나옴
variance = VarianceThreshold(threshold = 0.1)
variance.fit(train)

# 0.1이하가 false로 나오기 때문에 변환작업을 해서 true로 바꿔주면 된다
train.columns[vector(variance.get_support())].tolist()


# **<위의 VarianceThreshold를 파악해보면 다음과 같다>**
# 
# 보통 분산이 적은것은 그 변수가 적을 경우가 많을 때 생기는 것이다. 따라서 binary와 ordinal같은 것은 상대적으로 분산이 적을 가능성이 높다 다시말해 continuous는 분산이 클 가능성이 높다는 말인데 위 결과 continuous 상당수가 분산이 매우 작다는 것을 파악할 수 있다
# 
# -> 따라서 continuous를 분석할 때 한쪽에 치우치거나 몰려있는 부분이 없는지를 파악해야 할 필요성이 있다

# ## EDA(Explotory Data Analysis)¶

# ### 1. Binary

# #### 1) Visualization Analysis - [Absolute number of each columns]

# In[9]:


# 데이터를 활용하여 binary컬럼을 가지고 와 시각화시켜본다

level = data["level"] == "binary"
role = data["role"] == "input"
binary_columns = data[level & role].index

# data visualization
fig = plt.figure(figsize = [40,50])
sns.set(font_scale = 2)

for i, col in zip(np.arange(1, len(binary_columns) + 1), binary_columns): 
    
    ax1 = fig.add_subplot(6, 3, i)
    sns.countplot(x = col, data = train)
    plt.xlabel(col)


# **<위의 그래프를 파악해보면 다음과 같다>**
# 
# binary에서 중요한 것은 '어느 한쪽으로 쏠려있는가?' 혹은 '컬럼들 간 다른 방향성을 보이는 부분이 있는가?' 두가지이다. 이 두가지가 컬럼의 중요성 및 차이점을 만든다는 가정하에 분석한다
# 
# 'ps_ind_10_bin', 'ps_ind_11_bin' , 'ps_ind_12_bin', 'ps_ind_13_bin'
# -> 대부분의 그래프는 0>1을 보이고 있는데 그 차이가 크다면 그 컬럼은 머신러닝 모델 점수 향상에 도움이 될 가능성이 크다. 특히 4개 컬럼의 경우 대부분 0을 보여주고 있는 유의한 데이터라고 볼 수 있다.
# 
# 'ps_ind_16_bin', 'ps_calc_16_bin', 'ps_calc_17_bin'
# -> 관련 컬럼 3개는 다른 데이터들과 달리 1이 0보다 더 많은 경향을 보이고 있다. 즉 이 컬럼의 거리는 보험금을 청구할 가능성이 매우 높은 곳이라는 점에서 관련 데이터를 활용해야할 필요성이 있다. 심화적으로 들어간다면 관련 거리에 1로 결과를 나오게 하는 무언가가 존재한다는 것이고 이는 특정 차량 종류에 영향을 미칠 수도 있는 것이고, 낮과 밤, 성별 등등 기타 이유로 인해 0과 1로 갈리게 될 수도 있다는 점에서 세부적인 분석이 필요한 곳이라 보인다.
# 
# 나머지 부분들
# -> 1번처럼 0이 1보다 대부분 더 큰 모습을 보이고 있다. 물론 'ps_ind_06_bin'의 경우 1의 숫자도 매우 높기 때문에 관련 컬럼을 배제하고 머신러닝을 진행하면 점수가 내려갈 가능성이 높다.

# #### 2) Visualization Analysis - [Comparation by using "target"]

# In[10]:


fig = plt.figure(figsize = [40,50])
sns.set(font_scale = 2)

for i, col in zip(np.arange(1, len(binary_columns) + 1), binary_columns): 
    
    ax1 = fig.add_subplot(6, 3, i)
    sns.countplot(x = col, data = train, hue = "target")
    plt.xlabel(col)


# **<위의 그래프를 파악해보면 다음과 같다>**
# 
# 만약 각 column에 0,1 여부에 따라 target의 변동성이 있다면 그 column에 주목해야 한다. 분석 결과 절대적인 숫자에서는 컬럼별로 0, 1 자체가 차이가 있었지만, target을 기준으로 다시 분석을 해본 결과 0,1의 숫자와 상관없이 모두 target = 0인 부분이 대부분임을 파악할 수 있다.
# 
# **-> binary columns는 target이 0, 1이냐를 파악하는 것보다는 그 컬럼 자체의 숫자가 0, 1이 어디쪽에 분포가 되어있는지를 파악하는 것이 key point라는 점이 파악된다**
# 
# **-> 보통 binary 관련 destination은 0을 보여주고 이는 나름대로 안정적인 곳이라 할수 있지만 주요 몇곳은 보험금확률이 매우 높게 나오는 곳이라는 점을 파악해야 할 것으로 보인다**

# #### 3) Correlation Analysis

# In[11]:


# 1번을 부가적으로 정리하기 위해 상관관계분석을 적용해본다 
# train[binary_columns].corr()

train[['ps_ind_16_bin', 'ps_ind_17_bin',        'ps_ind_18_bin', 'ps_calc_15_bin',       'ps_calc_16_bin', 'ps_calc_17_bin']].corr()


# 'ps_ind_16_bin'과 'ps_ind_17_bin','ps_ind_18_bin' 관련 3개 항목은 예상했던데로 상관관계가 -0.5이상을 보이고 있었다. 이는 음의 상관관계로 어느정도 반대의 방향을 보이고 있다는 사실을 의미한다.
# 
# **관련해서 이들 간에 관계성을 분석해서 파악을 해본다면 0, 1을 퍼센트를 어느정도 조정해볼 수 있다는 생각이 든다**
# 
# 반대로 'ps_calc_15_bin'과 'ps_calc_16_bin', 'ps_calc_17_bin'은 큰 상관관계 차이가 보이지는 않았다. 이를 통해 이들의 0,1 비율이 차이는 보이지만 그 관계성은 크지 않다는것을 의미한다.

# #### 4) Density Analysis

# In[12]:


target_0 = train[train["target"] == 0]
target_1 = train[train["target"] == 1]

print(train.shape)
print(target_0.shape)
print(target_1.shape)


fig = plt.figure(figsize = [35,50])

for i, col in zip(np.arange(1, len(binary_columns) + 1), binary_columns): 
    
    ax1 = fig.add_subplot(6, 3, i)
    sns.kdeplot(target_0[col], bw = 0.5, label = "target = 0")
    sns.kdeplot(target_1[col], bw = 0.5, label = "target = 1")
    plt.xlabel(col)


# "ps_ind_06_bin", "ps_ind_07_bin", "ps_ind_16_bin", "ps_ind_17_bin"
# 
# -> 관련 4개 columns는 target간에 밀도가 imblance함을 알 수 있다. 이러한 컬럼이 많이 있다면 데이터의 불균형으로 좋지 않은 결과가 나올 수도 있음에 유의해야 한다. 점수가 잘 나오지 않는다면 컬럼을 삭제하는 방법도 고려해야 한다. 다만 저정도의 차이가 큰 변수를 만들지 못한다는 점에서 그대로 가지고 가는 방향으로 한다
# 
# 

# ### 2.categorical

# #### 1) Visualization Analysis - [Absolute number of each columns]

# In[13]:


level2 = data["level"] == "categorical"
role = data["role"] == "input"
categorical = data[level2 & role].index

# ps_car_03_cat, ps_car_05_cat 제거
categorical_fix = categorical.difference(['ps_car_03_cat', 'ps_car_05_cat'])

fig = plt.figure(figsize = [30,40])
sns.set(font_scale = 2)

for i, col in enumerate(categorical_fix):
    
    ax1 = fig.add_subplot(4,3,i+1)
    sns.countplot(x = col, data= train)
    plt.xlabel(col)


# 위 그래프를 통해서는 1. categorical 간 동일한 숫자가 배열되어 있지는 않다는 사실, 2. 산별적으로 흩어져 있는 관계로 그래프 상으로는 의미있는 insight를 발견하기 어렵다는 점이다
# 
# 따라서 관련해서는 각 컬럼별로 숫자 어느쪽에 무게중심이 있는지 파악하고 / get_dummies를 통해 categical를 독립형으로 만들어준다

# #### 2) Density Analysis

# In[14]:


fig = plt.figure(figsize = [35,50])

for i, col in zip(np.arange(1, len(categorical_fix) + 1), categorical_fix): 
    
    ax1 = fig.add_subplot(6, 3, i)
    sns.kdeplot(target_0[col], bw = 0.5, label = "target = 0")
    sns.kdeplot(target_1[col], bw = 0.5, label = "target = 1")
    plt.xlabel(col)


# In[15]:


number_unique = [len(train[col].unique()) for col in categorical_fix]

pd.DataFrame(number_unique, categorical_fix).sort_values(by = 0, ascending = False)


# **<위의 그래프를 파악해보면 다음과 같다>**
# 
# 1.일단 binaary에 비해서 target간 밀도의 차이가 조금씩 발생한다. 보통 categorical은 그 변수가 많기 때문에 binary보다 밀도차이가 발생하기 마련이다. 따라서 그런 것을 감안하고 본다면 예외변수를 만들만한 차이는 발생하지 않는다고 추측해볼 수 있다.
# 
# 2.주목해야 할점은 폭이다. target이 0인 경우가 1인 경우보다 매우 많다는 점에서 0인 그래프의 꼭대기(평균)지점의 위치는 1보다는 높은 경우가 많다. 전 columns를 보면 숫자가 낮을수록 0그래프가 위쪽에 있으나 숫자가 커질수록 1의 그래프가 위에 있는 경우가 많다는 점을 파악할 수 있다. 이는 columns별로 작은 숫자일 수록 0에 많이 몰려 있다는 점. 그리고 만약 점수를 높이기 위해서는 각 컬럼의 변수에서 중심에서 벗어난 부분들을 제거해주는 것이 좋다는 점을 파악할 수 있다
# 
# 3.categorical별로 숫자가 다르다는 점은 각각을 계산해서 구해야 함을 말한다. categorical이기 때문에 -1은 제거를 하는것이 옳다

# #### 3) Get_dummies

# In[16]:


# categorical을 일일이 분할하여 더미변수로 전환시킨다

train_copy = train.copy()
test_copy = test.copy()

# train
for i in categorical_fix:
    
    frist = pd.get_dummies(train_copy[i], prefix = i + "_")  
    train_copy = pd.concat([train_copy, frist], axis = 1)
    
# test
for y in categorical_fix:
    
    frist = pd.get_dummies(test_copy[y], prefix = y + "_") 
    test_copy = pd.concat([test_copy, frist], axis = 1)


print(train_copy.shape)
print(train.shape)
print(test_copy.shape)
print(test.shape)


# In[17]:



train_category = train_copy.drop(['target', 'ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03', 'ps_ind_04_cat',
                                 'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin',
                                 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
                                 'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15', 'ps_ind_16_bin',
                                 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03',
                                 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat',
                                 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat',
                                 'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11',
                                 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_calc_01',
                                 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06',
                                 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11',
                                 'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin',
                                 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin',
                                 'ps_calc_20_bin'], axis = 1)

test_category  = test_copy.drop(['ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03', 'ps_ind_04_cat',
                                 'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin',
                                 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
                                 'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15', 'ps_ind_16_bin',
                                 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03',
                                 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat',
                                 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat',
                                 'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11',
                                 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_calc_01',
                                 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06',
                                 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11',
                                 'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin',
                                 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin',
                                 'ps_calc_20_bin'], axis = 1)

# 2. -1인 것들은 결측치이기 때문에 제거해준다

train_category = train_category.drop(['ps_car_01_cat__-1', 'ps_car_02_cat__-1', 
                                      'ps_car_07_cat__-1', 'ps_car_09_cat__-1',
                                      'ps_ind_02_cat__-1', 'ps_ind_04_cat__-1', 
                                      'ps_ind_05_cat__-1'], axis = 1)

test_category = test_category.drop(['ps_car_01_cat__-1', 'ps_car_02_cat__-1', 
                                    'ps_car_07_cat__-1', 'ps_car_09_cat__-1',
                                    'ps_ind_02_cat__-1', 'ps_ind_04_cat__-1', 
                                    'ps_ind_05_cat__-1'], axis = 1)

# 후에 feature_name 작성시 관련 컬럼 활용해서 csr_matrix, hstack 활용한다
print(train_category.shape)
print(test_category.shape)


# ### 3. continuous

# In[18]:


level3 = data["level"] == "continuous"
role = data["role"] == "input"
continuous = data[level3 & role].index
continuous


# 본래 regression 및 continuous부분에서는 scatterplot을 많이 쓰지만 종속변수(결과변수)가 0,1인 현 데이터에서는 사실 scatterplot이 그렇게 좋은 것은 아니다. 하지만 대다수가 0인 데이터에서 1위 위치를 대략적으로 파악해본다면, insight를 얻을 수 있다고 보고 진행한다
# 
# -> scatterplot을 개별적으로 구하는것 보다는 각각의 공통된 가운데 글자를 소유한 컬럼끼리 묶어본다. float에서 middle name이 같은건 그만한 이유가 있다는 가정을 해본다
# 
# 

# #### 1) Scatterplot Analysis

# In[19]:


# ps_reg 

px.scatter_matrix(train, dimensions = ['ps_reg_01', 'ps_reg_02', 'ps_reg_03', "target"],
                 color = "target")


# In[20]:


# ps_calc
px.scatter_matrix(train, 
                  dimensions = ['ps_calc_01', 'ps_calc_02', 'ps_calc_03', "target"],
                  color = "target")


# **<위의 그래프를 파악해보면 다음과 같다>**
# 
# 개별적으로 scatterplot으로 분석을 해보면 유의한 결과가 나오지 않지만, 다른 변수들과 같이 비교를 했을 때는 어느정도 괜찮은 결과가 나옴을 알수 있다. target이 1인 지점의 위치를 주목해본다
# 
# **-> ps_reg를 보면 각각의 변수에 대해 특정 위치에 1점이 강하게 보인다. 이 의미는 'ps_reg_01', 'ps_reg_02', 'ps_reg_03'간에 각각의 특정 지점과 위치에서 target이 1이 나온다는 것을 의미한다. 이는 별도의 column 같지만 들여다보면 어느정도 상관관계가 있을 수 있다는 점을 의미한다. 관련해서 좀더 세부적인 정보로 파악을 해본다면 target의 정확성을 높일 수 있고 그 말은 관련 데이터는 매우 중요하기 때문에 머신러닝 활용에 써야함을 의미한다**
# 
# **-> 'ps_calc_01'과 'ps_calc_02', 'ps_calc_03'간에 어느정도 연관성이 있어 보인다. 특정 포인트에 1이 몰려있음을 파악할 수 있음으로 관련 데이터에 활용한다**

# #### 2) density Analysis

# In[21]:


target_0 = train[train["target"] == 0]
target_1 = train[train["target"] == 1]

fig = plt.figure(figsize = [35,50])
sns.set(font_scale = 2)
for i, col in zip(np.arange(1, len(continuous) + 1), continuous): 
    
    ax1 = fig.add_subplot(6, 3, i)
    sns.kdeplot(target_0[col], bw = 0.5, label = "target = 0")
    sns.kdeplot(target_1[col], bw = 0.5, label = "target = 1")
    plt.xlabel(col)


# continuous의 경우 데이터의 imblance가 어느정도 있을 수 있다는 점을 감안하고, 각 데이터별로 어느쪽에 몰려 있는지 파악해보는 것이 중요하다
# 
# 위의 scatterplot와 달리 개별적인 column으로 접근을 하는 것인데 'ps_reg_01'과 'ps_car_15'에 주목해본다. target이 1인 부분이 다른 column과 달리 평균에 더 많이 몰려있다는 점이 중요한데, 이는 관련 continuous value number에서 1이 발생할 가능성이 크다는 것. 그렇기 때문에 이 데이터를 활용하면 그 number의 활용성을 높일 수 있다

# ### 4. Ordinal

# ordinal은 순서이기 때문에 일종의 categorical value인데 관련 숫자안에서 의미가 있을 수 있다는 가정하에 target을 기준점으로 나누어 각 column별로 어느정도 포지션을 취하고 있는지 파악해본다
# 
# lineplot은 원래 연속성을 위해 쓰는것이지만, 본 그래프에서는 시각적 효과로 인해 활용해본다

# In[22]:


level4 = data["level"] == "ordinal"
role = data["role"] == "input"
ordinal = data[level4 & role].index


# #### 1) lineplot - all data

# In[23]:


for i, col in enumerate(ordinal):
    
    train.groupby([col, "target"])["target"].count().unstack().plot(figsize = [10,3])


# 분석 결과 일단 1의 표본이 매우 작기 때문에 그래프로는 구체적인 차이점을 발견하기 어렵다.
# 
# 따라서 taget이 0, 1인 것을 각각 나눈뒤에 개별적으로 관련 사항을 분석해본다. 그래프가 같은 방향성을 보이면 column에 큰 차이가 있는 것이 아니지만, 만약 다른 방향성을 보인다면 그것에서 insight를 얻을 수 있다

# #### 2) lineplot - each data by using "target"

# In[24]:


# target = 0 
target_0 = train[train["target"] == 0]

fig = plt.figure(figsize = [25,25])
sns.set(font_scale = 2)
for i, col in zip(np.arange(1, len(ordinal) + 1), ordinal): 
    
    ax1 = fig.add_subplot(4, 4, i)
    sns.countplot(x = col, data = target_0)
    plt.xlabel(col)


# In[25]:


# terget = 1
target_1 = train[train["target"] == 1]

fig = plt.figure(figsize = [25,25])
sns.set(font_scale = 2)
for i, col in zip(np.arange(1, len(ordinal) + 1), ordinal): 
    
    ax1 = fig.add_subplot(4, 4, i)
    sns.countplot(x = col, data = target_1)
    plt.xlabel(col)


# **<관련 그래프에서 몇가지 사실을 알 수 있다>**
# 
# -> 다른 column은 대부분 비슷한 방향이 0, 1사이에서 보이고 있다. 이를 통해 관련 column에서 순서의 중요성이 상당히 있다는 것을 알 수 있다. 관련 데이터를 대부분 활용해준다면 결과물에 좋은 경향을 보일 것을 보인다
# 
# -> "ps_ind_03" column은 다른 것과 달리 다른 그래프를 보인다는 사실에 주목해야 한다. 5,6,7부근에서 target이 1인 경우의 수가 더 target이 0인 경우보다 더 확률적으로 높다는 것을 의미하는데 관련 부근을 파악해본다면 유의한 차이를 파악해볼 수 있을 것으로 보인다
# 
# -> 'ps_car_11'에서 target이 1인 것에는 결측치(-1)가 없다. 이는 -1로 측정이 안된 것들은 전부 target이 0인 쪽으로 보내졌다는 것을 의미한다. 그래서 관련 column은 1)데이터의 효용성이 낮거나 2)다른 column의 target을 결정하는데 더 큰 영향을 줄 수 있다는 것을 의미한다고 본다

# In[26]:


train_fix = train.copy()

for i in ordinal:
    frist = pd.get_dummies(train_fix[i], prefix = i + "_")
    train_fix = pd.concat([train_fix, frist], axis = 1)
    
    
train_fix = train_fix.drop(['target', 'ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03', 'ps_ind_04_cat',
                            'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin',
                            'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
                            'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15', 'ps_ind_16_bin',
                            'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03',
                            'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat',
                            'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat',
                            'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11',
                            'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_calc_01',
                            'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06',
                            'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11',
                            'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin',
                            'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin',
                            'ps_calc_20_bin', 'ps_car_11__-1'], axis = 1)

train_number = train_fix.columns
len(train_fix.columns)


# In[27]:


test_fix = test.copy()

for i in ordinal:
    frist = pd.get_dummies(test_fix[i], prefix = i + "_")
    test_fix = pd.concat([test_fix, frist], axis = 1)

    
test_fix  =  test_fix.drop(['ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03', 'ps_ind_04_cat',
                            'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin',
                            'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
                            'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15', 'ps_ind_16_bin',
                            'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03',
                            'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat',
                            'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat',
                            'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11',
                            'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_calc_01',
                            'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06',
                            'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11',
                            'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin',
                            'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin',
                            'ps_calc_20_bin', 'ps_car_11__-1'], axis = 1)

test_number = test_fix.columns
len(test_fix.columns)


# In[28]:


# 데이터가 겹치지 않는 부분이 있는지 체크를 해본다 

test_number = set(test_number)
train_number = set(train_number)

# 각각 다음이 개별적으로 있다
print(test_number - train_number)
print(train_number - test_number)

# 데이터 갯수를 분석해본 결과 그 수가 매우 적다. 따라서 관련 데이터는 제거하고 진행하도록 한다 
print(len(test[test["ps_calc_08"] == 1]))
print(len(test[test["ps_calc_12"] == 11]))
print(len(test[test["ps_calc_13"] == 15]))
print(len(test[test["ps_calc_11"] == 20]))
print(len(test[test["ps_calc_14"] == 28]))
print(len(test[test["ps_calc_13"] == 14]))
print(len(train[train["ps_calc_12"] == 10]))
print(len(train[train["ps_calc_06"] == 0]))


# In[29]:


test_fix = test_fix.drop(['ps_calc_08__1','ps_calc_11__20','ps_calc_12__11','ps_calc_13__14',
                          'ps_calc_13__15','ps_calc_14__28'], axis = 1)

train_fix = train_fix.drop(['ps_calc_12__10','ps_calc_06__0'], axis = 1)

print(len(test_fix.columns))
print(len(train_fix.columns))


# ## precessing

# ### 1. Data Total Concat

# In[30]:


# feature_name만들기 전에 먼저 데이터를 하나로 합친다
# 새로 만든 category, ordinal 데이터 concat
train_sum_makings = pd.concat([train_category, train_fix], axis = 1)
test_sum_makings = pd.concat([test_category, test_fix], axis = 1)

# 위에서 만든 sum_making과 기존 데이터 concat
train_final = pd.concat([train, train_sum_makings], axis = 1)
test_final = pd.concat([test, test_sum_makings], axis = 1)

print(train_sum_makings.shape)
print(test_sum_makings.shape)


# ### 2. Feature_name / label_name making

# In[31]:


# 위에서 만든 feature중에서 더미변수로 생성하지 않은 부분부터 합친다
feature_names = list(binary_columns)
feature_names = feature_names + list(continuous)
# feature_names += list(train_category.columns)
# feature_names += list(train_fix.columns)

# label_name 생성
label_name = "target"

# select data for modeling
x_train = train[feature_names]
x_test = test[feature_names]
y_train = train[label_name]

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)


# ### 3. sparse_matrix

# dummy feature가 많을 경우에 sparse_matrix를 많이 씀으로 활용한다

# In[32]:


from scipy import sparse
from scipy.sparse import csr_matrix, hstack

# dummy 변수를 적용
train_sum_making = csr_matrix(train_sum_makings)
test_sum_making = csr_matrix(test_sum_makings)

x_train = hstack([x_train.astype("float"), train_sum_making])
x_test = hstack([x_test.astype("float"), test_sum_making])

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)


# ### 4. Selection of modeling

# In[33]:


# 모델링을 총 4가지를 해봐서 가장 점수가 높은 것으로 진행을 한다

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# In[34]:


model_randomforst = RandomForestClassifier()
model_randomforst


# In[35]:


model_gradient = LGBMClassifier()
model_gradient


# In[36]:


model_catboost = CatBoostClassifier()
model_catboost


# ### 5. make_scorer - cross_validation

# holdout_validation과 cross_validation 둘 중 하나 중에 make_score를 통한 지니계수의 점수를 구하는 것이기 때문에 holdout_validation을 쓴다

# In[37]:


from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, train_test_split

# make score
def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype = np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)

# make score finish
score = make_scorer(gini_normalized)
score


# In[38]:


x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(x_train, 
                                                            y_train, 
                                                            test_size = 0.4, 
                                                            random_state = 37)

print(x_train_t.shape)
print(y_train_t.shape)
print(x_test_t.shape)
print(y_test_t.shape)


# In[39]:


model = CatBoostClassifier(#n_estimators = 2000,
                           iterations = 5000,
                           learning_rate = 0.01, 
                           loss_function = "MultiClass",
                           one_hot_max_size = 5,
                           # eval_metric='AUC',
                           # task_type = "CPU", 
                           # random_seed = 1234, 
                           verbose = True)

model


# In[40]:


model.fit(x_train, y_train)


# In[41]:


prediction = model.predict(x_test)
prediction

