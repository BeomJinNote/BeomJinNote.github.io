---
layout: single
title: "로지스틱회귀 ."
---


# 과제1

## 조기 종료를 사용한 배치 경사 하강법으로 로지스틱 회귀를 구현하라. 단 사이킷런을 전혀 사용하지 않아야 한다.


```python
## 데이터 준비
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
X = iris["data"][:, 3:]                   # 1개의 특성(꽃잎 너비)만 사용
y = (iris["target"] == 2).astype(np.int)  # 버지니카(Virginica) 품종일 때 1(양성)

X_with_bias = np.c_[np.ones([len(X), 1]), X]
np.random.seed(2042)

## 데이터 분할
test_ratio = 0.2                                         # 테스트 세트 비율 = 20%
validation_ratio = 0.2                                   # 검증 세트 비율 = 20%
total_size = len(X_with_bias)                            # 전체 데이터셋 크기

test_size = int(total_size * test_ratio)                 # 테스트 세트 크기: 전체의 20%
validation_size = int(total_size * validation_ratio)     # 검증 세트 크기: 전체의 20%
train_size = total_size - test_size - validation_size    # 훈련 세트 크기: 전체의 60%

rnd_indices = np.random.permutation(total_size)

X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]

X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]

X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]

Y_train = np.reshape(y_train,(90,1))
y_valid = np.reshape(y_valid,(30,1))
```


```python
def logistic_sigmoid(x): # 시그모이드 함수 정의
    return 1 / (1 + np.exp(-x))

n_inputs = X_train.shape[1]           # 특성 수(n) + 1, 붓꽃의 경우: 특성 2개 + 1
```

로지스틱함수 구현


```python
def logistic_sigmoid(x): # 시그모이드 함수 정의
    return 1 / (1 + np.exp(-x))

n_inputs = X_train.shape[1]           # 특성 수(n) + 1, 붓꽃의 경우: 특성 2개 + 1
```

조기종료 + 배치 경사 하강법 + 로지스틱 회귀


```python
eta = 0.1
n_iterations = 10001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1           # 규제 하이퍼파라미터
best_loss = np.infty   # 최소 손실값 기억 변수

Theta = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    # 훈련 및 손실 계산
    logits = X_train.dot(Theta)
    Y_proba = logistic_sigmoid(logits)
    error = Y_proba - Y_train
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, 1]), alpha * Theta[1:]]
    Theta = Theta - eta * gradients

    # 검증 세트에 대한 손실 계산
    logits = X_valid.dot(Theta)
    Y_proba = logistic_sigmoid(logits)
    xentropy_loss2 = -1/m*(np.sum(y_valid * np.log(Y_proba + epsilon) + (1 - y_valid ) * np.log(1 - Y_proba + epsilon)))
    l2_loss = 1/2 * np.sum(np.square(Theta[1:]))
    loss = xentropy_loss2 + alpha * l2_loss

   
    # 500 에포크마다 검증 세트에 대한 손실 출력
    if iteration % 500 == 0:
        print(iteration, loss)
        
    # 에포크마다 최소 손실값 업데이트
    if loss < best_loss:
        best_loss = loss
    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료
        print(iteration - 1, best_loss)        # 종료되지 이전 에포크의 손실값 출력
        print(iteration, loss, "조기 종료!")
        break
```

    0 0.2463555980311719
    143 0.18575075571718722
    144 0.18575156557083228 조기 종료!



```python
logits = X_valid.dot(Theta)
Y_proba = logistic_sigmoid(logits)
Y_proba2 = np.where(Y_proba >= 0.5 ,1,0)
y_predict = Y_proba2

y_valid = y_valid.reshape(30, 1)

accuracy_score = np.mean(y_predict == y_valid)
accuracy_score
```




    0.9




```python
logits = X_valid.dot(Theta)
Y_proba = logistic_sigmoid(logits)
Y_proba2 = np.where(Y_proba >= 0.5 ,1,0)
y_predict = Y_proba2

y_valid = y_test.reshape(30, 1)

accuracy_score = np.mean(y_predict == y_test)
accuracy_score
```




    0.5888888888888889




```python
Theta
```




    array([[-1.34559323],
           [ 0.728139  ]])



# 과제2

## 과제 1에서 구현된 로지스틱 회귀 알고리즘에 일대다(OvR) 방식을 적용하여 붓꽃에 대한 다중 클래스 분류 알고리즘을 구현하라. 단 사이킷런을 전혀 사용하지 않아야 한다.


```python
## 데이터 준비
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris["data"][:, 3:]                   # 1개의 특성(꽃잎 너비)만 사용
y = (iris["target"]).astype(np.int)  # 버지니카(Virginica) 품종일 때 1(양성)

X_with_bias = np.c_[np.ones([len(X), 1]), X]
np.random.seed(2042)

## 데이터셋 분할
test_ratio = 0.2                                         # 테스트 세트 비율 = 20%
validation_ratio = 0.2                                   # 검증 세트 비율 = 20%
total_size = len(X_with_bias)                            # 전체 데이터셋 크기

test_size = int(total_size * test_ratio)                 # 테스트 세트 크기: 전체의 20%
validation_size = int(total_size * validation_ratio)     # 검증 세트 크기: 전체의 20%
train_size = total_size - test_size - validation_size    # 훈련 세트 크기: 전체의 60%

rnd_indices = np.random.permutation(total_size)

X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]

X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]

X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]


## 타깃 변환
def to_one_hot(y):
    n_classes = y.max() + 1                 # 클래스 수
    m = len(y)                              # 샘플 수
    Y_one_hot = np.zeros((m, n_classes))    # (샘플 수, 클래스 수) 0-벡터 생성
    Y_one_hot[np.arange(m), y] = 1          # 샘플 별로 해당 클래스의 값만 1로 변경. (넘파이 인덱싱 활용)
    return Y_one_hot

Y_train_one_hot = to_one_hot(y_train)
Y_valid_one_hot = to_one_hot(y_valid)
Y_test_one_hot = to_one_hot(y_test)

Setosa_train=Y_train_one_hot[:,0]
Versicolor_train = Y_train_one_hot[:,1]
Virginica_train = Y_train_one_hot[:,2]

Setosa_valid = Y_valid_one_hot[:,0]
Versicolor_valid = Y_valid_one_hot[:,1]
Virginica_valid = Y_valid_one_hot[:,2]

Setosa_test = Y_test_one_hot[:,0]
Versicolor_test = Y_test_one_hot[:,1]
Virginica_test = Y_test_one_hot[:,2]

Setosa_train = np.reshape(Setosa_train,(90,1))
Versicolor_train = np.reshape(Versicolor_train,(90,1))
Virginica_train = np.reshape(Virginica_train,(90,1))

Setosa_valid = np.reshape(Setosa_valid,(30,1))
Versicolor_valid = np.reshape(Versicolor_valid,(30,1))
Virginica_valid = np.reshape(Virginica_valid,(30,1))

Y_train = np.reshape(y_train,(90,1))
y_valid = np.reshape(y_valid,(30,1))
```

##Setosa의 최적의Theta값 구하기


```python
eta = 0.0008
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.005          # 규제 하이퍼파라미터
best_loss = np.infty   # 최소 손실값 기억 변수

Theta1 = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    # 훈련 및 손실 계산
    logits = X_train.dot(Theta1)
    Y_proba = logistic_sigmoid(logits)
    error = Y_proba - Setosa_train
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, 1]), alpha * Theta1[1:]]
    Theta1 = Theta1 - eta * gradients

    # 검증 세트에 대한 손실 계산
    logits = X_valid.dot(Theta1)
    Y_proba = logistic_sigmoid(logits)
    xentropy_loss2 = -1/m*(np.sum(Setosa_valid * np.log(Y_proba + epsilon) + (1 - Setosa_valid ) * np.log(1 - Y_proba + epsilon)))
    l2_loss = 1/2 * np.sum(np.square(Theta1[1:]))
    loss = xentropy_loss2 + alpha * l2_loss
    
    
    # 500 에포크마다 검증 세트에 대한 손실 출력
    if iteration % 500 == 0:
        print(iteration, loss)
        
    # 에포크마다 최소 손실값 업데이트
    if loss < best_loss:
        best_loss = loss
    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료
        print(iteration - 1, best_loss)        # 종료되지 이전 에포크의 손실값 출력
        print(iteration, loss, "조기 종료!")
        break
```

    0 0.1996296961008965
    500 0.17739812210024686
    1000 0.16299034924823363
    1500 0.15304665975108023
    2000 0.14576656980282693
    2500 0.14015313089183962
    3000 0.135630668832707
    3500 0.13185268070140238
    4000 0.12860251358669347
    4500 0.1257400034208949
    5000 0.1231716577682144



```python
logits = X_valid.dot(Theta1)
Y_proba = logistic_sigmoid(logits)
Y_proba1 = np.where(Y_proba >= 0.5 ,1,0)
y_predict = Y_proba1

Setosa_valid = Setosa_valid.reshape(30, 1)

accuracy_score = np.mean(y_predict == Setosa_valid)
accuracy_score
```




    0.7666666666666667




```python
Theta1
```




    array([[ 0.1879383 ],
           [-1.08177422]])



##Versicolor의 최적의Theta값 구하기


```python
eta = 0.0008
n_iterations = 10001
m = len(X_train)
epsilon = 1e-7
alpha = 0.005        # 규제 하이퍼파라미터
best_loss = np.infty   # 최소 손실값 기억 변수

Theta2 = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    # 훈련 및 손실 계산
    logits = X_train.dot(Theta2)
    Y_proba = logistic_sigmoid(logits)
    error = Y_proba - Versicolor_train
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, 1]), alpha * Theta2[1:]]
    Theta2 = Theta2 - eta * gradients

    # 검증 세트에 대한 손실 계산
    logits = X_valid.dot(Theta2)
    Y_proba = logistic_sigmoid(logits)
    xentropy_loss2 = -1/m*(np.sum(Versicolor_valid * np.log(Y_proba + epsilon) + (1 - Versicolor_valid ) * np.log(1 - Y_proba + epsilon)))
    l2_loss = 1/2 * np.sum(np.square(Theta2[1:]))
    loss = xentropy_loss2 + alpha * l2_loss
    
    # 500 에포크마다 검증 세트에 대한 손실 출력
    if iteration % 500 == 0:
        print(iteration, loss)
        
    # 에포크마다 최소 손실값 업데이트
    if loss < best_loss:
        best_loss = loss
    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료
        print(iteration - 1, best_loss)        # 종료되지 이전 에포크의 손실값 출력
        print(iteration, loss, "조기 종료!")
        break
```

    0 0.2501474525276718
    500 0.2354633253410091
    1000 0.22737208988730617
    1500 0.2230404286972585
    2000 0.220775948521791
    2500 0.21962683007372982
    3000 0.21907136046660802
    3500 0.2188265423228695
    4000 0.21873979067563534
    4344 0.2187276387887131
    4345 0.2187276388602661 조기 종료!



```python
logits = X_valid.dot(Theta2)
Y_proba = logistic_sigmoid(logits)
Y_proba2 = np.where(Y_proba >= 0.5 ,1,0)
y_predict = Y_proba2

Versicolor_valid = Versicolor_valid.reshape(30, 1)

accuracy_score = np.mean(y_predict == Versicolor_valid)
accuracy_score

```




    0.6333333333333333




```python
Theta2
```




    array([[-0.6019138 ],
           [ 0.05651226]])



##Virginica의 최적의Theta값 구하기


```python
eta = 0.008
n_iterations = 10001
m = len(X_train)
epsilon = 1e-7
alpha = 0.005           # 규제 하이퍼파라미터
best_loss = np.infty   # 최소 손실값 기억 변수

Theta3 = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    # 훈련 및 손실 계산
    logits = X_train.dot(Theta3)
    Y_proba = logistic_sigmoid(logits)
    error = Y_proba - Virginica_train
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, 1]), alpha * Theta3[1:]]
    Theta3 = Theta3 - eta * gradients

    # 검증 세트에 대한 손실 계산
    logits = X_valid.dot(Theta3)
    Y_proba = logistic_sigmoid(logits)
    xentropy_loss2 = -1/m*(np.sum(Virginica_valid * np.log(Y_proba + epsilon) + (1 - Virginica_valid ) * np.log(1 - Y_proba + epsilon)))
    l2_loss = 1/2 * np.sum(np.square(Theta3[1:]))
    loss = xentropy_loss2 + alpha * l2_loss
    
    # 500 에포크마다 검증 세트에 대한 손실 출력
    if iteration % 500 == 0:
        print(iteration, loss)
        
    # 에포크마다 최소 손실값 업데이트
    if loss < best_loss:
        best_loss = loss
    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료
        print(iteration - 1, best_loss)        # 종료되지 이전 에포크의 손실값 출력
        print(iteration, loss, "조기 종료!")
        break
```

    0 0.3071496303877346
    500 0.19003757265572152
    1000 0.16832111271168623
    1500 0.15358816175031445
    2000 0.1429649320569367
    2500 0.13503119590818174
    3000 0.1289289598008823
    3500 0.12411975432260966
    4000 0.12025301909929337
    4500 0.11709255918482002
    5000 0.1144742113381786
    5500 0.11228067128753232
    6000 0.11042603494315612
    6500 0.10884601110333854
    7000 0.1074915478759696
    7500 0.10632457395054987
    8000 0.10531508562125125
    8500 0.10443911153183927
    9000 0.10367726296976444
    9500 0.10301368295886074
    10000 0.10243527215439507



```python
logits = X_valid.dot(Theta3)
Y_proba = logistic_sigmoid(logits)
Y_proba3 = np.where(Y_proba >= 0.5 ,1,0)
y_predict = Y_proba3

Virginica_valid = Virginica_valid.reshape(30, 1)

accuracy_score = np.mean(y_predict == Virginica_valid)
accuracy_score
```




    0.9666666666666667




```python
Theta3
```




    array([[-4.20758763],
           [ 2.563671  ]])



### 각각의 최적의Theta값을 Y_proba에 배열시킨 후 가장 높은 확률을 갖는 클래스를 선택함.


```python
## 검증세트

logits_Setosa = X_valid.dot(Theta1)
logits_Versicolor = X_valid.dot(Theta2)
logits_Virginica = X_valid.dot(Theta3)

Y_proba_Setosa = logistic_sigmoid(logits_Setosa)
Y_proba_Versicolor = logistic_sigmoid(logits_Versicolor)
Y_proba_Virginica = logistic_sigmoid(logits_Virginica)

Y_proba = np.hstack((Y_proba_Setosa,Y_proba_Versicolor,Y_proba_Virginica))
y_predict = np.argmax(Y_proba, axis=1)          # 가장 높은 확률을 갖는 클래스 선택

y_predict = np.reshape(y_predict,(30,1))

accuracy_score = np.mean(y_predict == y_valid)  # 정확도 계산
accuracy_score
```




    0.9




```python
##테스트 세트

logits_Setosa = X_test.dot(Theta1)
logits_Versicolor = X_test.dot(Theta2)
logits_Virginica = X_test.dot(Theta3)

Y_proba_Setosa = logistic_sigmoid(logits_Setosa)
Y_proba_Versicolor = logistic_sigmoid(logits_Versicolor)
Y_proba_Virginica = logistic_sigmoid(logits_Virginica)

Y_proba = np.hstack((Y_proba_Setosa, Y_proba_Versicolor, Y_proba_Virginica))
y_predict = np.argmax(Y_proba, axis=1)

accuracy_score = np.mean(y_predict == y_test)
accuracy_score
```




    0.9333333333333333


