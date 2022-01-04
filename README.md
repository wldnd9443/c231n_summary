# c231n_summary

## 목차

## Image classificaiton
이미지 분류, Input으로 사진을 받으면 Output으로 Label을 반환한다. 

![classification3](https://user-images.githubusercontent.com/44831709/147997141-22520588-335f-4b81-a436-5c830f987a07.png)

위와 같은 사진을 classifier를 이용하여 분류할 수 있다.

### Nearest Neighbor

 아래의 그림과 같이 이미지의 모든 픽셀을 수치화 하고 다른 이미지와 각 픽셀 값의 차이의 절대값을 취하여 유사도를 측정한다. 아래 그림은 456의 유사도를 갖는다. 이와 같은 방법은 각각의 픽셀을 서로 비교하는 원초적인 방법이라 간단하지만 그만큼 학습에 많은 시간이 필요하다.
 
 ![image](https://user-images.githubusercontent.com/44831709/148003066-850ecdc1-61a9-4d27-97c1-db6ee37978af.png)
 
 
 
### K-Neareast Neighbor 

  이미지를 평면상에 배치하고 새로운 이미지가 들어왔을때 그 이미지가 이미 배치된 이미지와 가장 가까운 거리에 있는 값으로 label을 예측하는 방법이다. 
  
![image](https://user-images.githubusercontent.com/44831709/148005103-fcff083b-e717-4246-ba4f-10c4574d3f9f.png)



### Linear Classifier

![image](https://user-images.githubusercontent.com/44831709/148019387-bbd6ae8b-40f6-4d1d-abd9-aacaeafa39bb.png)

 위의 식처럼 선형의 함수 형태로 이미지를 분류한다. 

![linear_classifier1](https://user-images.githubusercontent.com/44831709/147997717-d019ff36-488d-4452-af67-90a951340eca.png)

여기서 W는 임의로 지정한 Weight 값을 행렬로 나열하고 x는 이미지의 픽셀값을 일렬로 정리한다. 그 결과로 얻은 score에 따라 현 이미지의 label을 예측한다.


## Loss Function and Optimization
### Loss
Loss는 선형분류에서 하나의 예측값이 선형함수로 부터 얼마나 차이가 있는지 나타내는 하나의 값으로 이 값이 클 수록 손실 값이 큼을 의미한다.
Loss는 정의하기에 따라 다양하고 이번 강의에서는 Softmax와 SVM을 소개한다.
![softmax_loss](https://user-images.githubusercontent.com/44831709/148022449-17ca9072-4018-4f18-b4f4-651c42466c47.png)

![SVM_loss](https://user-images.githubusercontent.com/44831709/148022451-35c11d53-c8a4-4af2-9c09-cef67860ca02.png)

### Loss Function 
Loss Function은 학습이 얼마나 잘 이루어 질 수 있는지 비교할 수 있는 기준되므로 기계학습에 있어 앞으로 Hyperparameter를 수정하는데 중요한 단서가 된다. Loss function은 Loss들의 총합에서 전체 개수를 나눈 값으로 다음과 같이 정의한다.

![loss_function](https://user-images.githubusercontent.com/44831709/148022899-22956956-dccd-496e-90a1-4301b2de2088.png)

Loss function을 정의할 때 Regularization을 추가하는 것이 좋다.

![regul_equation](https://user-images.githubusercontent.com/44831709/148027054-e3338e20-b393-4dc6-b98b-b0b6e69d706d.png)

![regularization_isneeded](https://user-images.githubusercontent.com/44831709/148025472-ee8ea757-9794-4d51-a4ac-57b6230d7f85.png)

정규화 과정이 없다면 왼쪽 그래프처럼 일부 값이 우연히 일치 하는 것으로 학습이 완료되었다 착각하고 학습이 조기에 종료될 수 있다.

### Optimization

위에서 Loss Function을 정의했다. 기계학습에서는 이 Loss Function의 값을 최소화 할 수 있는 Parameter를 설정하는 것이 중요한 목표이다. Loss Function의 값을 최소화하기 위해서는 임의의 값을 무한하게 검색하여 그중에 가장 작은 값을 취할 수 있지만 굉장히 비효율적이다.

![loss_fn_opt](https://user-images.githubusercontent.com/44831709/148028366-37b9bf10-ff22-4777-99f4-1dc33eeefb7b.png)


위 그림에서 J 즉 Loss function의 값이 가장 낮은 곳으로 진행해야 한다면 왼쪽에서 오른쪽으로 진행해야 하는데 이는 매 순간에서의 gradient를 구하고 그만큼 진행해 나가며 최적의 cost function을 제공하는 Weight를 찾을 수 있다.
![update_weight](https://user-images.githubusercontent.com/44831709/148030328-a2233741-07ac-40f7-9127-2d399e4e5594.png)



## Backpropagation and Neural Networks

## Convolutional Neural Networks

## Training Neural Networks

## Deep Learning Software

## CNN Architectures

## Recurrent Neural Networks

## Detection Segmentation

## Visualizing and Understanding 

## Generative Models

## Reinforcement Learning 
