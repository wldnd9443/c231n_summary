# c231n_summary

## 목차

## Image classificaiton
이미지 분류, Input으로 사진을 받으면 Output으로 Label을 반환한다. 

![classification3](https://user-images.githubusercontent.com/44831709/147997141-22520588-335f-4b81-a436-5c830f987a07.png)

위와 같은 사진을 classifier를 이용하여 분류할 수 있다.

* Nearest Neighbor

 아래의 그림과 같이 이미지의 모든 픽셀을 수치화 하고 다른 이미지와 각 픽셀 값의 차이의 절대값을 취하여 유사도를 측정한다. 아래 그림은 456의 유사도를 갖는다. 이와 같은 방법은 각각의 픽셀을 서로 비교하는 원초적인 방법이라 간단하지만 그만큼 학습에 많은 시간이 필요하다.
 
 ![image](https://user-images.githubusercontent.com/44831709/148003066-850ecdc1-61a9-4d27-97c1-db6ee37978af.png)
 
 
 
* K-Neareast Neighbor 

  이미지를 평면상에 배치하고 새로운 이미지가 들어왔을때 그 이미지가 이미 배치된 이미지와 가장 가까운 거리에 있는 값으로 label을 예측하는 방법이다. 
  
![image](https://user-images.githubusercontent.com/44831709/148005103-fcff083b-e717-4246-ba4f-10c4574d3f9f.png)



* Linear Classifier

![image](https://user-images.githubusercontent.com/44831709/148019387-bbd6ae8b-40f6-4d1d-abd9-aacaeafa39bb.png)

 위의 식처럼 선형의 함수 형태로 이미지를 분류한다. 

![linear_classifier1](https://user-images.githubusercontent.com/44831709/147997717-d019ff36-488d-4452-af67-90a951340eca.png)

여기서 W는 임의로 지정한 Weight 값을 행렬로 나열하고 x는 이미지의 픽셀값을 일렬로 정리한다. 그 결과로 얻은 score에 따라 현 이미지의 label을 예측한다.


## Loss Function and Optimization
### Loss Function 
Loss Function은 학습이 얼마나 잘 이루어 질 수 있는지 비교할 수 있는 기준되므로 기계학습에 있어 앞으로 Hyperparameter를 수정하는데 중요한 단서가 된다. 

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
