# Video-Summarization (Capstone Design 2021-1)
소프트웨어융합학과 옥수빈

## Overview
 비디오 촬영 및 업로드가 간편해지고 특히 YouTube가 크게 성장하면서, 비디오 데이터의 생산량이 급격하게 증가하고 있다. 이러한 방대한 데이터 중에서 원하는 정보만을 효율적으로 다루기 위해서 비디오 영상의 요약 기술이 필요하다. 이번 캡스톤 디자인 수업에서는 LSTM 모델을 기반으로 하는 비디오의 요약 모델을 구현할 것이다. 구현 모델은 주어진 비디오 영상에서 중요한 부분만을 요약하여 보여줄 수 있도록 한다. 요약이 필요한 비디오를 입력함으로써 필요한 부분의 영상을 볼 수 있도록 하며, 요약 영상을 통해서 원본 영상의 전체적인 내용도 알 수 있도록 한다. 
   “Video Summarization with Long Short-term Memory[1]” 논문을 참고하였는데, 본 논문에서는 dppLSTM이라는 모델을 제안하는데 이는 2개의 LSTM 레이어로 구성된 vsLSTM (본 논문에서 제안하는 또 다른 모델)과 DPP (Determinantal Point Processes)를 결합한 모델이다. 여기서 제안하는 두 모델 (vsLSTM과 dppLSTM)의 성능을 평가하기 위해서는 SumMe, TVSum 데이터셋을 사용하였고, 다른 모델인 MLP와 vsLSTM의 성능을 비교한 후, vsLSTM과 dppLSTM의 성능을 비교하는 과정을 거친다. 그 결과로 MLP보다 vsLSTM이 대부분의 경우에 좋은 성능을 보였고, vsLSTM보다 dppLSTM이 더 좋은 성능을 보였다. 이 점을 참고하여 과제를 수행해보려고 한다.


## 1. Model 설계
&nbsp;&nbsp;참고한 모델의 전반적인 구조는 [그림 1]과 같다.<br/>
![model](./ReadMe/model.jpg)<br/>
[그림 1] 모델의 구조  

**(1) **<br/>
&nbsp;

## 2. Model 구현


## 3. Result
***

## 4. conclusion  
***

## 5. 시연 동영상
***

## 참고문헌
[1] Zhang, K., Chao, W. L., Sha, F., & Grauman, K. (2016, October). Video summarization with long short-term memory. In European conference on computer vision (pp. 766-782). Springer, Cham.
