11주차 마지막에 가상데이터를 활용해서 이진분류의 ROC를 그리는 것을 해 보았습니다

 

X, y = make_classification(
    n_samples=1000,  # 샘플 개수
    n_classes=3,     # 클래스 개수를 3으로 변경
    n_features=20,   # 특성 개수
    n_informative=15,# 정보를 가진 특성 개수
    random_state=42,
)

 

1. 코드에서 일단 class 수를 3개로 줄이기 위해 위와 같이 가상 데이터 생성을 수정해 보세요.

 

2. ROC 커브는 기본적으로 True/False에 대해 그리므로, 이진분류에 대해서만 그릴 수 있습니다. 그래서 OVR전략을 가정하고, 각 클래스별 ROC 커브를 그려보세요

 

3. 모든 클래스에 대해서 Macro와 Micro average ROC 커브를 그려보세요. 크게 2가지 전략이 있습니다 (다른 전략도 있을 수 있지만...^^;;)

   1) 모든 클래스의 threshold값을 모아서 정렬하는 방법

   2) np.interp를 사용해서 클래스 사이의 값을 일정하게 보간(interpolate)하는 방법

(두 가지 모두를 시도해보시되, 둘 중 하나만 성공해도 정답으로 인정할 예정입니다.)

 

정답은 아래와 같은 형태의 그림이 나와야 합니다. 

<img width="857" height="699" alt="image" src="https://github.com/user-attachments/assets/e040bfb6-132e-457f-869b-f807d8e93076" />
