import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# 1. 가상 데이터 생성 (n_classes=3 적용)
X, y = make_classification(
    n_samples=1000,
    n_classes=3,
    n_features=20,
    n_informative=15,
    random_state=42,
)

# 2. 데이터 전처리 (Binarize label for ROC)
# ROC Curve는 이진 분류 기반이므로 타겟 y를 One-hot 형태로 변환
y_bin = label_binarize(y, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

# 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.5, random_state=42)

# OVR(One-Vs-Rest) 전략을 사용한 분류기 학습
# LogisticRegression을 사용하여 확률값(decision_function)을 얻습니다.
clf = OneVsRestClassifier(LogisticRegression(random_state=42))
y_score = clf.fit(X_train, y_train).decision_function(X_test)

# FPR, TPR, AUC 계산을 위한 딕셔너리 생성
fpr = dict()
tpr = dict()
roc_auc = dict()

# 2. 각 클래스별 ROC Curve 및 AUC 계산
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 3. 모든 클래스에 대해서 Macro와 Micro average ROC 커브 그리기

# 3-1) Micro Average ROC Curve 계산
# y_test와 y_score를 1차원 배열로 펼쳐서(ravel) 전체 데이터에 대해 계산
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# 3-2) Macro Average ROC Curve 계산 (np.interp 보간법 사용)
# 모든 FPR 값을 0부터 1사이의 100개 구간으로 통일
all_fpr = np.linspace(0, 1, 100)

# 각 클래스의 TPR을 all_fpr에 맞춰 보간(interpolate)하고 합산
mean_tpr = 0.0
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# 클래스 개수로 나누어 평균 구하기
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# 4. 그래프 그리기
plt.figure(figsize=(10, 8))

# Micro Average Plot (핑크 점선)
plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

# Macro Average Plot (네이비 점선)
plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

# 각 클래스별 Plot (색상 순환)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2) # 대각선(Random Guess)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve (Micro & Macro Average)')
plt.legend(loc="lower right")
plt.show()
