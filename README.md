<div align="center">

# 🎓 Student Risk Prediction

**Decision Tree(ID3) 기반 학생 위험군 예측 시스템**

학생의 학습 데이터를 분석하여 학업 위험군 여부를 예측하고,<br>위험 원인과 개선 방향을 자동으로 제시합니다.

<br>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

<br>

![Accuracy](https://img.shields.io/badge/Accuracy-86.08%25-success?style=flat-square)
![Recall](<https://img.shields.io/badge/Recall%20(위험군)-96%25-blue?style=flat-square>)
![Records](https://img.shields.io/badge/Records-6%2C607-lightgrey?style=flat-square)
![Features](https://img.shields.io/badge/Features-9-lightgrey?style=flat-square)

</div>

---

## 📌 프로젝트 개요

| 항목           | 내용                                                     |
| -------------- | -------------------------------------------------------- |
| 🤖 알고리즘    | Decision Tree (ID3, entropy criterion)                   |
| 📂 데이터셋    | StudentPerformanceFactors_encoded.csv (Kaggle 원본 파생) |
| 📊 데이터 크기 | 6,607개 레코드 · 9개 피처                                |
| ⚠️ 위험군 기준 | Train set 기준 하위 23% (77th percentile 미만)           |
| 🔍 최적 Depth  | 5 (5-Fold CV · Recall 기준 탐색)                         |

---

## 📁 파일 구조

```
student-risk-prediction/
├── 📓 Student-Performance-Factors.ipynb   # 데이터 분석 + 모델 학습
└── 📄 StudentPerformanceFactors_encoded.csv  # 인코딩 완료 데이터셋
```

---

## 🗂️ 데이터셋

### 출처

[![Kaggle](https://img.shields.io/badge/Kaggle-Student%20Performance%20Factors-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)

원본 데이터셋: **Student Performance Factors** by [lainguyn123](https://www.kaggle.com/lainguyn123) (Kaggle)

> `StudentPerformanceFactors_encoded.csv` 는 위 원본 데이터셋에서 모델 학습에 필요한 피처를 선별하고,
> 범주형 변수를 숫자로 인코딩하여 가공한 **파생 데이터셋**입니다.

### 피처 설명

| 피처                   | 설명                     | 범위                         |
| ---------------------- | ------------------------ | ---------------------------- |
| `Hours_Studied`        | 주간 학습 시간           | 1 ~ 44                       |
| `Attendance`           | 출석률 (%)               | 60 ~ 100                     |
| `Parental_Involvement` | 부모 참여도              | 0(Low) / 1(Medium) / 2(High) |
| `Access_to_Resources`  | 학습 자료 접근성         | 0(Low) / 1(Medium) / 2(High) |
| `Sleep_Hours`          | 하루 수면 시간           | 4 ~ 10                       |
| `Previous_Scores`      | 이전 시험 점수           | 50 ~ 100                     |
| `Motivation_Level`     | 학습 동기 수준           | 0(Low) / 1(Medium) / 2(High) |
| `Tutoring_Sessions`    | 월간 튜터링 횟수         | 0 ~ 8                        |
| `Exam_Score`           | ⭐ 최종 시험 점수 (타겟) | 55 ~ 101                     |

> ✅ 결측값 없음 — 6,607개 전체 레코드 완전

### 기초 통계

| 피처              | 평균  | 표준편차 | 최솟값 | 최댓값 |
| ----------------- | ----- | -------- | ------ | ------ |
| Hours_Studied     | 19.98 | 5.99     | 1      | 44     |
| Attendance        | 79.98 | 11.55    | 60     | 100    |
| Sleep_Hours       | 7.03  | 1.47     | 4      | 10     |
| Previous_Scores   | 75.07 | 14.40    | 50     | 100    |
| Tutoring_Sessions | 1.49  | —        | 0      | 8      |

---

## 🔬 데이터 분석 결과

### Exam_Score와의 상관관계

```
Attendance           ████████████████████████████░░  0.581  ★ 1위
Hours_Studied        ██████████████████████░░░░░░░░  0.445  ★ 2위
Previous_Scores      █████████░░░░░░░░░░░░░░░░░░░░░  0.175
Access_to_Resources  ████████░░░░░░░░░░░░░░░░░░░░░░  0.170
Parental_Involvement ████████░░░░░░░░░░░░░░░░░░░░░░  0.157
Tutoring_Sessions    ████████░░░░░░░░░░░░░░░░░░░░░░  0.157
Motivation_Level     ████░░░░░░░░░░░░░░░░░░░░░░░░░░  0.087
Sleep_Hours          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ -0.017
```

> 📌 **출석률(0.581)** 과 **학습 시간(0.445)** 이 시험 점수에 가장 강하게 영향을 미칩니다.

---

## 🤖 모델 성능

### 평가 결과 — 정확도 86.08%

| 클래스       | Precision | Recall   | F1-Score | Support |
| ------------ | --------- | -------- | -------- | ------- |
| 0 · 정상군   | 0.83      | 0.59     | 0.69     | 348     |
| 1 · 위험군   | 0.87      | **0.96** | 0.91     | 974     |
| Weighted Avg | **0.86**  | **0.86** | **0.85** | 1,322   |

> 🎯 위험군 **Recall 0.96** — 실제 위험 학생의 96%를 정확히 탐지

### Confusion Matrix

```
                  예측: 정상    예측: 위험
  실제: 정상  │    206    │    142    │
  실제: 위험  │     42    │    932    │
```

### Feature Importance

```
Attendance           ██████████████████████████  51.43%  ★★★
Hours_Studied        ██████████████████          35.90%  ★★★
Previous_Scores      █████                        8.46%  ★★
Tutoring_Sessions    █                            1.63%
Access_to_Resources  ▌                            0.85%
Parental_Involvement ▌                            0.84%
기타                 ░                           < 0.60%
```

---

## 🔎 샘플 예측 예시

### 입력 프로파일

| 피처                 | 값    | 평가         |
| -------------------- | ----- | ------------ |
| Hours_Studied        | 5     | 🔴 매우 낮음 |
| Attendance           | 60%   | 🔴 최솟값    |
| Sleep_Hours          | 5시간 | 🟡 부족      |
| Previous_Scores      | 50점  | 🔴 하위권    |
| Tutoring_Sessions    | 0회   | 🔴 없음      |
| Parental_Involvement | Low   | 🔴 낮음      |
| Access_to_Resources  | Low   | 🔴 낮음      |
| Motivation_Level     | Low   | 🔴 낮음      |

### 예측 결과

```
┌─────────────────────────────────────┐
│  ⚠️  예측 결과 : 위험군 학생           │
└─────────────────────────────────────┘
```

### 의사결정 경로 분석

| 분기 조건                   | 현재 값 | 판정           |
| --------------------------- | ------- | -------------- |
| Attendance ≤ 84.5           | 60      | 🔴 위험도 증가 |
| Hours_Studied ≤ 26.5        | 5       | 🔴 위험도 증가 |
| Hours_Studied ≤ 20.5        | 5       | 🔴 위험도 증가 |
| Previous_Scores ≤ 89.5      | 50      | 🔴 위험도 증가 |
| Motivation_Level_High ≤ 0.5 | 0       | 🔴 위험도 증가 |

### 개선 방향

| 우선순위 | 항목         | 권고 사항                     |
| -------- | ------------ | ----------------------------- |
| 1        | 📅 출석률    | 출석률 85% 이상 유지          |
| 2        | 📚 학습 시간 | 주간 학습 시간 증가 필요      |
| 3        | 📝 기초 학력 | 이전 점수 기반 기초 개념 복습 |
| 4        | 💡 학습 동기 | 학습 동기 향상 프로그램 참여  |

---

## 🚀 실행 방법

**Google Colab 환경에서 실행합니다.**

1. 두 파일을 같은 폴더에 두고 Colab에서 `Student-Performance-Factors.ipynb` 열기
2. 전체 실행 (`런타임 > 모두 실행`)

```python
df = pd.read_csv("StudentPerformanceFactors_encoded.csv")
```

---

## 🛠️ 기술 스택

<div align="center">

| 라이브러리     | 용도                                   |
| -------------- | -------------------------------------- |
| `pandas`       | 데이터 로드 및 전처리                  |
| `scikit-learn` | DecisionTreeClassifier, 교차검증, 평가 |
| `matplotlib`   | 분포 시각화, 히스토그램                |
| `seaborn`      | 상관관계 히트맵                        |

</div>
