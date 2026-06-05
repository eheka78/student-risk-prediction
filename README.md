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

| 항목           | 내용                                                                 |
| -------------- | -------------------------------------------------------------------- |
| 🤖 알고리즘    | Decision Tree (ID3, entropy criterion)                               |
| 📂 데이터셋    | StudentPerformanceFactors_encoded.csv (Kaggle 원본 파생)             |
| 📊 데이터 크기 | 6,607개 레코드 · 9개 피처                                            |
| ⚠️ 위험군 기준 | Train set 기준 하위 23% (77th percentile 미만, 내신 3등급 이상 기준) |
| 🔍 최적 Depth  | 5 (5-Fold CV · Recall 기준 탐색)                                     |

---

## 🎯 문제 정의

학생의 학업 성취도를 결정짓는 요인은 무엇인가?

일반적으로 공부 시간이 많을수록 성적이 높아질 것이라 생각하지만, 실제 학습 성과는 출석률 · 학습 동기 · 이전 성적 · 학습 환경 등 **다차원적 요인이 복합적으로 작용**하여 결정된다.

본 프로젝트는 학생의 생활 및 학습 패턴과 관련된 다차원적 요인을 활용하여, 학생이 다음 시험에서 **상위 23% (내신 3등급 이상)** 에 진입할 수 있는지를 예측하는 모델을 구축하는 것을 목표로 한다.

> 피처 선정 기준: 학생이 **스스로 개선할 수 있는 요인**에 집중하여 8개 변수를 선정하였다.

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

## 🧠 알고리즘 선정 이유

본 문제는 학생이 상위 23%에 진입할 수 있는지를 예측하는 **이진 분류 문제**이다.

| 알고리즘             | 검토 결과                                                                            |
| -------------------- | ------------------------------------------------------------------------------------ |
| Logistic Regression  | 복합적인 비선형 관계를 충분히 표현하기 어려움                                        |
| SVM                  | 높은 성능을 기대할 수 있으나 의사결정 과정 해석 어려움                               |
| MLP                  | 높은 성능을 기대할 수 있으나 의사결정 과정 해석 어려움                               |
| **Decision Tree** ✅ | 비선형 관계 학습 + 수치형/범주형 변수 동시 처리 + **의사결정 과정 직관적 설명 가능** |

Decision Tree를 선택한 핵심 이유는 **예측 근거의 해석 가능성**이다. 단순히 위험 여부를 예측하는 것을 넘어, 학생별 위험 원인과 개선 방향을 `decision_path()`를 통해 단계별로 추적·설명할 수 있다.

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

> ⚠️ **클래스 불균형 분석**
> 테스트 데이터 내 위험군(974명) : 정상군(348명) ≈ **3 : 1** 로 불균형이 존재한다.
> 이로 인해 모델이 위험군 쪽으로 편향되어 정상군 학생 142명을 위험군으로 과대 예측하는 경향이 나타났다.
> 정상군 Recall(0.59)이 낮은 주된 원인으로 해석된다.

### Feature Importance

| 피처                 | 가중치     | 시각화                                    |
| -------------------- | ---------- | ----------------------------------------- |
| Attendance           | **0.5143** | `██████████████████████████░░░░░░░░░░░░░` |
| Hours_Studied        | **0.3590** | `██████████████████░░░░░░░░░░░░░░░░░░░░░` |
| Previous_Scores      | 0.0846     | `█████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` |
| Tutoring_Sessions    | 0.0163     | `█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` |
| Access_to_Resources  | 0.0085     | `▌░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` |
| Parental_Involvement | 0.0084     | `▌░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` |
| Motivation_Level     | 0.0034     | `░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` |
| Sleep_Hours          | 0.0000     | `░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` |

---

## 🔎 샘플 예측 예시

### 입력 프로파일

| 피처                 | 값     | 평가         |
| -------------------- | ------ | ------------ |
| Hours_Studied        | 27시간 | 🟢 평균 이상 |
| Attendance           | 75%    | 🟡 평균 미만 |
| Sleep_Hours          | 6시간  | 🟡 적정      |
| Previous_Scores      | 65점   | 🟡 중간      |
| Tutoring_Sessions    | 1회    | 🟡 낮음      |
| Parental_Involvement | Medium | 🟡 보통      |
| Access_to_Resources  | Medium | 🟡 보통      |
| Motivation_Level     | Medium | 🟡 보통      |

### 예측 결과

```
┌─────────────────────────────────────┐
│  ⚠️  예측 결과 : 위험군 학생           │
└─────────────────────────────────────┘
```

### 의사결정 경로 분석

| 분기 조건              | 현재 값 | 판정           |
| ---------------------- | ------- | -------------- |
| Attendance ≤ 84.5      | 75      | 🔴 위험도 증가 |
| Hours_Studied > 26.5   | 27      | 🟢 위험도 감소 |
| Attendance ≤ 75.5      | 75      | 🔴 위험도 증가 |
| Previous_Scores ≤ 81.5 | 65      | 🔴 위험도 증가 |
| Attendance > 64.5      | 75      | 🟢 위험도 감소 |

### 개선 방향

| 우선순위 | 항목         | 권고 사항                     |
| -------- | ------------ | ----------------------------- |
| 1        | 📅 출석률    | 출석률 76% 이상으로 개선 필요 |
| 2        | 📝 기초 학력 | 이전 점수 기반 기초 개념 복습 |

---

## 🚀 실행 방법

### ☁️ Google Colab (권장)

**1. 파일 업로드**

[Google Colab](https://colab.research.google.com) 접속 후 노트북을 업로드합니다.

```
파일 > 노트북 업로드 > Student-Performance-Factors.ipynb 선택
```

**2. 데이터셋 업로드**

왼쪽 파일 탐색기에서 CSV 파일을 업로드합니다.

```
좌측 파일 아이콘 클릭 > 파일 업로드 > StudentPerformanceFactors_encoded.csv 선택
```

**3. 전체 실행**

```
런타임 > 모두 실행 (Ctrl + F9)
```

---

### 💻 로컬 환경

**1. 저장소 클론**

```bash
git clone https://github.com/eheka78/student-risk-prediction.git
```

**2. 노트북 열고 전체 실행**

VS Code 또는 Jupyter에서 `Student-Performance-Factors.ipynb` 열고 **Run All** 클릭

> 첫 번째 셀이 필요한 패키지를 자동으로 설치합니다.

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

---

## 📝 결론 및 향후 개선 방향

### 결론

- 출석률(0.514)과 학습 시간(0.359)이 학업 성취도에 **압도적인 영향**을 미침을 실험적으로 확인하였다. 이는 실험 전 예상과 일치하는 결과다.
- Decision Tree는 단순 예측을 넘어 `decision_path()`를 통해 **학생별 위험 원인을 단계별로 설명**할 수 있어, 교육 현장에서 실질적인 개입 근거를 제공할 수 있다.
- 위험군 Recall **0.96** — 학업 지원이 필요한 학생을 효과적으로 탐지한다.

### 한계점

- 테스트 데이터 내 위험군 : 정상군 비율이 약 3 : 1로, **클래스 불균형**으로 인해 정상군 Recall(0.59)이 낮다.
- 일부 정상군 학생을 위험군으로 과대 예측하는 경향이 존재한다.

### 향후 개선 방향

| 방향                | 내용                                                             |
| ------------------- | ---------------------------------------------------------------- |
| 클래스 불균형 완화  | 오버샘플링(SMOTE) 또는 클래스 가중치 조정으로 정상군 Recall 개선 |
| 하이퍼파라미터 튜닝 | 트리 깊이 · 분할 기준 등 추가 최적화                             |
| 변수 확장           | 추가 학습 관련 변수 활용으로 일반화 성능 향상                    |
| 데이터 다양화       | 다양한 교육 데이터와 결합하여 모델 범용성 확대                   |
