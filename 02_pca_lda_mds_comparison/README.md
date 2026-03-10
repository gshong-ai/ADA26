# 02. PCA vs LDA vs MDS — 각 기법이 빛나는 순간

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gshong-ai/ADA26/blob/claude/mnist-dimensionality-reduction-H2Meg/02_pca_lda_mds_comparison/pca_lda_mds_comparison.ipynb)

## 학습 목표

차원 축소 기법은 **데이터 특성에 따라 성능이 크게 달라집니다**.
이 실습은 각 기법이 **성공하는 조건**과 **실패하는 조건**을 시각적으로 비교합니다.

## 시나리오 요약

| 시나리오 | 데이터 | 성공 기법 | 실패 기법 | 실패 원인 |
|:--------:|:------:|:---------:|:---------:|:---------:|
| **A** | 길쭉한 가우시안 (합성) | **LDA** | PCA, MDS | 최대 분산 ≠ 판별 방향 |
| **B** | XOR 다중 모달 + 고차원 노이즈 (합성) | **PCA** | LDA, MDS | 클래스 평균 동일 / 차원의 저주 |
| **C** | Swiss Roll (sklearn) | **MDS (Isomap)** | PCA, LDA | 비선형 매니폴드 |

## 성공/실패 기준

- **구조 보존**: 원본 데이터의 기하학적 구조가 저차원 투영에서 유지되는가?
- **실루엣 스코어**로 정량 측정

## 기법 요약

| 기법 | 핵심 아이디어 | 지도/비지도 |
|------|--------------|:-----------:|
| PCA | 분산이 최대인 방향으로 투영 | 비지도 |
| LDA | 클래스 간 분산 / 클래스 내 분산을 최대화 | 지도 |
| MDS (Isomap) | 측지선 거리를 보존하며 저차원 매핑 | 비지도 |

## 실행 방법

### Google Colab (권장)
위 뱃지를 클릭하거나 노트북을 직접 Colab에서 열어 셀을 순서대로 실행합니다.

### 로컬 실행
```bash
pip install scikit-learn plotly numpy matplotlib pandas
jupyter notebook pca_lda_mds_comparison.ipynb
```

## 핵심 교훈

> **"모든 상황에서 최고인 차원 축소 기법은 없다."**
> 데이터의 특성(레이블 유무, 선형/비선형, 차원 수)을 먼저 파악하고 목적에 맞는 기법을 선택하자.
