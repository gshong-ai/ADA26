# 02. Swiss Roll로 배우는 PCA / LDA / MDS

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gshong-ai/ADA26/blob/claude/mnist-dimensionality-reduction-H2Meg/02_pca_lda_mds_comparison/pca_lda_mds_comparison.ipynb)

## 학습 목표

동일한 **Swiss Roll** 데이터에 PCA / LDA / 고전 MDS / Isomap(측지 MDS)을 적용하여
각 기법이 **왜 성공하고 왜 실패하는지**를 시각적·정량적으로 비교합니다.

## 실험 설계

| 기법 | 핵심 아이디어 | 지도 | Swiss Roll 결과 |
|------|-------------|:----:|:--------------:|
| PCA | 분산 최대 방향으로 선형 투영 | ❌ | ✗ 접힌 층이 겹침 |
| LDA | 클래스 간/내 분산비 최대화 | ✅ | ✗ 선형 경계로 층 구분만 |
| 고전 MDS | 유클리드 쌍별 거리 보존 | ❌ | ✗ PCA와 유사 (비선형 구조 무시) |
| Isomap | **측지 거리** 보존 (k-NN 최단 경로) | ❌ | ✓ 매니폴드 언폴딩 |

## 핵심 인사이트

**유클리드 거리 ≠ 측지 거리** — Swiss Roll 실패의 근본 원인

- Swiss Roll에서 가까운 두 점이 **다른 층(spiral revolution)**에 있을 수 있음
- 유클리드 거리는 작지만 매니폴드 위에서의 실제 거리(측지)는 매우 큼
- PCA/MDS는 유클리드 거리를 사용 → 접힌 구조를 풀지 못함
- Isomap은 k-NN 그래프 최단 경로로 측지 거리를 근사 → 언폴딩 성공

## 정량 평가 지표

- **이웃 t값 분산** (낮을수록 = 매니폴드 구조 잘 보존)
- **실루엣 스코어** (t 등분 클래스 기준)
- **Shepard 다이어그램** (유클리드 vs 측지 거리 왜곡 시각화)

## 실행 방법

```bash
pip install scikit-learn plotly scipy numpy
jupyter notebook pca_lda_mds_comparison.ipynb
```
