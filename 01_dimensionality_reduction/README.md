# 01 — 차원축소 (Dimensionality Reduction)

MNIST 데이터를 3차원 공간에 투영하여 각 기법의 차이를 시각적으로 비교합니다.

## 예제 파일

| 파일 | 설명 | Colab |
|------|------|-------|
| `mnist_dim_reduction.ipynb` | PCA / LDA / MDS 3종 비교 — 인터랙티브 3D | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gshong-ai/ADA26/blob/main/01_dimensionality_reduction/mnist_dim_reduction.ipynb) |
| `mnist_manifold_colab.ipynb` | 병목 레이어(3노드) 신경망 3D Manifold | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gshong-ai/ADA26/blob/main/01_dimensionality_reduction/mnist_manifold_colab.ipynb) |

## 실행 방법

### Colab (권장)
위 배지를 클릭하면 바로 실행할 수 있습니다.

### 로컬 실행
```bash
cd 01_dimensionality_reduction
jupyter notebook mnist_dim_reduction.ipynb
```

## 결과 요약

| 기법 | 설명분산 | 특징 |
|------|----------|------|
| PCA | 15% | 분산 최대화, 비지도 |
| LDA | 58% | 클래스 분리 최대화, 지도 |
| MDS | Stress 기반 | 원본 거리 구조 보존, 비지도 |
