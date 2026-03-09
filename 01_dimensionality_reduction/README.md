# 01 — 차원축소 (Dimensionality Reduction)

MNIST 데이터를 3차원 공간에 투영하여 각 기법의 차이를 시각적으로 비교합니다.

## 예제 파일

| 파일 | 설명 |
|------|------|
| `mnist_dim_reduction.py` | PCA / LDA / MDS 3종 비교 — 인터랙티브 3D |
| `mnist_manifold.py` | 병목 레이어(3노드) 신경망 3D Manifold |
| `mnist_manifold_colab.ipynb` | Colab 실행용 노트북 |

## 실행 방법

```bash
cd 01_dimensionality_reduction

# PCA / LDA / MDS 비교 (sklearn만 필요)
python mnist_dim_reduction.py
# → mnist_dim_reduction_3d.html 생성 (브라우저로 열기)

# 병목 신경망 Manifold (tensorflow 필요)
python mnist_manifold.py
# → mnist_manifold_3d.html 생성
```

## 결과 요약

| 기법 | 설명분산 | 특징 |
|------|----------|------|
| PCA | 15% | 분산 최대화, 비지도 |
| LDA | 58% | 클래스 분리 최대화, 지도 |
| MDS | Stress 기반 | 원본 거리 구조 보존, 비지도 |
