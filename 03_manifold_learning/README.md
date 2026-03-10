# 03. MNIST 매니폴드 학습: t-SNE · UMAP · Isomap · LLE

## 개요

MNIST 손글씨 숫자 데이터셋을 **비선형 매니폴드 학습** 기법으로 3차원 투영하고 Plotly로 인터랙티브하게 비교합니다.

| 기법 | 유형 | 핵심 아이디어 |
|------|------|--------------|
| **t-SNE** | 비지도 / 비선형 | 고차원 유사도 → 저차원 확률 분포 매칭 (KL 발산 최소화) |
| **UMAP** | 비지도 / 비선형 | 리만 기하 기반 위상 구조 보존 |
| **Isomap** | 비지도 / 비선형 | 측지 거리(그래프 최단 경로) 기반 MDS |
| **LLE** | 비지도 / 비선형 | 이웃 재구성 가중치 보존 |

## 파일 구성

```
03_manifold_learning/
├── mnist_manifold_3d.ipynb     # 메인 노트북
├── mnist_manifold_3d.html      # 인터랙티브 시각화 결과 (실행 후 생성)
└── README.md
```

## 이전 실습

- `01_dimensionality_reduction/` — PCA·LDA·MDS 기초
- `02_pca_lda_mds_comparison/` — PCA·LDA·MDS 성공/실패 시나리오 비교

## 실행 방법

### Google Colab
노트북 상단의 **Open in Colab** 배지 클릭

### 로컬 실행
```bash
pip install plotly scikit-learn umap-learn tensorflow
jupyter notebook mnist_manifold_3d.ipynb
```
