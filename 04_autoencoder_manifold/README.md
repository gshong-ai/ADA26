# 04. PCA는 선형 뉴런, MLP는 비선형 매니폴드 언폴딩?

## 개요

**핵심 질문:**
> PCA가 linear activation 하나짜리 뉴런이라면,
> ReLU를 쓰는 MLP로 Swiss Roll을 2D에 투영하면 Isomap처럼 펼쳐질 수 있을까?

## 실험 설계

| 모델 | 인코더 구조 | 활성화 |
|------|------------|--------|
| PCA | — (분석해) | 선형 |
| 선형 AE | 3→2 | linear (PCA와 이론적 동치) |
| MLP-4 | 3→**4**→2 | ReLU hidden |
| MLP-8 | 3→**8**→2 | ReLU hidden |
| MLP-16 | 3→**16**→2 | ReLU hidden |
| Isomap | — (측지 거리) | — |

## 파일 구성

```
04_autoencoder_manifold/
├── autoencoder_swiss_roll.ipynb    # 메인 노트북
├── autoencoder_swiss_roll.html     # 시각화 결과 (실행 후 생성)
└── README.md
```

## 실행 방법

```bash
pip install plotly scikit-learn tensorflow scipy
jupyter notebook autoencoder_swiss_roll.ipynb
```
