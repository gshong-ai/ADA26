"""
MNIST 차원축소 비교 시각화: PCA vs LDA vs MDS (3D Interactive)
- 세 가지 고전적 차원축소 기법으로 MNIST를 3차원에 투영
- Plotly로 인터랙티브 3D 시각화 (마우스 회전/줌/호버)
- 결과를 단일 HTML 파일로 저장
"""

import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
N_SAMPLES   = 3000   # 전체 사용 샘플 수 (MDS 계산 비용 고려)
N_MDS       = 1500   # MDS에 사용할 샘플 수 (O(n²) 메모리)
RANDOM_SEED = 42
OUTPUT_HTML = "mnist_dim_reduction_3d.html"

COLORS = [
    "#E74C3C",  # 0 — 빨강
    "#3498DB",  # 1 — 파랑
    "#2ECC71",  # 2 — 초록
    "#F39C12",  # 3 — 주황
    "#9B59B6",  # 4 — 보라
    "#1ABC9C",  # 5 — 청록
    "#E67E22",  # 6 — 진주황
    "#34495E",  # 7 — 짙은 회색
    "#E91E63",  # 8 — 핑크
    "#00BCD4",  # 9 — 하늘색
]

# ─────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────
def _read_idx(path):
    """IDX 바이너리 파일 파싱"""
    import struct, gzip, os
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim  = magic & 0xFF
        dims  = tuple(struct.unpack(">I", f.read(4))[0] for _ in range(ndim))
        data  = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(dims)


def load_mnist(n_samples=N_SAMPLES, seed=RANDOM_SEED):
    """MNIST IDX 파일 로드 → 정규화 → 무작위 서브샘플링"""
    print("MNIST 데이터 로드 중...")
    base = "data/MNIST/raw"
    x_train = _read_idx(f"{base}/train-images-idx3-ubyte").reshape(-1, 784).astype("float32")
    y_train = _read_idx(f"{base}/train-labels-idx1-ubyte")
    x_test  = _read_idx(f"{base}/t10k-images-idx3-ubyte").reshape(-1, 784).astype("float32")
    y_test  = _read_idx(f"{base}/t10k-labels-idx1-ubyte")
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    # 정규화
    x = x / 255.0

    # 클래스 균형 서브샘플링 (클래스당 동일 개수)
    rng = np.random.default_rng(seed)
    per_class = n_samples // 10
    idx = []
    for c in range(10):
        ci = np.where(y == c)[0]
        idx.append(rng.choice(ci, size=per_class, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)

    print(f"  사용 샘플: {len(idx):,}개  (클래스당 {per_class}개)")
    return x[idx], y[idx]


# ─────────────────────────────────────────
# 2. 차원축소
# ─────────────────────────────────────────
def apply_pca(x, y):
    print("\n[PCA] 3D 투영 중...")
    t0 = time.time()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pca = PCA(n_components=3, random_state=RANDOM_SEED)
    coords = pca.fit_transform(x_scaled)
    elapsed = time.time() - t0
    var = pca.explained_variance_ratio_
    print(f"  완료 ({elapsed:.1f}s)  |  설명분산: PC1={var[0]:.3f}, PC2={var[1]:.3f}, PC3={var[2]:.3f}  |  합계={var.sum():.3f}")
    return coords, {
        "explained_variance": var,
        "total_variance": var.sum(),
        "elapsed": elapsed,
    }


def apply_lda(x, y):
    print("\n[LDA] 3D 투영 중...")
    t0 = time.time()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    # LDA: 최대 (n_classes - 1) = 9개 성분 → 3D 선택
    lda = LDA(n_components=3)
    coords = lda.fit_transform(x_scaled, y)
    elapsed = time.time() - t0
    var = lda.explained_variance_ratio_
    print(f"  완료 ({elapsed:.1f}s)  |  설명분산: LD1={var[0]:.3f}, LD2={var[1]:.3f}, LD3={var[2]:.3f}  |  합계={var.sum():.3f}")
    return coords, {
        "explained_variance": var,
        "total_variance": var.sum(),
        "elapsed": elapsed,
    }


def apply_mds(x, y, n_mds=N_MDS):
    print(f"\n[MDS] 3D 투영 중... (샘플 {n_mds}개, 시간 소요 예상)")
    # MDS는 O(n²) 메모리/시간 → 서브샘플
    rng = np.random.default_rng(RANDOM_SEED)
    per_class = n_mds // 10
    idx = []
    for c in range(10):
        ci = np.where(y == c)[0]
        idx.append(rng.choice(ci, size=min(per_class, len(ci)), replace=False))
    idx = np.concatenate(idx)

    x_sub = x[idx]
    y_sub = y[idx]

    # PCA로 먼저 50차원 축소 → MDS 속도 개선
    pca_pre = PCA(n_components=50, random_state=RANDOM_SEED)
    x_pre = pca_pre.fit_transform(x_sub)

    t0 = time.time()
    mds = MDS(
        n_components=3,
        metric_mds=True,
        init="random",
        n_init=1,
        max_iter=300,
        random_state=RANDOM_SEED,
        normalized_stress="auto",
        n_jobs=-1,
    )
    coords = mds.fit_transform(x_pre)
    elapsed = time.time() - t0
    stress = mds.stress_
    print(f"  완료 ({elapsed:.1f}s)  |  Stress: {stress:.4f}")
    return coords, y_sub, {
        "stress": stress,
        "elapsed": elapsed,
        "n_samples": len(idx),
    }


# ─────────────────────────────────────────
# 3. Plotly 서브플롯 구성
# ─────────────────────────────────────────
def add_scatter3d_traces(fig, coords, labels, row, col, show_legend):
    """한 subplot에 10개 클래스 scatter3d 트레이스 추가"""
    for digit in range(10):
        mask = labels == digit
        fig.add_trace(
            go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode="markers",
                name=f"숫자 {digit}",
                legendgroup=f"digit_{digit}",
                showlegend=show_legend,
                marker=dict(
                    size=2.5,
                    color=COLORS[digit],
                    opacity=0.75,
                    line=dict(width=0),
                ),
                hovertemplate=(
                    f"<b>클래스 {digit}</b><br>"
                    "x: %{x:.3f}<br>"
                    "y: %{y:.3f}<br>"
                    "z: %{z:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=row, col=col,
        )


def build_html(pca_coords, pca_labels, pca_info,
               lda_coords, lda_labels, lda_info,
               mds_coords, mds_labels, mds_info,
               output_path):
    print("\n인터랙티브 3D 그래프 생성 중...")

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=[
            f"PCA  (설명분산 {pca_info['total_variance']*100:.1f}%,  {pca_info['elapsed']:.1f}s)",
            f"LDA  (설명분산 {lda_info['total_variance']*100:.1f}%,  {lda_info['elapsed']:.1f}s)",
            f"MDS  (Stress={mds_info['stress']:.3f},  n={mds_info['n_samples']},  {mds_info['elapsed']:.1f}s)",
        ],
        horizontal_spacing=0.02,
    )

    axis_style = dict(
        backgroundcolor="#F0F2F5",
        gridcolor="white",
        showticklabels=False,
    )
    scene_defaults = dict(
        xaxis=dict(title="", **axis_style),
        yaxis=dict(title="", **axis_style),
        zaxis=dict(title="", **axis_style),
        bgcolor="#F0F2F5",
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
    )

    add_scatter3d_traces(fig, pca_coords, pca_labels, row=1, col=1, show_legend=True)
    add_scatter3d_traces(fig, lda_coords, lda_labels, row=1, col=2, show_legend=False)
    add_scatter3d_traces(fig, mds_coords, mds_labels, row=1, col=3, show_legend=False)

    fig.update_layout(
        title=dict(
            text=(
                "MNIST 차원축소 비교: PCA  vs  LDA  vs  MDS  (3D Interactive)<br>"
                "<sup>마우스 드래그: 회전 · 스크롤: 줌 · 호버: 값 확인 · 범례 클릭: 클래스 토글</sup>"
            ),
            x=0.5,
            font=dict(size=18),
        ),
        scene=scene_defaults,
        scene2=scene_defaults,
        scene3=scene_defaults,
        legend=dict(
            title=dict(text="숫자 클래스", font=dict(size=13)),
            itemsizing="constant",
            font=dict(size=12),
            x=1.01,
            y=0.5,
        ),
        width=1600,
        height=700,
        margin=dict(l=0, r=120, t=100, b=0),
        paper_bgcolor="white",
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"\n✅ 저장 완료: {output_path}")
    print("   → 브라우저에서 파일을 열면 각 subplot을 독립적으로 회전/줌할 수 있습니다.")


# ─────────────────────────────────────────
# 4. 메인
# ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("  MNIST 차원축소 3D 시각화: PCA / LDA / MDS")
    print("=" * 60)

    x, y = load_mnist(n_samples=N_SAMPLES)

    pca_coords, pca_info          = apply_pca(x, y)
    lda_coords, lda_info          = apply_lda(x, y)
    mds_coords, mds_labels, mds_info = apply_mds(x, y, n_mds=N_MDS)

    # PCA / LDA는 전체 N_SAMPLES, MDS는 서브샘플
    build_html(
        pca_coords, y,        pca_info,
        lda_coords, y,        lda_info,
        mds_coords, mds_labels, mds_info,
        output_path=OUTPUT_HTML,
    )

    print("\n" + "=" * 60)
    print(f"  완료!  {OUTPUT_HTML}  을 브라우저로 열어보세요.")
    print("=" * 60)


if __name__ == "__main__":
    main()
