"""
MNIST 3D Manifold 시각화
- 병목 레이어(3노드)를 통해 10개 클래스가 3차원 공간에 어떻게 매핑되는지 시각화
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import plotly.graph_objects as go
import numpy as np
import os

# ─────────────────────────────────────────
# 1. 하이퍼파라미터 및 설정
# ─────────────────────────────────────────
BATCH_SIZE   = 256
EPOCHS       = 15
LR           = 1e-3
BOTTLENECK   = 3       # 3D manifold 병목 노드 수
VIZ_SAMPLES  = 5000   # 시각화에 사용할 테스트 샘플 수
DATA_DIR     = "./data"
OUTPUT_HTML  = "mnist_manifold_3d.html"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {DEVICE}")


# ─────────────────────────────────────────
# 2. 데이터 로드
# ─────────────────────────────────────────
def get_dataloaders():
    """MNIST 데이터셋을 로드하고 DataLoader 반환"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 평균/표준편차
    ])

    train_dataset = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"학습 샘플: {len(train_dataset):,}  /  테스트 샘플: {len(test_dataset):,}")
    return train_loader, test_loader


# ─────────────────────────────────────────
# 3. 모델 정의
# ─────────────────────────────────────────
class MNISTManifoldNet(nn.Module):
    """
    병목 레이어(3노드)를 포함한 MNIST 분류 네트워크.

    구조: 784 → 512 → 256 → 128 → [3] → 10
                                    ↑
                              3D manifold 병목
    """
    def __init__(self, bottleneck_dim=3):
        super().__init__()

        # 인코더: 입력 → 3차원 병목
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, bottleneck_dim),  # ← 3D 병목 레이어
        )

        # 분류기: 3차원 표현 → 10개 클래스
        self.classifier = nn.Linear(bottleneck_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)       # (N, 1, 28, 28) → (N, 784) 평탄화
        features = self.encoder(x)       # (N, 3) — 3D manifold 좌표
        logits   = self.classifier(features)  # (N, 10) — 클래스 점수
        return logits, features


# ─────────────────────────────────────────
# 4. 학습 루프
# ─────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    """한 에폭 학습 수행"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    """테스트셋 평가"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits, _ = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────
# 5. 3D 특징 추출
# ─────────────────────────────────────────
def extract_features(model, loader, max_samples=5000):
    """
    병목 레이어(3노드)의 출력을 추출.
    반환: features (N, 3), labels (N,)
    """
    model.eval()
    all_features, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            _, features = model(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

            # 최대 샘플 수 초과 시 중단
            if sum(len(f) for f in all_features) >= max_samples:
                break

    features = np.concatenate(all_features, axis=0)[:max_samples]
    labels   = np.concatenate(all_labels,   axis=0)[:max_samples]

    # ✅ 검증 포인트 1: shape 확인
    assert features.shape[1] == 3, f"병목 출력 shape 오류: {features.shape}"
    print(f"추출된 특징 shape: {features.shape}  (기대값: ({max_samples}, 3))")

    return features, labels


# ─────────────────────────────────────────
# 6. 인터랙티브 3D 시각화 (Plotly)
# ─────────────────────────────────────────
def visualize_3d_manifold(features, labels, accuracy, output_path):
    """
    10개 클래스의 3D 분포를 인터랙티브 산점도로 시각화.
    마우스로 회전/줌/호버 가능.
    """
    # 클래스별 색상 팔레트 (선명하게 구분)
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

    fig = go.Figure()

    for digit in range(10):
        mask = labels == digit
        x = features[mask, 0]
        y = features[mask, 1]
        z = features[mask, 2]

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            name=f"숫자 {digit}",
            marker=dict(
                size=2.5,
                color=COLORS[digit],
                opacity=0.7,
                line=dict(width=0),
            ),
            hovertemplate=(
                f"<b>클래스: {digit}</b><br>"
                "x: %{x:.3f}<br>"
                "y: %{y:.3f}<br>"
                "z: %{z:.3f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text=(
                f"MNIST 3D Manifold 시각화<br>"
                f"<sup>병목 레이어 3노드 → 3차원 공간 매핑 "
                f"(테스트 정확도: {accuracy*100:.2f}%)</sup>"
            ),
            x=0.5,
            font=dict(size=18),
        ),
        scene=dict(
            xaxis=dict(title="차원 1", backgroundcolor="#F8F9FA", gridcolor="white"),
            yaxis=dict(title="차원 2", backgroundcolor="#F8F9FA", gridcolor="white"),
            zaxis=dict(title="차원 3", backgroundcolor="#F8F9FA", gridcolor="white"),
            bgcolor="#F0F2F5",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        legend=dict(
            title="숫자 클래스",
            itemsizing="constant",
            font=dict(size=13),
        ),
        width=1000,
        height=750,
        margin=dict(l=0, r=0, t=80, b=0),
        paper_bgcolor="white",
    )

    # HTML로 저장 (브라우저에서 인터랙티브하게 열기 가능)
    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"\n✅ 인터랙티브 3D 그래프 저장 완료: {output_path}")
    print("   → 브라우저에서 파일을 열면 마우스로 회전/줌/호버 가능합니다.")

    return fig


# ─────────────────────────────────────────
# 7. 메인 실행
# ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("  MNIST 3D Manifold 시각화 시작")
    print("=" * 55)

    # 데이터 로드
    train_loader, test_loader = get_dataloaders()

    # 모델 / 손실함수 / 옵티마이저 초기화
    model     = MNISTManifoldNet(bottleneck_dim=BOTTLENECK).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n모델 파라미터 수: {total_params:,}")
    print(f"병목 레이어 차원: {BOTTLENECK}D\n")

    # ── 학습 루프 ──
    print(f"{'에폭':>5} | {'학습 손실':>10} | {'학습 정확도':>10} | {'테스트 손실':>11} | {'테스트 정확도':>12}")
    print("-" * 60)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion)
        scheduler.step()

        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc*100:>9.2f}% | {test_loss:>11.4f} | {test_acc*100:>11.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pt")

    print(f"\n✅ 학습 완료 — 최고 테스트 정확도: {best_acc*100:.2f}%")

    # ✅ 검증 포인트 2: 정확도 출력
    print(f"   (3노드 병목 제약에도 불구하고 {best_acc*100:.1f}% 달성)")

    # 최적 모델 로드 후 특징 추출
    model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))

    print(f"\n병목 레이어 3D 특징 추출 중 ({VIZ_SAMPLES:,}개 샘플)...")
    features, labels = extract_features(model, test_loader, max_samples=VIZ_SAMPLES)

    # ✅ 검증 포인트 3: 클래스별 분포 확인
    print("\n클래스별 샘플 수:")
    for d in range(10):
        print(f"  숫자 {d}: {(labels == d).sum():>5}개")

    # 인터랙티브 3D 시각화
    print("\n인터랙티브 3D 그래프 생성 중...")
    visualize_3d_manifold(features, labels, best_acc, OUTPUT_HTML)

    print("\n" + "=" * 55)
    print("  완료! mnist_manifold_3d.html 을 브라우저로 열어보세요.")
    print("=" * 55)


if __name__ == "__main__":
    main()
