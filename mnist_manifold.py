"""
MNIST 3D Manifold 시각화
- 병목 레이어(3노드)를 통해 10개 클래스가 3차원 공간에 어떻게 매핑되는지 시각화
- 모델: Keras Functional API
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import plotly.graph_objects as go

# ─────────────────────────────────────────
# 1. 하이퍼파라미터 및 설정
# ─────────────────────────────────────────
BATCH_SIZE   = 256
EPOCHS       = 15
LR           = 1e-3
BOTTLENECK   = 3       # 3D manifold 병목 노드 수
VIZ_SAMPLES  = 5000   # 시각화에 사용할 테스트 샘플 수
OUTPUT_HTML  = "mnist_manifold_3d.html"
MODEL_PATH   = "best_model.keras"

print(f"TensorFlow 버전: {tf.__version__}")
print(f"GPU 사용 가능: {len(tf.config.list_physical_devices('GPU')) > 0}")


# ─────────────────────────────────────────
# 2. 데이터 로드
# ─────────────────────────────────────────
def get_datasets():
    """MNIST 데이터셋을 로드하고 전처리하여 반환"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 정규화 (MNIST 평균/표준편차 적용) 및 평탄화: (N, 28, 28) → (N, 784)
    mean, std = 0.1307, 0.3081
    x_train = (x_train.reshape(-1, 784).astype("float32") / 255.0 - mean) / std
    x_test  = (x_test.reshape(-1, 784).astype("float32") / 255.0 - mean) / std

    print(f"학습 샘플: {len(x_train):,}  /  테스트 샘플: {len(x_test):,}")
    return (x_train, y_train), (x_test, y_test)


# ─────────────────────────────────────────
# 3. 모델 정의 (Keras Functional API)
# ─────────────────────────────────────────
def build_model(bottleneck_dim=3):
    """
    병목 레이어(3노드)를 포함한 MNIST 분류 네트워크.

    구조: 784 → 512 → 256 → 128 → [3] → 10
                                    ↑
                              3D manifold 병목
    반환: (전체 분류 모델, 병목 출력 추출용 서브모델)
    """
    inputs = keras.Input(shape=(784,), name="input")

    # 인코더: 입력 → 3차원 병목
    x = layers.Dense(512)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    bottleneck = layers.Dense(bottleneck_dim, name="bottleneck")(x)  # ← 3D 병목 레이어

    # 분류기: 3차원 표현 → 10개 클래스
    outputs = layers.Dense(10, activation="softmax", name="classifier")(bottleneck)

    # 전체 모델 (학습용)
    model = keras.Model(inputs=inputs, outputs=outputs, name="MNISTManifoldNet")

    # 병목 출력 추출용 서브모델 (특징 시각화용)
    feature_model = keras.Model(inputs=inputs, outputs=bottleneck, name="FeatureExtractor")

    return model, feature_model


# ─────────────────────────────────────────
# 4. 3D 특징 추출
# ─────────────────────────────────────────
def extract_features(feature_model, x_test, y_test, max_samples=5000):
    """
    병목 레이어(3노드)의 출력을 추출.
    반환: features (N, 3), labels (N,)
    """
    features = feature_model.predict(x_test[:max_samples], batch_size=BATCH_SIZE, verbose=0)
    labels   = y_test[:max_samples]

    # 검증: shape 확인
    assert features.shape[1] == 3, f"병목 출력 shape 오류: {features.shape}"
    print(f"추출된 특징 shape: {features.shape}  (기대값: ({max_samples}, 3))")

    return features, labels


# ─────────────────────────────────────────
# 5. 인터랙티브 3D 시각화 (Plotly)
# ─────────────────────────────────────────
def visualize_3d_manifold(features, labels, accuracy, output_path):
    """
    10개 클래스의 3D 분포를 인터랙티브 산점도로 시각화.
    마우스로 회전/줌/호버 가능.
    """
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
        fig.add_trace(go.Scatter3d(
            x=features[mask, 0],
            y=features[mask, 1],
            z=features[mask, 2],
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

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"\n✅ 인터랙티브 3D 그래프 저장 완료: {output_path}")
    print("   → 브라우저에서 파일을 열면 마우스로 회전/줌/호버 가능합니다.")

    return fig


# ─────────────────────────────────────────
# 6. 메인 실행
# ─────────────────────────────────────────
def main():
    print("=" * 55)
    print("  MNIST 3D Manifold 시각화 시작")
    print("=" * 55)

    # 데이터 로드
    (x_train, y_train), (x_test, y_test) = get_datasets()

    # 모델 빌드
    model, feature_model = build_model(bottleneck_dim=BOTTLENECK)

    total_params = model.count_params()
    print(f"\n모델 파라미터 수: {total_params:,}")
    print(f"병목 레이어 차원: {BOTTLENECK}D\n")
    model.summary()

    # 컴파일
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 콜백: 최적 모델 저장 + 학습률 스케줄
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
    ]

    # 학습
    print(f"\n{'─'*55}")
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    # 최적 모델 로드 후 최종 평가
    model.load_weights(MODEL_PATH)
    _, best_acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)

    print(f"\n✅ 학습 완료 — 최고 테스트 정확도: {best_acc*100:.2f}%")
    print(f"   (3노드 병목 제약에도 불구하고 {best_acc*100:.1f}% 달성)")

    # 병목 레이어 특징 추출
    print(f"\n병목 레이어 3D 특징 추출 중 ({VIZ_SAMPLES:,}개 샘플)...")
    # feature_model도 동일한 가중치를 공유하므로 별도 로드 불필요
    features, labels = extract_features(feature_model, x_test, y_test, max_samples=VIZ_SAMPLES)

    # 클래스별 분포 확인
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
