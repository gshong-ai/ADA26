# ADA26 코딩 지침

이 저장소의 모든 코드는 **수업용 예제**입니다. 아래 지침을 항상 준수하세요.

---

## 1. 파일 형식

- **메인 코드는 Jupyter Notebook(`.ipynb`)으로 작성**합니다.
- 노트북은 셀 단위로 설명(Markdown) + 코드가 교차되도록 구성합니다.
- 보조 유틸리티(공통 함수 등)는 `.py`로 작성해도 무방합니다.

## 2. 딥러닝 모델

- **항상 Keras Functional API**를 사용합니다. Sequential API 사용 금지.

```python
# ✅ Keras Functional API
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# ❌ Sequential API 사용 금지
model = keras.Sequential([...])
```

## 3. 노트북 구성 순서

노트북은 아래 순서로 셀을 구성합니다:

1. **제목 & 개요** (Markdown) — 이 예제가 무엇을 다루는지
2. **라이브러리 임포트**
3. **하이퍼파라미터 & 설정** — 상단에 한 곳에 모아서 정의
4. **데이터 로드 & 전처리**
5. **모델 정의** (Keras Functional API)
6. **학습**
7. **평가 & 시각화**

## 4. 폴더 구조

- 주제별로 번호가 붙은 폴더로 분리합니다: `01_주제명/`, `02_주제명/`, ...
- 각 폴더에는 `README.md`를 작성합니다.
- 데이터는 해당 폴더 내 `data/`에 보관합니다.

## 5. 시각화

- 인터랙티브 시각화는 **Plotly**를 사용합니다.
- 정적 시각화는 **Matplotlib / Seaborn**을 사용합니다.
- 결과 HTML은 노트북과 같은 폴더에 저장합니다.

## 6. 코드 스타일

- 변수명·함수명은 **snake_case**, 상수는 **UPPER_CASE**
- 주석과 출력 메시지는 **한국어**로 작성합니다.
- 각 주요 블록은 구분선(`# ─── 제목 ───`)으로 명확히 나눕니다.
