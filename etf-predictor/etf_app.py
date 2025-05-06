import streamlit as st
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

# ✅ 모델 불러오기
model_path = os.path.join(os.path.dirname(__file__), "etf_rf_model.pkl")
model = joblib.load(model_path)

# ✅ 특성 정의 (5개로 축소)
feature_names = ['SPY_Close', 'SPY_Volume', 'QQQ_Close', 'SPY_return', 'QQQ_return']
example_values = [423.8, 75000000, 341.6, 0.0021, 0.0017]

# ✅ 앱 UI 시작
st.title("📈 S&P 500 ETF 상승 예측기 (간소화 버전)")

st.markdown("""
### 📌 이 앱은 무엇을 하나요?
**미국 S&P 500 ETF(SPY)**의 **다음 날 주가가 오를지 내릴지를 예측**합니다.

### ✅ 입력값 설명
| 입력값 | 의미 |
|--------|------|
| `SPY_Close` | 오늘의 S&P 500 ETF 종가 |
| `SPY_Volume` | 오늘의 SPY 거래량 |
| `QQQ_Close` | 오늘의 나스닥 100 ETF 종가 |
| `SPY_return` | 오늘 SPY의 전일 대비 수익률 (예: 0.01 = +1%) |
| `QQQ_return` | 오늘 QQQ의 전일 대비 수익률 (예: -0.005 = -0.5%) |

→ 이 5개의 값을 바탕으로, **내일 SPY가 상승할 확률이 높은지 판단**해줍니다.
""", unsafe_allow_html=True)

# ✅ 사용자 입력
user_inputs = []
for i, name in enumerate(feature_names):
    label = f"{name} ({'가격' if 'Close' in name else '거래량' if 'Volume' in name else '수익률'})"
    value = st.number_input(label, value=float(example_values[i]))
    user_inputs.append(value)

# ✅ 예측 버튼
if st.button("📊 예측 실행"):
    input_array = np.array(user_inputs).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.subheader("📍 예측 결과")
    st.success("✅ 내일 SPY는 **상승할 가능성이 높습니다!** 📈" if prediction == 1 else "⚠ 내일 SPY는 **하락할 가능성이 높습니다.** 📉")
