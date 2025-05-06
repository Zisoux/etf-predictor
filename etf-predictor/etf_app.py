import streamlit as st
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier  # 반드시 있어야 함

# ✅ 모델 로드
model_path = os.path.join(os.path.dirname(__file__), "etf_rf_model.pkl")
model = joblib.load(model_path)

# ✅ 특성 이름 (훈련할 때 사용한 열)
feature_names = [
    "SPY_Open", "SPY_High", "SPY_Low", "SPY_Close", "SPY_Volume",
    "QQQ_Open", "QQQ_High", "QQQ_Low", "QQQ_Close", "QQQ_Volume",
    "SPY_return", "QQQ_return"
]

# ✅ 예시 값 (사용자가 참고할 수 있도록 미리 채워줌)
example_values = [
    420.5, 425.1, 419.2, 423.8, 75000000,      # SPY
    340.3, 342.9, 338.8, 341.6, 62000000,      # QQQ
    0.0021, 0.0017                             # 수익률
]

# ✅ 앱 UI 시작
st.title("📈 S&P 500 ETF 상승 예측기")
st.markdown("""
이 앱은 SPY(미국 S&P 500 ETF)의 다음 날 **상승 여부**를 예측합니다.  
QQQ(나스닥 100 ETF)와 함께 최근 가격 데이터를 기반으로 작동합니다.

아래에 **현재 또는 예측하고 싶은 날의 데이터**를 입력해 주세요.
""")

# ✅ 사용자 입력 받기
user_inputs = []
for i, name in enumerate(feature_names):
    label = f"{name} ({'가격' if 'Open' in name or 'High' in name or 'Low' in name or 'Close' in name else '수량' if 'Volume' in name else '수익률'})"
    default = float(example_values[i])
    value = st.number_input(label, value=default)
    user_inputs.append(value)

# ✅ 예측 실행
if st.button("📊 예측 실행"):
    input_array = np.array(user_inputs).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.subheader("📍 예측 결과")
    st.success("✅ 내일 SPY는 **상승할 가능성이 높습니다!** 📈" if prediction == 1 else "⚠ 내일 SPY는 **하락할 가능성이 높습니다.** 📉")
