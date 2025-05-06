import streamlit as st
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier  # 불러오기용

# ✅ 모델 로드
model_path = os.path.join(os.path.dirname(__file__), "etf_rf_model.pkl")
model = joblib.load(model_path)

# ✅ 간소화된 특성 목록
feature_names = ['SPY_Close', 'SPY_Volume', 'QQQ_Close', 'SPY_return', 'QQQ_return']
example_values = [423.8, 75000000, 341.6, 0.0021, 0.0017]

# ✅ UI
st.title("📈 S&P 500 ETF 상승 예측기 (간소화 버전)")
st.markdown("""
이 앱은 SPY(미국 S&P 500 ETF)의 다음 날 상승 여부를 예측합니다.  
QQQ(나스닥 100 ETF)와 거래량, 수익률 등 간단한 데이터만 입력하면 결과를 확인할 수 있어요.
""")

# ✅ 사용자 입력
user_inputs = []
for i, name in enumerate(feature_names):
    label = f"{name} ({'가격' if 'Close' in name else '거래량' if 'Volume' in name else '수익률'})"
    value = st.number_input(label, value=float(example_values[i]))
    user_inputs.append(value)

# ✅ 예측 실행
if st.button("📊 예측 실행"):
    input_array = np.array(user_inputs).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.subheader("📍 예측 결과")
    st.success("✅ 내일 SPY는 **상승할 가능성이 높습니다!** 📈" if prediction == 1 else "⚠ 내일 SPY는 **하락할 가능성이 높습니다.** 📉")
