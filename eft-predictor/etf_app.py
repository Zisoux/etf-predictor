import streamlit as st
import numpy as np
import joblib

# 모델 불러오기
model = joblib.load("etf_rf_model.pkl")

st.title("📈 ETF 수익률 예측기 (S&P 500 기반)")

st.markdown("""
이 앱은 SPY와 QQQ ETF의 과거 데이터를 기반으로  
**S&P 500 ETF(SPY)**가 다음 날 상승할지를 예측합니다.
""")

# 입력값 생성 함수
def user_input_features():
    inputs = []
    for i in range(model.n_features_in_):
        val = st.number_input(f"입력 특성 {i+1}", value=0.0)
        inputs.append(val)
    return np.array(inputs).reshape(1, -1)

# 입력 받기
user_input = user_input_features()

# 예측 버튼
if st.button("📊 예측 실행"):
    prediction = model.predict(user_input)
    result = "📈 상승할 가능성이 높습니다." if prediction[0] == 1 else "📉 하락할 가능성이 높습니다."
    st.success(result)
