import streamlit as st
import numpy as np
import joblib

# 저장된 모델 불러오기
model = joblib.load("etf_rf_model.pkl") 

st.title("📈 ETF 수익률 예측기 (S&P 500 기반)")

st.markdown("""
이 앱은 SPY와 QQQ ETF의 과거 데이터를 기반으로  
**S&P 500 ETF(SPY)**가 다음 날 상승할지를 예측합니다.
""")

# 사용자 입력
def user_input_features():
    inputs = []
    for i in range(model.n_features_in_):
        val = st.number_input(f"입력 특성 {i+1}", value=0.0)
        inputs.append(val)
    return np.array(inputs).reshape(1, -1)

features = user_input_features()

if st.button("📊 예측 실행"):
    result = model.predict(features)
    st.write("✅ 예측 결과:", "상승 📈" if result[0] == 1 else "하락 📉")
