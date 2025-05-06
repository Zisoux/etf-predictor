import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# 🟢 1. 데이터 다운로드 (주의: 날짜 형식)
spy = yf.download("SPY", start="2018-01-01", end="2025-04-30", auto_adjust=False)
qqq = yf.download("QQQ", start="2018-01-01", end="2025-04-30", auto_adjust=False)

# 🟢 2. 열이 MultiIndex면 평탄화 처리 (핵심!)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.droplevel(1)
if isinstance(qqq.columns, pd.MultiIndex):
    qqq.columns = qqq.columns.droplevel(1)

# 🟢 3. 필요한 열만 추출하고 접두사 추가
def prepare(df, name):
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = [f"{name}_{col}" for col in df.columns]
    return df

spy = prepare(spy, "SPY")
qqq = prepare(qqq, "QQQ")

# 🟢 4. 데이터 결합 및 타겟 생성
df = pd.concat([spy, qqq], axis=1).dropna()
df['SPY_return'] = df['SPY_Close'].pct_change()
df['QQQ_return'] = df['QQQ_Close'].pct_change()
df['Target'] = (df['SPY_return'].shift(-1) > 0).astype(int)
df.dropna(inplace=True)

# 🟢 5. 학습 준비 및 모델 학습
X = df.drop(columns=["Target"])
y = df["Target"]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 🟢 6. 모델 저장
joblib.dump(model, "etf_rf_model.pkl")
print("✅ etf_rf_model.pkl 저장 완료")
