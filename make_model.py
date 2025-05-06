import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# 🟢 데이터 다운로드
spy = yf.download("SPY", start="2018-01-01", end="2025-04-30", auto_adjust=False)
qqq = yf.download("QQQ", start="2018-01-01", end="2025-04-30", auto_adjust=False)

# 🟢 MultiIndex 열 제거
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.droplevel(1)
if isinstance(qqq.columns, pd.MultiIndex):
    qqq.columns = qqq.columns.droplevel(1)

# 🟢 열 이름 접두사 추가
def prepare(df, name):
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = [f"{name}_{col}" for col in df.columns]
    return df

spy = prepare(spy, "SPY")
qqq = prepare(qqq, "QQQ")

# 🟢 병합 및 수익률 계산
df = pd.concat([spy, qqq], axis=1).dropna()
df['SPY_return'] = df['SPY_Close'].pct_change()
df['QQQ_return'] = df['QQQ_Close'].pct_change()
df['Target'] = (df['SPY_return'].shift(-1) > 0).astype(int)
df.dropna(inplace=True)

# ✅ 핵심 특성만 선택
X = df[['SPY_Close', 'SPY_Volume', 'QQQ_Close', 'SPY_return', 'QQQ_return']]
y = df["Target"]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

# 🟢 모델 학습 및 저장
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "etf_rf_model.pkl")
print("✅ etf_rf_model.pkl 저장 완료")
