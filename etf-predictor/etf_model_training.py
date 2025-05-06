import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 🟢 1. 데이터 다운로드
spy = yf.download("SPY", start="2018-01-01", end="2025-04-30", auto_adjust=False)
qqq = yf.download("QQQ", start="2018-01-01", end="2025-04-30", auto_adjust=False)

if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.droplevel(1)
if isinstance(qqq.columns, pd.MultiIndex):
    qqq.columns = qqq.columns.droplevel(1)

# 🟢 2. 열 필터링 및 접두사 추가
def prepare(df, name):
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = [f"{name}_{col}" for col in df.columns]
    return df

spy = prepare(spy, "SPY")
qqq = prepare(qqq, "QQQ")
df = pd.concat([spy, qqq], axis=1).dropna()

# 🟢 3. 수익률 및 타겟
df['SPY_return'] = df['SPY_Close'].pct_change()
df['QQQ_return'] = df['QQQ_Close'].pct_change()
df['Target'] = (df['SPY_return'].shift(-1) > 0).astype(int)
df.dropna(inplace=True)

# ✅ 4. EDA (시각화는 파일로 저장)
cor = df[['SPY_return', 'QQQ_return', 'SPY_Close', 'QQQ_Close']].corr()
sns.heatmap(cor, annot=True, cmap='coolwarm')
plt.title("📊 상관관계 히트맵")
plt.savefig("eda_corr_heatmap.png")
plt.close()

# ✅ 5. 특성 선택
X = df[['SPY_Close', 'SPY_Volume', 'QQQ_Close', 'SPY_return', 'QQQ_return']]
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# ✅ 6. 모델 비교
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc))
    print(f"\n📌 {name} 결과:")
    print(classification_report(y_test, y_pred))

# ✅ 7. 가장 정확도가 높은 모델 저장
best_model_name, best_acc = max(results, key=lambda x: x[1])
final_model = models[best_model_name]
joblib.dump(final_model, "etf_rf_model.pkl")
print(f"\n✅ 최종 선택 모델: {best_model_name} (정확도: {best_acc:.4f}) 저장 완료")
