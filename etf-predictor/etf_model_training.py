import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import zscore
import joblib
import warnings
warnings.filterwarnings("ignore")

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
df = pd.concat([spy, qqq], axis=1)

# ✅ 3. 결측치 확인 및 제거
print("\n[결측치 개수]")
print(df.isnull().sum())
df.dropna(inplace=True)

# ✅ 4. 수익률 및 타겟
df['SPY_return'] = df['SPY_Close'].pct_change()
df['QQQ_return'] = df['QQQ_Close'].pct_change()
df['Target'] = (df['SPY_return'].shift(-1) > 0).astype(int)
df.dropna(inplace=True)

# ✅ 5. 이상치 제거 (거래량 기준 z-score)
for col in ['SPY_Volume', 'QQQ_Volume']:
    df[f'{col}_z'] = zscore(df[col])
df = df[(df['SPY_Volume_z'].abs() < 3) & (df['QQQ_Volume_z'].abs() < 3)]
df.drop(columns=['SPY_Volume_z', 'QQQ_Volume_z'], inplace=True)

# ✅ 6. 상관관계 시각화
cor = df[['SPY_return', 'QQQ_return', 'SPY_Close', 'QQQ_Close']].corr()
sns.heatmap(cor, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("eda_corr_heatmap.png")
plt.close()

# ✅ 7. 특성 선택
X = df[['SPY_Close', 'SPY_Volume', 'QQQ_Close', 'SPY_return', 'QQQ_return']]
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# ✅ 8. 모델별 하이퍼파라미터 튜닝 및 평가
results = {}

# Logistic Regression
logit = LogisticRegression(max_iter=500)
logit_params = {'C': [0.01, 0.1, 1, 10]}
logit_grid = GridSearchCV(logit, logit_params, cv=3)
logit_grid.fit(X_train, y_train)
logit_best = logit_grid.best_estimator_
logit_acc = accuracy_score(y_test, logit_best.predict(X_test))
results['Logistic Regression'] = (logit_best, logit_acc)
print("\n📌 Logistic Regression 결과:")
print(classification_report(y_test, logit_best.predict(X_test)))

# Random Forest
rf = RandomForestClassifier()
rf_params = {'n_estimators': [100, 150], 'max_depth': [5, 10, 15]}
rf_grid = GridSearchCV(rf, rf_params, cv=3)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
rf_acc = accuracy_score(y_test, rf_best.predict(X_test))
results['Random Forest'] = (rf_best, rf_acc)
print("\n📌 Random Forest 결과:")
print(classification_report(y_test, rf_best.predict(X_test)))

# Gradient Boosting
gb = GradientBoostingClassifier()
gb_params = {'n_estimators': [100, 150], 'learning_rate': [0.01, 0.1]}
gb_grid = GridSearchCV(gb, gb_params, cv=3)
gb_grid.fit(X_train, y_train)
gb_best = gb_grid.best_estimator_
gb_acc = accuracy_score(y_test, gb_best.predict(X_test))
results['Gradient Boosting'] = (gb_best, gb_acc)
print("\n📌 Gradient Boosting 결과:")
print(classification_report(y_test, gb_best.predict(X_test)))

# ✅ 9. 최적 모델 선택 및 저장
best_model_name, (best_model, best_acc) = max(results.items(), key=lambda x: x[1][1])
joblib.dump(best_model, "etf_rf_model.pkl")
print(f"\n✅ 최종 선택 모델: {best_model_name} (정확도: {best_acc:.4f}) 저장 완료")
