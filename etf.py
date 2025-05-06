# 필요한 라이브러리 불러오기
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1. 데이터 수집: SPY (S&P 500), QQQ (NASDAQ 100) ETF 가격 데이터 다운로드
spy = yf.download("SPY", start="2018-01-01", end="2025-4-30")
qqq = yf.download("QQQ", start="2018-01-01", end="2025-4-30")

# 2. 필요한 컬럼만 선택하고 접두사 붙이기
def prepare_data(df, name):
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = [f"{name}_{col}" for col in df.columns]
    return df

spy = prepare_data(spy, 'SPY')
qqq = prepare_data(qqq, 'QQQ')

# 날짜 기준으로 병합, 결측치 제거
df = pd.concat([spy, qqq], axis=1).dropna()

# 3. 기술적 지표(수익률) 추가 및 라벨 생성
df['SPY_return'] = df['SPY_Close'].pct_change()
df['QQQ_return'] = df['QQQ_Close'].pct_change()

# 다음 날 수익률이 양수면 1, 음수 또는 0이면 0으로 라벨링
df['Target'] = (df['SPY_return'].shift(-1) > 0).astype(int)

# 결측치 제거 (수익률 계산 때문에 생긴 NaN 포함)
df.dropna(inplace=True)

# 4. 학습을 위한 데이터 분리
features = df.drop(columns=['Target'])  # 입력 변수
target = df['Target']  # 예측할 값

# 시계열 데이터이므로 섞지 않고 분할 (최근 데이터를 테스트로)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

# 5. 모델 훈련 - RandomForestClassifier + GridSearchCV로 하이퍼파라미터 최적화
params = {'n_estimators': [100, 200], 'max_depth': [5, 10]}  # 실험할 파라미터 목록
model = RandomForestClassifier(random_state=42)
grid = GridSearchCV(model, param_grid=params, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

# 6. 학습된 모델 저장 (Streamlit 웹앱에서 불러올 수 있도록)
joblib.dump(grid.best_estimator_, "etf_rf_model.pkl")

# 7. 테스트 결과 확인 (성능 평가 지표 출력)
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))

