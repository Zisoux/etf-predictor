import yfinance as yf

spy = yf.download("SPY", start="2023-01-01", end="2023-12-31", auto_adjust=False)
print("spy.columns:", spy.columns.tolist())

def prepare(df, name):
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = [f"{name}_{col}" for col in df.columns]
    return df

spy_prepared = prepare(spy, "SPY")
print("spy_prepared.columns:", spy_prepared.columns.tolist())
