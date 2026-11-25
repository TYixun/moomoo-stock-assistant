"""
ml_signal.py - walk-forward machine learning signal for a daily price series.

What it does
- Builds features from past prices only (returns, moving-average distance, RSI, volatility, volume).
- Target: is the price higher N trading days later?
- Trains on the past and predicts the next block, over and over (walk-forward), so every test
  prediction comes from a model that had never seen that day.
- Compares the results with a "just guess the usual direction" baseline and with buy-and-hold.

This is a research tool. Most such models show little or no edge once fees are counted.
Not financial advice.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = ["Logistic regression", "Gradient boosting"]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Every column uses data up to and including that day's close, nothing later."""
    c = df["Close"]
    X = pd.DataFrame(index=df.index)
    for n in (1, 5, 10, 20):
        X[f"ret_{n}d"] = c.pct_change(n)
    for n in (10, 50, 200):
        X[f"vs_sma{n}"] = c / c.rolling(n).mean() - 1
    X["rsi14"] = _rsi(c)
    X["vol_20d"] = c.pct_change().rolling(20).std()
    if "Volume" in df:
        X["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean() - 1
    if {"High", "Low"} <= set(df.columns):
        X["day_range"] = (df["High"] - df["Low"]) / c
    X["from_52w_high"] = c / c.rolling(252).max() - 1
    return X.replace([np.inf, -np.inf], np.nan)


def make_model(name: str):
    if name == "Logistic regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=40, random_state=0
    )


def walk_forward(df, horizon=5, model_name="Logistic regression", min_train=756, step=63, rolling=False):
    """Out-of-sample probability of a rise for every day after the first `min_train` days.

    min_train is how many past days the model learns from. With rolling=True it only ever sees the
    most recent `min_train` days; otherwise it keeps all days so far and `min_train` is the minimum."""
    if min_train < 100:
        raise ValueError("The model needs to learn from at least 100 days.")
    X = build_features(df)
    fwd = df["Close"].shift(-horizon) / df["Close"] - 1          # return over the next N days
    nxt = df["Close"].pct_change().shift(-1)                     # next-day return, for the backtest
    d = X.assign(fwd=fwd, nxt=nxt).dropna()
    feats = list(X.columns)
    y = (d["fwd"] > 0).astype(int).to_numpy()
    n = len(d)
    if n < min_train + step:
        raise ValueError(f"Not enough history: need about {min_train + step} usable days, have {n}.")

    probs = np.full(n, np.nan)
    for i in range(min_train, n, step):
        # A training label at row j is only known once row j+horizon has happened,
        # so training stops `horizon` rows before the first day being predicted.
        start = max(0, i - horizon - min_train) if rolling else 0
        tr = slice(start, i - horizon)
        if len(set(y[tr])) < 2:
            continue
        model = make_model(model_name).fit(d[feats].iloc[tr], y[tr])
        j = min(i + step, n)
        probs[i:j] = model.predict_proba(d[feats].iloc[i:j])[:, 1]

    out = d[["fwd", "nxt"]].copy()
    out["up"] = y
    out["prob"] = probs
    return out.dropna(subset=["prob"])


def evaluate(oos: pd.DataFrame, buy_thr: float, fee: float) -> dict:
    p, up = oos["prob"], oos["up"]
    accuracy = float(((p >= 0.5).astype(int) == up).mean())
    up_rate = float(up.mean())
    baseline = max(up_rate, 1 - up_rate)
    try:
        auc = float(roc_auc_score(up, p))
    except ValueError:
        auc = None

    buys = p >= buy_thr
    n_buy = int(buys.sum())
    hit_rate = float(up[buys].mean()) if n_buy else None

    pos = buys.astype(int)
    trades = pos.diff().abs()
    trades.iloc[0] = pos.iloc[0]
    strat = pos * oos["nxt"] - trades * fee
    held = oos["nxt"].copy()
    held.iloc[0] -= fee
    eq, bh = (1 + strat).cumprod(), (1 + held).cumprod()

    def stats(curve):
        return {
            "Total return (%)": round((curve.iloc[-1] - 1) * 100, 1),
            "Max drawdown (%)": round(((curve / curve.cummax()) - 1).min() * 100, 1),
        }

    s, b = stats(eq), stats(bh)
    s["Buys / sells"] = int(trades.sum())
    s["Time invested (%)"] = round(float(pos.mean()) * 100)
    b["Buys / sells"] = 1
    b["Time invested (%)"] = 100

    has_edge = bool(
        auc is not None and auc >= 0.53 and hit_rate is not None and n_buy >= 30
        and hit_rate >= up_rate + 0.03
    )
    return {
        "accuracy": accuracy, "baseline": baseline, "up_rate": up_rate, "auc": auc,
        "n_days": int(len(oos)), "n_buy": n_buy, "hit_rate": hit_rate,
        "equity": eq, "buy_hold": bh, "strategy_stats": s, "buy_hold_stats": b,
        "has_edge": has_edge,
        "start": oos.index[0], "end": oos.index[-1],
    }


def latest_signal(df, horizon=5, model_name="Logistic regression", train_days=None):
    """Train on days whose outcome is already known (the most recent `train_days` if given),
    then score the latest day."""
    X = build_features(df)
    fwd = df["Close"].shift(-horizon) / df["Close"] - 1
    d = X.assign(fwd=fwd).dropna()
    if train_days:
        d = d.iloc[-int(train_days):]
    feats = list(X.columns)
    model = make_model(model_name).fit(d[feats], (d["fwd"] > 0).astype(int))
    last = X.dropna().iloc[[-1]]
    return float(model.predict_proba(last[feats])[0, 1]), last.index[0]


def run(df, horizon, model_name, buy_thr, sell_thr, fee, train_days=756, rolling=False):
    if df is None or df.empty or "Close" not in df:
        raise ValueError("No price history came back. Yahoo may be rate-limiting, or the ticker may be wrong.")
    oos = walk_forward(df, horizon, model_name, min_train=int(train_days), rolling=rolling)
    res = evaluate(oos, buy_thr, fee)
    prob, asof = latest_signal(df, horizon, model_name, int(train_days) if rolling else None)
    res.update({"prob": prob, "asof": str(asof)[:10], "horizon": horizon, "model": model_name,
                "buy_thr": buy_thr, "sell_thr": sell_thr, "train_days": int(train_days), "rolling": bool(rolling)})
    res["label"] = "BUY" if prob >= buy_thr else "SELL" if prob <= sell_thr else "HOLD"
    return res