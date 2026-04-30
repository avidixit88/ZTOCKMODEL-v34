from __future__ import annotations

import numpy as np
import pandas as pd


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    return pv.cumsum() / df["volume"].cumsum().replace(0, np.nan)



def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rolling_swing_lows(series: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    s = series
    is_low = pd.Series(False, index=s.index)
    for i in range(left, len(s) - right):
        window = s.iloc[i - left: i + right + 1]
        if s.iloc[i] == window.min():
            is_low.iloc[i] = True
    return is_low


def rolling_swing_highs(series: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    s = series
    is_high = pd.Series(False, index=s.index)
    for i in range(left, len(s) - right):
        window = s.iloc[i - left: i + right + 1]
        if s.iloc[i] == window.max():
            is_high.iloc[i] = True
    return is_high


def detect_fvg(df: pd.DataFrame):
    if len(df) < 3:
        return None, None
    h = df["high"].values
    l = df["low"].values
    bull = None
    bear = None
    for i in range(2, len(df)):
        if l[i] > h[i - 2]:
            bull = (float(h[i - 2]), float(l[i]))
        if h[i] < l[i - 2]:
            bear = (float(h[i]), float(l[i - 2]))
    return bull, bear


def find_order_block(df: pd.DataFrame, atr_series: pd.Series, side: str = "bull", lookback: int = 30):
    if len(df) < 10:
        return None, None, None
    df = df.tail(lookback).copy()
    atr_series = atr_series.reindex(df.index).ffill()

    o = df["open"].values
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    at = atr_series.values
    idx = df.index

    if side == "bull":
        for i in range(len(df) - 4, -1, -1):
            if c[i] < o[i]:
                atr_i = at[i] if np.isfinite(at[i]) else None
                if not atr_i:
                    continue
                for j in range(i + 1, min(i + 4, len(df))):
                    if c[j] > h[i] and (c[j] - c[i]) > 1.0 * atr_i:
                        zone_low = float(l[i])
                        zone_high = float(o[i])
                        return min(zone_low, zone_high), max(zone_low, zone_high), idx[i]
    else:
        for i in range(len(df) - 4, -1, -1):
            if c[i] > o[i]:
                atr_i = at[i] if np.isfinite(at[i]) else None
                if not atr_i:
                    continue
                for j in range(i + 1, min(i + 4, len(df))):
                    if c[j] < l[i] and (c[i] - c[j]) > 1.0 * atr_i:
                        zone_high = float(h[i])
                        zone_low = float(o[i])
                        return min(zone_low, zone_high), max(zone_low, zone_high), idx[i]
    return None, None, None


def find_breaker_block(df: pd.DataFrame, atr_series: pd.Series, side: str = "bull", lookback: int = 60):
    if len(df) < 20:
        return None, None, None

    df = df.tail(lookback).copy()
    atr_series = atr_series.reindex(df.index).ffill()
    atr_last = float(atr_series.iloc[-1]) if np.isfinite(atr_series.iloc[-1]) else 0.0
    pad = 0.2 * atr_last if atr_last else 0.0

    if side == "bull":
        zl, zh, ts = find_order_block(df, atr_series, side="bear", lookback=lookback)
        if zl is None:
            return None, None, None
        if not (df["close"] > (zh + pad)).any():
            return None, None, None
        last = float(df["close"].iloc[-1])
        if (last >= (zl - pad)) and (last <= (zh + pad)):
            return float(zl), float(zh), ts
        return None, None, None
    else:
        zl, zh, ts = find_order_block(df, atr_series, side="bull", lookback=lookback)
        if zl is None:
            return None, None, None
        if not (df["close"] < (zl - pad)).any():
            return None, None, None
        last = float(df["close"].iloc[-1])
        if (last >= (zl - pad)) and (last <= (zh + pad)):
            return float(zl), float(zh), ts
        return None, None, None


def in_zone(price: float, zone_low: float, zone_high: float, buffer: float = 0.0) -> bool:
    return (price >= (zone_low - buffer)) and (price <= (zone_high + buffer))


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's recursive moving average (RMA).

    This matches the smoothing convention used by canonical ATR / DMI / ADX
    implementations more closely than a simple rolling mean while preserving a
    Pandas-Series interface aligned to the source index.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.empty:
        return pd.Series(index=s.index, dtype="float64")
    return s.ewm(alpha=1.0 / float(period), adjust=False, min_periods=period).mean()



def adx(df: pd.DataFrame, period: int = 14):
    """Average Directional Index (ADX) with DI+ and DI-.

    Phase-1 robust ADX upgrade:
    - Wilder-style recursive smoothing for TR, +DM, -DM, and DX.
    - Stricter directional-movement assignment so simultaneous up/down moves do
      not inflate trend strength.
    - Conservative zero-division handling to avoid noisy spikes in flat tape.

    Returns a tuple: (adx, plus_di, minus_di) as Series aligned to df.index.
    """
    if df is None or df.empty or len(df) < period + 2:
        empty = pd.Series(index=(df.index if df is not None else None), dtype="float64")
        return empty, empty, empty

    high = pd.to_numeric(df["high"], errors="coerce").astype(float)
    low = pd.to_numeric(df["low"], errors="coerce").astype(float)
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)

    up_move = high.diff()
    down_move = low.shift(1) - low

    plus_dm = up_move.where((up_move > 0.0) & (up_move > down_move), 0.0).fillna(0.0)
    minus_dm = down_move.where((down_move > 0.0) & (down_move > up_move), 0.0).fillna(0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_rma = _wilder_rma(tr, period)
    plus_dm_rma = _wilder_rma(plus_dm, period)
    minus_dm_rma = _wilder_rma(minus_dm, period)

    atr_safe = atr_rma.replace(0.0, np.nan)
    plus_di = 100.0 * (plus_dm_rma / atr_safe)
    minus_di = 100.0 * (minus_dm_rma / atr_safe)

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx = _wilder_rma(dx, period)

    return adx, plus_di, minus_di


def adx_context(
    adx_series: pd.Series,
    plus_di_series: pd.Series,
    minus_di_series: pd.Series,
    *,
    slope_lookback: int = 3,
    dominance_lookback: int = 6,
) -> dict[str, object]:
    """Summarize ADX/DI state without changing external payload contracts.

    Phase-2 provides richer internal interpretation for engines while keeping the
    public scanner/execution workflow stable. The returned dictionary is intended
    for local scoring decisions; callers can continue emitting the same payload
    keys as before.
    """
    try:
        adx_s = pd.to_numeric(adx_series, errors="coerce").astype(float)
        pdi_s = pd.to_numeric(plus_di_series, errors="coerce").astype(float)
        mdi_s = pd.to_numeric(minus_di_series, errors="coerce").astype(float)
    except Exception:
        adx_s = pd.Series(dtype="float64")
        pdi_s = pd.Series(dtype="float64")
        mdi_s = pd.Series(dtype="float64")

    if adx_s.empty and pdi_s.empty and mdi_s.empty:
        return {
            "adx": None,
            "plus_di": None,
            "minus_di": None,
            "di_spread": None,
            "adx_slope": 0.0,
            "adx_accel": 0.0,
            "dominant_side": None,
            "dominance_bars": 0,
            "regime": "unknown",
        }

    data = pd.concat([adx_s.rename("adx"), pdi_s.rename("pdi"), mdi_s.rename("mdi")], axis=1)
    data = data.ffill()

    adx_clean = data["adx"].dropna()
    pdi_clean = data["pdi"].dropna()
    mdi_clean = data["mdi"].dropna()

    adx_last = float(adx_clean.iloc[-1]) if not adx_clean.empty else None
    pdi_last = float(pdi_clean.iloc[-1]) if not pdi_clean.empty else None
    mdi_last = float(mdi_clean.iloc[-1]) if not mdi_clean.empty else None
    di_spread = abs(float(pdi_last) - float(mdi_last)) if pdi_last is not None and mdi_last is not None else None

    slope_lb = max(2, int(slope_lookback or 3))
    adx_slope = 0.0
    if len(adx_clean) >= slope_lb:
        adx_slope = float(adx_clean.iloc[-1] - adx_clean.iloc[-slope_lb])

    adx_accel = 0.0
    if len(adx_clean) >= (2 * slope_lb - 1):
        recent = float(adx_clean.iloc[-1] - adx_clean.iloc[-slope_lb])
        prior = float(adx_clean.iloc[-slope_lb] - adx_clean.iloc[-(2 * slope_lb - 1)])
        adx_accel = float(recent - prior)

    dominant_side = None
    if pdi_last is not None and mdi_last is not None:
        if pdi_last > mdi_last:
            dominant_side = "LONG"
        elif mdi_last > pdi_last:
            dominant_side = "SHORT"

    dominance_bars = 0
    dom_lb = max(2, int(dominance_lookback or 6))
    if dominant_side and not pdi_clean.empty and not mdi_clean.empty:
        dom_vec = np.where(pdi_clean.reindex(data.index).ffill() > mdi_clean.reindex(data.index).ffill(), "LONG", "SHORT")
        for side in dom_vec[::-1][:dom_lb]:
            if side == dominant_side:
                dominance_bars += 1
            else:
                break

    regime = "unknown"
    level = float(adx_last) if adx_last is not None and np.isfinite(adx_last) else None
    spread = float(di_spread) if di_spread is not None and np.isfinite(di_spread) else 0.0
    slope = float(adx_slope)
    accel = float(adx_accel)
    if level is None:
        regime = "unknown"
    elif level < 14.0:
        regime = "coiling" if slope > 0.75 and spread >= 4.0 else "dead_chop"
    elif level < 20.0:
        regime = "emerging" if slope > 1.0 and spread >= 5.0 else "coiling"
    elif level < 28.0:
        if slope < -1.0 and accel <= 0.0:
            regime = "exhausting"
        elif slope > 0.5 and spread >= 6.0:
            regime = "strengthening"
        else:
            regime = "healthy_trend"
    else:
        if slope < -1.25:
            regime = "exhausting"
        elif spread >= 8.0 and dominance_bars >= 2:
            regime = "healthy_trend"
        else:
            regime = "mature_trend"

    return {
        "adx": level,
        "plus_di": pdi_last,
        "minus_di": mdi_last,
        "di_spread": (float(di_spread) if di_spread is not None and np.isfinite(di_spread) else None),
        "adx_slope": slope,
        "adx_accel": accel,
        "dominant_side": dominant_side,
        "dominance_bars": int(dominance_bars),
        "regime": regime,
    }


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig


def session_vwap(
    df: pd.DataFrame,
    tz: str = "America/New_York",
    include_premarket: bool = False,
    include_afterhours: bool = False,
) -> pd.Series:
    """
    Canonical session VWAP used across the scanner.

    Session VWAP **resets each trading day** and defaults to an RTH anchor
    beginning at the US cash open (09:30 ET).

    - include_premarket=False => VWAP starts at 09:30 (RTH-only VWAP).
    - include_premarket=True  => VWAP starts at 04:00 (Extended VWAP, useful for PM scalps).
    - include_afterhours=True => continues after 16:00 (extended into AH).

    Bars outside the included session window receive NaN.
    """
    if df.empty:
        return pd.Series(dtype="float64")

    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx_et = idx.tz_localize(tz)
    else:
        idx_et = idx.tz_convert(tz)

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]

    # session window
    start_h, start_m = (4, 0) if include_premarket else (9, 30)
    end_h, end_m = (20, 0) if include_afterhours else (16, 0)

    t = pd.Series([d.time() for d in idx_et], index=df.index)
    # Build boolean mask for included times
    def _in_window(tt):
        return (tt.hour, tt.minute) >= (start_h, start_m) and (tt.hour, tt.minute) < (end_h, end_m)

    mask = t.map(_in_window).astype(bool)

    # Group by ET date and compute VWAP cumulatives only within included session window
    dates = pd.Series(idx_et.date, index=df.index)
    out = pd.Series(index=df.index, dtype="float64")

    for d, g in df.groupby(dates):
        gi = g.index
        gmask = mask.loc[gi]
        if not gmask.any():
            continue
        gi_in = gi[gmask.values]
        gpv = pv.loc[gi_in].cumsum()
        gvol = g.loc[gi_in, "volume"].cumsum().replace(0, np.nan)
        out.loc[gi_in] = gpv / gvol

    return out
