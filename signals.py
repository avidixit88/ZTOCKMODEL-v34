from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import math

from indicators import (
    vwap as calc_vwap,
    session_vwap as calc_session_vwap,
    atr as calc_atr,
    ema as calc_ema,
    adx as calc_adx,
    adx_context as calc_adx_context,
    rolling_swing_lows,
    rolling_swing_highs,
    detect_fvg,
    find_order_block,
    find_breaker_block,
    in_zone,
)
from sessions import classify_session, classify_liquidity_phase


def _vwap_basis_metadata(*, engine: str, vwap_logic: str = "session", session_vwap_include_premarket: bool = False, session_vwap_include_afterhours: bool = False, note: str | None = None) -> dict[str, object]:
    """Standardized VWAP-basis metadata for diagnostics and alert payloads.

    This keeps chart-vs-engine audits honest by making the active reference line
    explicit in every engine payload.
    """
    logic = str(vwap_logic or "session").lower()
    if logic == "session":
        start = "04:00 ET" if session_vwap_include_premarket else "09:30 ET"
        end = "20:00 ET" if session_vwap_include_afterhours else "16:00 ET"
        label = f"session_vwap[{start}-{end}]"
        description = f"Session VWAP reset each ET trading day; includes bars from {start} to {end}."
    else:
        label = "cumulative_vwap[loaded_window]"
        description = "Cumulative VWAP across the currently loaded bar window; it does not reset at the cash open."

    meta: dict[str, object] = {
        "engine_name": str(engine),
        "vwap_reference_kind": logic,
        "vwap_reference_label": label,
        "vwap_reference_description": description,
        "session_vwap_include_premarket": bool(session_vwap_include_premarket),
        "session_vwap_include_afterhours": bool(session_vwap_include_afterhours),
    }
    if note:
        meta["vwap_reference_note"] = str(note)
    return meta


def _cap_score(x: float | int | None) -> int:
    """Scores are treated as 0..100 for UI + alerting.

    The internal point system can temporarily exceed 100 when multiple features
    stack or when ATR normalization scales up. We cap here so the UI never
    shows impossible percentages (e.g., 113%).
    """
    try:
        if x is None:
            return 0
        return int(np.clip(float(x), 0.0, 100.0))
    except Exception:
        return 0


@dataclass
class SignalResult:
    symbol: str
    bias: str                      # "LONG", "SHORT", "NEUTRAL"
    setup_score: int               # 0..100 (calibrated)
    reason: str
    entry: Optional[float]
    stop: Optional[float]
    target_1r: Optional[float]
    target_2r: Optional[float]
    last_price: Optional[float]
    timestamp: Optional[pd.Timestamp]
    session: str                   # OPENING/MIDDAY/POWER/PREMARKET/AFTERHOURS/OFF
    extras: Dict[str, Any]


# ---------------------------
# SWING / Intraday-Swing (structure-first) signal family
# ---------------------------

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample intraday OHLCV to a higher timeframe without additional API calls.

    We use this for Swing alerts so we don't add extra Alpha Vantage calls.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if not isinstance(df.index, pd.DatetimeIndex):
        return df.copy()
    out = (
        df[["open", "high", "low", "close", "volume"]]
        .resample(rule)
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna()
    )
    return out


def compute_swing_signal(
    symbol: str,
    ohlcv: pd.DataFrame,
    rsi5: pd.Series,
    rsi14: pd.Series,
    macd_hist: pd.Series,
    *,
    interval: str = "1min",
    pro_mode: bool = False,
    # Time filters
    allow_opening: bool = True,
    allow_midday: bool = True,
    allow_power: bool = True,
    allow_premarket: bool = False,
    allow_afterhours: bool = False,
    use_last_closed_only: bool = False,
    bar_closed_guard: bool = True,
    # Shared options
    vwap_logic: str = "session",
    session_vwap_include_premarket: bool = False,
    fib_lookback_bars: int = 240,
    orb_minutes: int = 15,
    liquidity_weighting: float = 0.55,
    target_atr_pct: float | None = None,
) -> SignalResult:
    """SWING v2 — HTF dip-buy positioning with 5m confirmation (CONFIRM-only emails).

    Design principles:
      - SWING is a *positioning* engine (define cheap-risk buy zones) not an execution engine.
      - Use HTF structure (30m derived from intraday bars) to define the impulse leg and retrace band.
      - Stages:
          STALK   : candidate is valid; waiting for price to enter the buy zone
          BUY ZONE: price has entered the retrace band; waiting for confirmation
          CONFIRM : 5m confirmation (2-of-3); actionable (email is sent on CONFIRM only)
      - Streamlit safety: extras contain primitives only (no dict/list objects).
    """
    swing_vwap_meta = _vwap_basis_metadata(
        engine="SWING",
        vwap_logic=vwap_logic,
        session_vwap_include_premarket=bool(session_vwap_include_premarket),
        session_vwap_include_afterhours=bool(allow_afterhours),
        note="HTF trend alignment uses a cumulative VWAP proxy on 30-minute resampled bars.",
    )

    # -------------------------
    # Basic guards
    # -------------------------
    if ohlcv is None or ohlcv.empty or len(ohlcv) < 120:
        return SignalResult(
            symbol, "CHOP", 0, "Not enough data",
            None, None, None, None,
            None, None, "OFF",
            {"family": "SWING", "stage": "OFF", "swing_stage": "OFF", **swing_vwap_meta}
        )

    df = ohlcv.copy()
    if use_last_closed_only and len(df) >= 2:
        df = df.iloc[:-1].copy()

    last_ts = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None
    last_price = float(df["close"].iloc[-1])

    # -------------------------
    # Session gating (respects allowed_sessions toggles)
    # -------------------------
    try:
        sess = classify_session(
            last_ts,
            allow_opening=allow_opening,
            allow_midday=allow_midday,
            allow_power=allow_power,
            allow_premarket=allow_premarket,
            allow_afterhours=allow_afterhours,
        )
    except Exception:
        sess = "OFF"

    if sess == "OFF":
        return SignalResult(
            symbol, "CHOP", 0, "Outside allowed session",
            None, None, None, None,
            last_price, last_ts, "OFF",
            {"family": "SWING", "stage": "OFF", "swing_stage": "OFF", **swing_vwap_meta}
        )

    # -------------------------
    # HTF (30m) context — derived via resample (no extra API call)
    #   - When running 5m, resampling preserves your interval choice.
    # -------------------------
    htf = _resample_ohlcv(df, "30T").tail(5 * 13 * 10).copy()  # ~10 trading days of 30m bars
    if len(htf) < 60:
        return SignalResult(
            symbol, "CHOP", 0, "Not enough HTF bars",
            None, None, None, None,
            last_price, last_ts, sess,
            {"family": "SWING", "stage": "OFF", "swing_stage": "OFF", **swing_vwap_meta}
        )

    # Core HTF measures
    htf["ema20"] = calc_ema(htf["close"], 20)
    htf["ema50"] = calc_ema(htf["close"], 50)
    htf["atr"] = calc_atr(htf, 14)

    atr30 = float(htf["atr"].iloc[-1]) if np.isfinite(htf["atr"].iloc[-1]) else float(calc_atr(df, 14).iloc[-1])
    atr30 = atr30 if np.isfinite(atr30) and atr30 > 0 else max(1e-9, float(df["high"].tail(30).max() - df["low"].tail(30).min()) / 30.0)

    # HTF VWAP proxy (cumulative on HTF bars)
    try:
        htf_vwap = calc_vwap(htf).iloc[-1]
        htf_vwap = float(htf_vwap) if np.isfinite(htf_vwap) else None
    except Exception:
        htf_vwap = None

    # -------------------------
    # HTF pivot structure
    # -------------------------
    is_low = rolling_swing_lows(htf["low"], left=3, right=3)
    is_high = rolling_swing_highs(htf["high"], left=3, right=3)

    pivot_lows = htf.loc[is_low]
    pivot_highs = htf.loc[is_high]

    if len(pivot_lows) < 2 or len(pivot_highs) < 2:
        return SignalResult(
            symbol, "CHOP", 0, "Awaiting structure",
            None, None, None, None,
            last_price, last_ts, sess,
            {"family": "SWING", "stage": "PRE", "actionable": False, "swing_stage": "STALK", **swing_vwap_meta}
        )

    last_low = float(pivot_lows["low"].iloc[-1])
    prev_low = float(pivot_lows["low"].iloc[-2])
    last_high = float(pivot_highs["high"].iloc[-1])
    prev_high = float(pivot_highs["high"].iloc[-2])

    uptrend = (last_high > prev_high) and (last_low >= prev_low)
    downtrend = (last_low < prev_low) and (last_high <= prev_high)

    if not uptrend and not downtrend:
        return SignalResult(
            symbol, "CHOP", 0, "CHOP (HTF trend unclear)",
            None, None, None, None,
            last_price, last_ts, sess,
            {"family": "SWING", "stage": "PRE", "actionable": False, "swing_stage": "STALK", **swing_vwap_meta}
        )

    # IMPORTANT WIRING NOTE:
    # Downstream (tables + email gating) expects SWING bias labels:
    #   SWING_LONG / SWING_SHORT / CHOP
    # We keep a separate direction helper for internal comparisons.
    bias_dir = "LONG" if uptrend else "SHORT"
    bias = "SWING_LONG" if bias_dir == "LONG" else "SWING_SHORT"

    # -------------------------
    # Trend lock score (0..5)
    # -------------------------
    tl = 0
    # (1) EMA stack
    ema20 = float(htf["ema20"].iloc[-1]) if np.isfinite(htf["ema20"].iloc[-1]) else None
    ema50 = float(htf["ema50"].iloc[-1]) if np.isfinite(htf["ema50"].iloc[-1]) else None
    if ema20 is not None and ema50 is not None:
        if bias_dir == "LONG" and ema20 >= ema50:
            tl += 1
        if bias_dir == "SHORT" and ema20 <= ema50:
            tl += 1
    # (2) Price vs EMA20
    if ema20 is not None:
        if bias_dir == "LONG" and float(htf["close"].iloc[-1]) >= ema20:
            tl += 1
        if bias_dir == "SHORT" and float(htf["close"].iloc[-1]) <= ema20:
            tl += 1
    # (3) HTF VWAP alignment
    if htf_vwap is not None:
        if bias_dir == "LONG" and float(htf["close"].iloc[-1]) >= htf_vwap:
            tl += 1
        if bias_dir == "SHORT" and float(htf["close"].iloc[-1]) <= htf_vwap:
            tl += 1
    # (4) Structure confirmation
    tl += 1  # uptrend/downtrend already established
    # (5) DI spread proxy via candle range expansion in trend direction (robust, no extra indicators)
    # A small boost if the last HTF candle expanded in the trend direction.
    last_htf = htf.iloc[-1]
    if bias_dir == "LONG" and float(last_htf["close"]) >= float(last_htf["open"]):
        tl += 1
    if bias_dir == "SHORT" and float(last_htf["close"]) <= float(last_htf["open"]):
        tl += 1
    trend_lock_score = int(max(0, min(5, tl)))

    if trend_lock_score < 2:
        return SignalResult(
            symbol, bias, 0, "WATCH - Trend not locked",
            None, None, None, None,
            last_price, last_ts, sess,
            {"family": "SWING", "stage": "PRE", "actionable": False, "swing_stage": "STALK", "trend_lock_score": trend_lock_score}
        )

    # -------------------------
    # Define impulse leg from HTF pivots
    # -------------------------
    # Build ordered pivot list
    piv = []
    for ts, row in pivot_lows.tail(12).iterrows():
        piv.append((ts, "L", float(row["low"])))
    for ts, row in pivot_highs.tail(12).iterrows():
        piv.append((ts, "H", float(row["high"])))
    piv.sort(key=lambda x: x[0])

    impulse_start = None
    impulse_end = None
    # Scan from the end to find the latest complete impulse in the trend direction.
    if bias_dir == "LONG":
        # need L then later H
        for i in range(len(piv) - 1, 0, -1):
            if piv[i][1] == "H":
                # find prior L
                for j in range(i - 1, -1, -1):
                    if piv[j][1] == "L":
                        impulse_start = piv[j][2]
                        impulse_end = piv[i][2]
                        break
            if impulse_start is not None:
                break
    else:
        # need H then later L
        for i in range(len(piv) - 1, 0, -1):
            if piv[i][1] == "L":
                for j in range(i - 1, -1, -1):
                    if piv[j][1] == "H":
                        impulse_start = piv[j][2]
                        impulse_end = piv[i][2]
                        break
            if impulse_start is not None:
                break

    if impulse_start is None or impulse_end is None:
        return SignalResult(
            symbol, bias, 0, "WATCH - Awaiting impulse leg",
            None, None, None, None,
            last_price, last_ts, sess,
            {"family": "SWING", "stage": "PRE", "actionable": False, "swing_stage": "STALK", "trend_lock_score": trend_lock_score}
        )

    impulse_range = abs(impulse_end - impulse_start)
    if not np.isfinite(impulse_range) or impulse_range < 1.2 * atr30:
        return SignalResult(
            symbol, bias, 0, "WATCH - Impulse too small",
            None, None, None, None,
            last_price, last_ts, sess,
            {
                "family": "SWING",
                "stage": "PRE",
                "swing_stage": "STALK",
                "trend_lock_score": trend_lock_score,
                "impulse_start": float(impulse_start),
                "impulse_end": float(impulse_end),
                "retrace_mode": "pivot-leg",
            }
        )

    # -------------------------
    # Retrace band (38.2%..61.8%) + sweet spot 50%
    # -------------------------
    if bias_dir == "LONG":
        # Retrace from impulse_end downwards
        pb1 = float(impulse_end - 0.382 * impulse_range)
        pb2 = float(impulse_end - 0.618 * impulse_range)
        band_high = max(pb1, pb2)
        band_low = min(pb1, pb2)
        retrace_pct = 100.0 * max(0.0, min(1.0, (impulse_end - last_price) / impulse_range))
        invalidated = last_price < (impulse_start - 0.10 * atr30)
    else:
        pb1 = float(impulse_end + 0.382 * impulse_range)
        pb2 = float(impulse_end + 0.618 * impulse_range)
        band_high = max(pb1, pb2)
        band_low = min(pb1, pb2)
        retrace_pct = 100.0 * max(0.0, min(1.0, (last_price - impulse_end) / impulse_range))
        invalidated = last_price > (impulse_start + 0.10 * atr30)

    if invalidated:
        return SignalResult(
            symbol, bias, 0, "Invalidated (broke impulse start)",
            None, None, None, None,
            last_price, last_ts, sess,
            {
                "family": "SWING",
                "stage": "PRE",
                "swing_stage": "FAIL",
                "trend_lock_score": trend_lock_score,
                "impulse_start": float(impulse_start),
                "impulse_end": float(impulse_end),
                "retrace_pct": float(retrace_pct),
                "pb1": float(band_low),
                "pb2": float(band_high),
                "retrace_mode": "pivot-leg",
            }
        )

    # Pullback quality (0..6)
    # - 6 if in band & near 50% retrace; lower if shallow/deep or outside.
    target_retrace = 0.50
    if bias_dir == "LONG":
        retr = (impulse_end - last_price) / impulse_range
    else:
        retr = (last_price - impulse_end) / impulse_range
    retr = float(retr) if np.isfinite(retr) else 0.0
    pb_in_band = (last_price >= band_low) and (last_price <= band_high)
    dist_to_mid = abs(retr - target_retrace)
    pbq = 0
    pbq_reasons = []
    if pb_in_band:
        pbq = 4
        pbq_reasons.append("In band")
        if dist_to_mid <= 0.08:
            pbq += 2
            pbq_reasons.append("Near 50%")
        elif dist_to_mid <= 0.14:
            pbq += 1
            pbq_reasons.append("Near mid")
    else:
        # shallow pullback
        if retr < 0.30:
            pbq = 1
            pbq_reasons.append("Too shallow")
        # deep pullback
        elif retr > 0.75:
            pbq = 1
            pbq_reasons.append("Too deep")
        else:
            pbq = 2
            pbq_reasons.append("Approaching band")

    pullback_quality = int(max(0, min(6, pbq)))

    # -------------------------
    # Confluence (capped)
    # -------------------------
    confluences = []
    conf_n = 0
    tol = 0.25 * atr30

    # (1) HTF VWAP overlap
    if htf_vwap is not None and (band_low - tol) <= htf_vwap <= (band_high + tol):
        conf_n += 1
        confluences.append("HTF VWAP")

    # (2) Prior day levels from intraday df
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            d = df.copy()
            # Use last completed date
            last_date = d.index[-1].date()
            prev_days = sorted({x.date() for x in d.index if x.date() < last_date})
            if prev_days:
                pd_date = prev_days[-1]
                d_prev = d[d.index.date == pd_date]
                if len(d_prev) >= 5:
                    pdh = float(d_prev["high"].max())
                    pdl = float(d_prev["low"].min())
                    pdo = float(d_prev["open"].iloc[0])
                    for name, lvl in [("PDH", pdh), ("PDL", pdl), ("PDO", pdo)]:
                        if (band_low - tol) <= lvl <= (band_high + tol):
                            conf_n += 1
                            confluences.append(name)
    except Exception:
        pass

    # (3) Pivot cluster overlap (nearest prior pivot opposite side)
    try:
        # nearest HTF pivot level near band center
        band_mid = (band_low + band_high) / 2.0
        # use recent pivot highs/lows levels
        levels = []
        levels += [float(x) for x in pivot_highs["high"].tail(8).values]
        levels += [float(x) for x in pivot_lows["low"].tail(8).values]
        if levels:
            closest = min(levels, key=lambda x: abs(x - band_mid))
            if abs(closest - band_mid) <= tol:
                conf_n += 1
                confluences.append("HTF Pivot")
    except Exception:
        pass

    confluence_count = int(min(3, conf_n))
    confluences_str = ", ".join(confluences[:6]) if confluences else ""

    # -------------------------
    # Stage determination: STALK / BUY ZONE / CONFIRM
    # -------------------------
    stage = "STALK"
    entry_trigger_reason = ""
    # Zone entry check (within band OR within proximity)
    prox = 0.20 * atr30
    in_zone_now = (last_price >= (band_low - prox)) and (last_price <= (band_high + prox))
    if in_zone_now:
        stage = "BUY ZONE"

    # Confirmation checks on execution timeframe (df assumed to match selected interval, usually 5m)
    # We only allow confirm if we have a recent zone touch.
    recent = df.tail(20).copy()
    touched = False
    if bias_dir == "LONG":
        touched = float(recent["low"].min()) <= (band_high + prox)
    else:
        touched = float(recent["high"].max()) >= (band_low - prox)

    # Compute 5m ATR for stop sizing
    atr5 = float(calc_atr(df, 14).iloc[-1]) if len(df) >= 20 else atr30
    atr5 = atr5 if np.isfinite(atr5) and atr5 > 0 else atr30

    # Confirm rules (2-of-3)
    confirm_hits = 0
    confirm_reasons = []

    # (1) Reclaim VWAP after zone touch
    vwap_s = None
    try:
        vwap_s = calc_session_vwap(
            df,
            include_premarket=session_vwap_include_premarket,
            include_afterhours=allow_afterhours,
        ).iloc[-1]
        vwap_s = float(vwap_s) if np.isfinite(vwap_s) else None
    except Exception:
        vwap_s = None

    if touched and vwap_s is not None and len(df) >= 2:
        prev_close = float(df["close"].iloc[-2])
        cur_close = float(df["close"].iloc[-1])
        if bias_dir == "LONG" and prev_close < vwap_s <= cur_close:
            confirm_hits += 1
            confirm_reasons.append("VWAP reclaim")
        if bias_dir == "SHORT" and prev_close > vwap_s >= cur_close:
            confirm_hits += 1
            confirm_reasons.append("VWAP reject")

    # (2) Two-candle reversal (robust)
    if touched and len(df) >= 3:
        c1 = df.iloc[-2]
        c2 = df.iloc[-1]
        if bias_dir == "LONG":
            if float(c1["close"]) < float(c1["open"]) and float(c2["close"]) > float(c2["open"]) and float(c2["close"]) > float(c1["high"]):
                confirm_hits += 1
                confirm_reasons.append("2-bar reversal")
        else:
            if float(c1["close"]) > float(c1["open"]) and float(c2["close"]) < float(c2["open"]) and float(c2["close"]) < float(c1["low"]):
                confirm_hits += 1
                confirm_reasons.append("2-bar reversal")

    # (3) RSI regime cross (compute from provided rsi14 if aligned; else compute on df closes)
    rsi_now = None
    rsi_prev = None
    try:
        if rsi14 is not None and len(rsi14) >= len(df):
            rsi_now = float(rsi14.iloc[-1])
            rsi_prev = float(rsi14.iloc[-2]) if len(rsi14) >= 2 else None
        else:
            # lightweight recompute using calc_rsi provided by engine (if available) is outside this module
            # so we compute locally here
            closes = df["close"].astype(float)
            delta = closes.diff()
            up = delta.clip(lower=0).rolling(14).mean()
            down = (-delta.clip(upper=0)).rolling(14).mean()
            rs = up / down.replace(0, np.nan)
            rsi_series = 100 - (100 / (1 + rs))
            rsi_now = float(rsi_series.iloc[-1])
            rsi_prev = float(rsi_series.iloc[-2]) if len(rsi_series) >= 2 else None
    except Exception:
        rsi_now = None
        rsi_prev = None

    if touched and rsi_now is not None and rsi_prev is not None:
        if bias_dir == "LONG" and (rsi_prev < 50 <= rsi_now):
            confirm_hits += 1
            confirm_reasons.append("RSI reclaim")
        if bias_dir == "SHORT" and (rsi_prev > 50 >= rsi_now):
            confirm_hits += 1
            confirm_reasons.append("RSI roll")

    confirmed = touched and (confirm_hits >= 2)

    # -------------------------
    # Score (0..100) — optimized for entry quality
    # -------------------------
    score = 0
    score += 10 * trend_lock_score  # 0..50
    if impulse_range >= 2.0 * atr30:
        score += 8
    if impulse_range >= 3.0 * atr30:
        score += 6  # stack a bit more for bigger displacement
    # Pullback quality boost
    if pullback_quality > 2:
        score += 12 * (pullback_quality - 2)  # max +48
    # Confluence
    score += 8 * confluence_count  # max +24
    # Stage boosts
    if stage == "BUY ZONE":
        score += 10
    if confirmed:
        score += 18
    # Penalties
    # "Actionable" here is a scoring concept (do we have enough quality to consider the setup real?).
    # Email/alert gating is handled separately via extras["actionable"].
    actionable_ok = (confluence_count >= 1) and (pullback_quality >= 2)
    if not actionable_ok:
        score -= 8
    # Character ok: avoid over-extended RSI > 80 longs / <20 shorts
    character_ok = True
    if rsi_now is not None:
        if bias_dir == "LONG" and rsi_now > 80:
            character_ok = False
        if bias_dir == "SHORT" and rsi_now < 20:
            character_ok = False
    if not character_ok:
        score -= 20

    score = int(_cap_score(score))

    # -------------------------
    # Entry / stop / targets (only actionable on CONFIRM)
    # -------------------------
    entry = None
    stop = None
    tp0 = None
    tp1 = None

    # TP0 preference: mean-reversion to session VWAP if available; else impulse end
    if vwap_s is not None and np.isfinite(vwap_s):
        tp0 = float(vwap_s)
    else:
        tp0 = float(impulse_end)

    tp1 = float(impulse_end)

    if confirmed:
        stage = "CONFIRM"
        entry_trigger_reason = ", ".join(confirm_reasons)
        entry = last_price
        if bias_dir == "LONG":
            stop = float(band_low - 0.25 * atr5)
        else:
            stop = float(band_high + 0.25 * atr5)

        # Ensure targets are in the correct direction for R-multiples
        if bias_dir == "LONG":
            tp0 = max(tp0, entry + 0.5 * atr5)
            tp1 = max(tp1, entry + 1.0 * atr5)
        else:
            tp0 = min(tp0, entry - 0.5 * atr5)
            tp1 = min(tp1, entry - 1.0 * atr5)

        return SignalResult(
            symbol,
            bias,
            score,
            "CONFIRM - Dip-buy confirmed",
            float(entry),
            float(stop),
            float(tp0),
            float(tp1),
            last_price,
            last_ts,
            sess,
            {
                "family": "SWING",
                "stage": "CONFIRMED",
                # Email gating expects an explicit boolean.
                # For SWING we only want emails on CONFIRMED signals.
                "actionable": True,
                "swing_stage": "CONFIRM",
                "trend_lock_score": trend_lock_score,
                "retrace_pct": float(retrace_pct),
                "impulse_start": float(impulse_start),
                "impulse_end": float(impulse_end),
                "retrace_mode": "pivot-leg",
                "pullback_quality": pullback_quality,
                "pullback_quality_reasons": "; ".join(pbq_reasons),
                "confluence_count": confluence_count,
                "confluences": confluences_str,
                "pb1": float(band_low),
                "pb2": float(band_high),
                "pullback_band": [float(band_low), float(band_high)],
                "entry_zone": f"{band_low:.4f}–{band_high:.4f}",
                "entry_trigger_reason": entry_trigger_reason,
                **swing_vwap_meta,
            }
        )

    # Not confirmed yet (no email)
    why = "WATCH - Trend locked, awaiting zone entry"
    if stage == "BUY ZONE":
        why = "SETUP - In buy zone, awaiting confirmation"
    return SignalResult(
        symbol,
        bias,
        score,
        why,
        None,
        None,
        None,
        None,
        last_price,
        last_ts,
        sess,
        {
            "family": "SWING",
            "stage": "PRE",
            "actionable": False,
            "swing_stage": stage,
            "trend_lock_score": trend_lock_score,
            "retrace_pct": float(retrace_pct),
            "impulse_start": float(impulse_start),
            "impulse_end": float(impulse_end),
            "retrace_mode": "pivot-leg",
            "pullback_quality": pullback_quality,
            "pullback_quality_reasons": "; ".join(pbq_reasons),
            "confluence_count": confluence_count,
            "confluences": confluences_str,
            "pb1": float(band_low),
            "pb2": float(band_high),
            "pullback_band": [float(band_low), float(band_high)],
            "entry_zone": f"{band_low:.4f}–{band_high:.4f}",
            **swing_vwap_meta,
        }
    )


def _mfe_percentile_from_history(
    df: pd.DataFrame,
    *,
    direction: str,
    occur_mask: pd.Series,
    horizon_bars: int,
    pct: float,
) -> tuple[float | None, int]:
    """Compute a percentile of forward MFE for occurrences marked by occur_mask.

    LONG MFE is max(high fwd) - close at signal bar.
    SHORT MFE is close - min(low fwd).
    Returns (mfe_pct, n_samples).
    """
    try:
        h = int(horizon_bars)
        if h <= 0:
            return None, 0
    except Exception:
        return None, 0

    if occur_mask is None or df is None or len(df) == 0:
        return None, 0

    try:
        close = df["close"].astype(float)
        hi = df["high"].astype(float)
        lo = df["low"].astype(float)
    except Exception:
        return None, 0

    idxs = [i for i, ok in enumerate(occur_mask.values.tolist()) if bool(ok)]
    idxs = [i for i in idxs if i + h < len(df)]
    if len(idxs) < 10:
        return None, len(idxs)

    mfes: list[float] = []
    for i in idxs:
        ref = float(close.iloc[i])
        if direction.upper() == "LONG":
            fwd_max = float(hi.iloc[i + 1 : i + h + 1].max())
            mfes.append(max(0.0, fwd_max - ref))
        else:
            fwd_min = float(lo.iloc[i + 1 : i + h + 1].min())
            mfes.append(max(0.0, ref - fwd_min))

    if not mfes:
        return None, 0

    mfes.sort()
    k = int(round((pct / 100.0) * (len(mfes) - 1)))
    k = max(0, min(len(mfes) - 1, k))
    return float(mfes[k]), len(mfes)


def _tp3_from_expected_excursion(
    df: pd.DataFrame,
    *,
    direction: str,
    signature: dict,
    entry_px: float,
    interval_mins: int,
    lookback_bars: int = 600,
    horizon_bars: int | None = None,
) -> tuple[float | None, dict]:
    """Compute TP3 using expected excursion (rolling MFE) for similar historical signatures.

    Lightweight rolling backtest per symbol+interval:
    - Find prior bars where the same boolean signature fired
    - Compute forward Max Favorable Excursion (MFE) over horizon
    - Use a high percentile (95th) as TP3 (runner/lottery)

    Returns (tp3, diagnostics).
    """
    diag = {
        "tp3_mode": "mfe_p95",
        "samples": 0,
        "horizon_bars": None,
        "signature": signature,
    }
    if df is None or len(df) < 60:
        return None, diag

    try:
        n = int(lookback_bars)
    except Exception:
        n = 600
    n = max(120, min(len(df), n))
    d = df.iloc[-n:].copy()

    # Default horizon: 1m -> 15 bars (15m); 5m -> 6 bars (~30m)
    if horizon_bars is None:
        hb = 15 if int(interval_mins) <= 1 else 6
    else:
        hb = int(horizon_bars)
    hb = max(3, hb)
    diag["horizon_bars"] = hb

    # vwap series for signature matching (prefer a precomputed 'vwap_use')
    if "vwap_use" in d.columns:
        vwap_use = d["vwap_use"].astype(float)
    elif "vwap_sess" in d.columns:
        vwap_use = d["vwap_sess"].astype(float)
    elif "vwap_cum" in d.columns:
        vwap_use = d["vwap_cum"].astype(float)
    else:
        return None, diag

    close = d["close"].astype(float)

    # Recompute simple boolean events in-window to find prior occurrences.
    was_below = (close.shift(3) < vwap_use.shift(3)) | (close.shift(5) < vwap_use.shift(5))
    reclaim = (close > vwap_use) & (close.shift(1) <= vwap_use.shift(1))
    was_above = (close.shift(3) > vwap_use.shift(3)) | (close.shift(5) > vwap_use.shift(5))
    reject = (close < vwap_use) & (close.shift(1) >= vwap_use.shift(1))

    rsi5 = d.get("rsi5")
    rsi14 = d.get("rsi14")
    macd_hist = d.get("macd_hist")
    vol = d.get("volume")

    if rsi5 is not None:
        rsi5 = rsi5.astype(float)
    if rsi14 is not None:
        rsi14 = rsi14.astype(float)
    if macd_hist is not None:
        macd_hist = macd_hist.astype(float)

    # RSI events (match current engine semantics approximately)
    rsi_snap = None
    rsi_down = None
    if rsi5 is not None:
        rsi_snap = ((rsi5 >= 30) & (rsi5.shift(1) < 30)) | ((rsi5 >= 25) & (rsi5.shift(1) < 25))
        rsi_down = ((rsi5 <= 70) & (rsi5.shift(1) > 70)) | ((rsi5 <= 75) & (rsi5.shift(1) > 75))

    # MACD turns
    macd_up = None
    macd_dn = None
    if macd_hist is not None:
        macd_up = (macd_hist > macd_hist.shift(1)) & (macd_hist.shift(1) > macd_hist.shift(2))
        macd_dn = (macd_hist < macd_hist.shift(1)) & (macd_hist.shift(1) < macd_hist.shift(2))

    # Volume confirm: last bar volume >= multiplier * rolling median(30)
    vol_ok = None
    if vol is not None:
        v = vol.astype(float)
        med = v.rolling(30, min_periods=10).median()
        mult = float(signature.get("vol_mult") or 1.25)
        vol_ok = v >= (mult * med)

    # Micro-structure: higher-low / lower-high
    hl_ok = None
    lh_ok = None
    try:
        lows = d["low"].astype(float)
        highs = d["high"].astype(float)
        hl_ok = lows.iloc[-1] > lows.rolling(10, min_periods=5).min()
        lh_ok = highs.iloc[-1] < highs.rolling(10, min_periods=5).max()
    except Exception:
        pass

    # Build occurrence mask to match the CURRENT signature
    diru = direction.upper()
    if diru == "LONG":
        m = (was_below & reclaim)
        if signature.get("rsi_event") and rsi_snap is not None:
            m = m & rsi_snap
        if signature.get("macd_event") and macd_up is not None:
            m = m & macd_up
        if signature.get("vol_event") and vol_ok is not None:
            m = m & vol_ok
        if signature.get("struct_event") and hl_ok is not None:
            m = m & hl_ok
    else:
        m = (was_above & reject)
        if signature.get("rsi_event") and rsi_down is not None:
            m = m & rsi_down
        if signature.get("macd_event") and macd_dn is not None:
            m = m & macd_dn
        if signature.get("vol_event") and vol_ok is not None:
            m = m & vol_ok
        if signature.get("struct_event") and lh_ok is not None:
            m = m & lh_ok

    mfe95, n_samples = _mfe_percentile_from_history(d, direction=diru, occur_mask=m.fillna(False), horizon_bars=hb, pct=95.0)
    diag["samples"] = int(n_samples)
    if mfe95 is None or not np.isfinite(mfe95):
        return None, diag

    try:
        mfe95 = float(mfe95)
        if diru == "LONG":
            return float(entry_px) + mfe95, diag
        return float(entry_px) - mfe95, diag
    except Exception:
        return None, diag

def _candidate_levels_from_context(
    *,
    levels: Dict[str, Any],
    recent_swing_high: float,
    recent_swing_low: float,
    hi: float,
    lo: float,
) -> Dict[str, float]:
    """Collect common structure/liquidity levels into a flat dict of floats.

    We use these as *potential* scalp targets (TP0). We intentionally favor
    levels that are meaningful to traders (prior day hi/lo, ORB, swing pivots),
    but fall back gracefully when some session levels aren't available.
    """
    out: Dict[str, float] = {}

    def _add(name: str, v: Any):
        try:
            if v is None:
                return
            fv = float(v)
            if np.isfinite(fv):
                out[name] = fv
        except Exception:
            return

    # Session liquidity levels (may be None)
    _add("orb_high", levels.get("orb_high"))
    _add("orb_low", levels.get("orb_low"))
    _add("prior_high", levels.get("prior_high"))
    _add("prior_low", levels.get("prior_low"))
    _add("premarket_high", levels.get("premarket_high"))
    _add("premarket_low", levels.get("premarket_low"))

    # Swing + range context
    _add("recent_swing_high", recent_swing_high)
    _add("recent_swing_low", recent_swing_low)
    _add("range_high", hi)
    _add("range_low", lo)
    return out


def _pick_tp0(
    direction: str,
    *,
    entry_px: float,
    last_px: float,
    atr_last: float,
    levels: Dict[str, float],
) -> Optional[float]:
    """Pick TP0 as the nearest meaningful level beyond entry.

    For scalping, TP0 should usually be *closer* than 1R/2R and should map to
    real structure. If no structure exists in-range, we fall back to an ATR-based
    objective.
    """
    try:
        entry_px = float(entry_px)
        last_px = float(last_px)
    except Exception:
        return None

    max_dist = None
    if atr_last and atr_last > 0:
        # Don't pick a target 10 ATR away for a scalp; keep it sane.
        max_dist = 3.0 * float(atr_last)

    cands: List[float] = []
    if direction == "LONG":
        for _, lvl in levels.items():
            if lvl > entry_px:
                cands.append(float(lvl))
        if cands:
            tp0 = min(cands, key=lambda x: abs(x - entry_px))
            if max_dist is None or abs(tp0 - entry_px) <= max_dist:
                return float(tp0)
        # Fallback: small objective beyond last/entry
        bump = 0.8 * float(atr_last) if atr_last else max(0.001 * last_px, 0.01)
        return float(max(entry_px, last_px) + bump)

    # SHORT
    for _, lvl in levels.items():
        if lvl < entry_px:
            cands.append(float(lvl))
    if cands:
        tp0 = min(cands, key=lambda x: abs(x - entry_px))
        if max_dist is None or abs(tp0 - entry_px) <= max_dist:
            return float(tp0)
    bump = 0.8 * float(atr_last) if atr_last else max(0.001 * last_px, 0.01)
    return float(min(entry_px, last_px) - bump)


def _eta_minutes_to_tp0(
    *,
    last_px: float,
    tp0: Optional[float],
    atr_last: float,
    interval_mins: int,
    liquidity_mult: float,
) -> Optional[float]:
    """Rough expected minutes to TP0 using ATR as a speed proxy.

    This is not meant to be precise. It's a UI helper to detect *slow* setups
    (common midday / low-liquidity conditions).
    """
    try:
        if tp0 is None:
            return None
        if not atr_last or atr_last <= 0:
            return None
        dist = abs(float(tp0) - float(last_px))
        bars = dist / float(atr_last)
        # liquidity_mult >1 means faster; <1 slower.
        speed = max(0.5, float(liquidity_mult))
        mins = bars * float(interval_mins) / speed
        return float(min(max(mins, 0.0), 999.0))
    except Exception:
        return None


def _entry_limit_and_chase(
    direction: str,
    *,
    entry_px: float,
    last_px: float,
    atr_last: float,
    slippage_mode: str,
    fixed_slippage_cents: float,
    atr_fraction_slippage: float,
) -> Tuple[float, float]:
    """Return (limit_entry, chase_line).

    - limit_entry: your planned limit.
    - chase_line: a "max pain" price where, if crossed, you're late and should
      reassess or switch to a different execution model.
    """
    slip = _slip_amount(
        slippage_mode=slippage_mode,
        fixed_slippage_cents=fixed_slippage_cents,
        atr_last=float(atr_last or 0.0),
        atr_fraction_slippage=float(atr_fraction_slippage or 0.0),
    )
    try:
        entry_px = float(entry_px)
        last_px = float(last_px)
    except Exception:
        return entry_px, entry_px

    # "Chase" is intentionally tight for scalps.
    chase_pad = 0.25 * float(atr_last) if atr_last else max(0.001 * last_px, 0.01)
    if direction == "LONG":
        chase = max(entry_px, last_px) + chase_pad + slip
        return float(entry_px), float(chase)
    chase = min(entry_px, last_px) - chase_pad - slip
    return float(entry_px), float(chase)


def _is_rising(series: pd.Series, bars: int = 3) -> bool:
    """Simple monotonic rise check over the last N bars."""
    try:
        s = series.dropna().tail(int(bars))
        if len(s) < int(bars):
            return False
        return bool(all(float(s.iloc[i]) > float(s.iloc[i - 1]) for i in range(1, len(s))))
    except Exception:
        return False


def _is_falling(series: pd.Series, bars: int = 3) -> bool:
    """Simple monotonic fall check over the last N bars."""
    try:
        s = series.dropna().tail(int(bars))
        if len(s) < int(bars):
            return False
        return bool(all(float(s.iloc[i]) < float(s.iloc[i - 1]) for i in range(1, len(s))))
    except Exception:
        return False


PRESETS: Dict[str, Dict[str, float]] = {
    "Fast scalp": {
        "min_actionable_score": 70,
        "vol_multiplier": 1.15,
        "require_volume": 0,
        "require_macd_turn": 1,
        "require_vwap_event": 1,
        "require_rsi_event": 1,
    },
    "Cleaner signals": {
        "min_actionable_score": 80,
        "vol_multiplier": 1.35,
        "require_volume": 1,
        "require_macd_turn": 1,
        "require_vwap_event": 1,
        "require_rsi_event": 1,
    },
}


def _fib_retracement_levels(hi: float, lo: float) -> List[Tuple[str, float]]:
    ratios = [0.382, 0.5, 0.618, 0.786]
    rng = hi - lo
    if rng <= 0:
        return []
    # "pullback" levels for an up-move: hi - r*(hi-lo)
    return [(f"Fib {r:g}", hi - r * rng) for r in ratios]


def _fib_extensions(hi: float, lo: float) -> List[Tuple[str, float]]:
    # extensions above hi for longs, below lo for shorts (we'll mirror in logic)
    ratios = [1.0, 1.272, 1.618]
    rng = hi - lo
    if rng <= 0:
        return []
    return [(f"Ext {r:g}", hi + (r - 1.0) * rng) for r in ratios]


def _closest_level(price: float, levels: List[Tuple[str, float]]) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    if not levels:
        return None, None, None
    name, lvl = min(levels, key=lambda x: abs(price - x[1]))
    return name, float(lvl), float(abs(price - lvl))


def _session_liquidity_levels(df: pd.DataFrame, interval_mins: int, orb_minutes: int):
    """Compute simple liquidity levels: prior session high/low, today's premarket high/low, and ORB high/low."""
    if df is None or len(df) < 5:
        return {}
    # normalize timestamps to ET
    if "time" in df.columns:
        ts = pd.to_datetime(df["time"])
    else:
        ts = pd.to_datetime(df.index)

    try:
        ts = ts.dt.tz_localize("America/New_York") if getattr(ts.dt, "tz", None) is None else ts.dt.tz_convert("America/New_York")
    except Exception:
        try:
            ts = ts.dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
        except Exception:
            # if tz ops fail, fall back to naive dates
            pass

    d = df.copy()
    d["_ts"] = ts
    # derive dates
    try:
        cur_date = d["_ts"].iloc[-1].date()
        dates = sorted({x.date() for x in d["_ts"] if pd.notna(x)})
    except Exception:
        cur_date = pd.to_datetime(df.index[-1]).date()
        dates = sorted({pd.to_datetime(x).date() for x in df.index})

    prev_date = dates[-2] if len(dates) >= 2 else cur_date

    def _t(x):
        try:
            return x.time()
        except Exception:
            return None

    def _is_pre(x):
        t = _t(x)
        return t is not None and (t >= pd.Timestamp("04:00").time()) and (t < pd.Timestamp("09:30").time())

    def _is_rth(x):
        t = _t(x)
        return t is not None and (t >= pd.Timestamp("09:30").time()) and (t <= pd.Timestamp("16:00").time())

    prev = d[d["_ts"].dt.date == prev_date] if "_ts" in d else df.iloc[:0]
    prev_rth = prev[prev["_ts"].apply(_is_rth)] if len(prev) else prev
    prior_high = float(prev_rth["high"].max()) if len(prev_rth) else (float(prev["high"].max()) if len(prev) else None)
    prior_low = float(prev_rth["low"].min()) if len(prev_rth) else (float(prev["low"].min()) if len(prev) else None)

    cur = d[d["_ts"].dt.date == cur_date] if "_ts" in d else df
    cur_pre = cur[cur["_ts"].apply(_is_pre)] if len(cur) else cur
    pre_hi = float(cur_pre["high"].max()) if len(cur_pre) else None
    pre_lo = float(cur_pre["low"].min()) if len(cur_pre) else None

    cur_rth = cur[cur["_ts"].apply(_is_rth)] if len(cur) else cur
    orb_bars = max(1, int(math.ceil(float(orb_minutes) / max(float(interval_mins), 1.0))))
    orb_slice = cur_rth.head(orb_bars)
    orb_hi = float(orb_slice["high"].max()) if len(orb_slice) else None
    orb_lo = float(orb_slice["low"].min()) if len(orb_slice) else None

    return {
        "prior_high": prior_high, "prior_low": prior_low,
        "premarket_high": pre_hi, "premarket_low": pre_lo,
        "orb_high": orb_hi, "orb_low": orb_lo,
    }

def _asof_slice(df: pd.DataFrame, interval_mins: int, use_last_closed_only: bool, bar_closed_guard: bool) -> pd.DataFrame:
    """Return df truncated so the last row represents the 'as-of' bar we can legally use."""
    if df is None or len(df) < 3:
        return df
    asof_idx = len(df) - 1

    # Always allow "snapshot mode" to use last fully completed bar
    if use_last_closed_only:
        asof_idx = max(0, len(df) - 2)

    if bar_closed_guard and len(df) >= 2:
        try:
            # Determine timestamp of latest bar
            if "time" in df.columns:
                last_ts = pd.to_datetime(df["time"].iloc[-1], utc=False)
            else:
                last_ts = pd.to_datetime(df.index[-1], utc=False)

            # Normalize to ET if timezone-naive
            now = pd.Timestamp.now(tz="America/New_York")
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("America/New_York")
            else:
                last_ts = last_ts.tz_convert("America/New_York")

            bar_end = last_ts + pd.Timedelta(minutes=int(interval_mins))
            # If bar hasn't ended yet, step back one candle (avoid partial)
            if now < bar_end:
                asof_idx = min(asof_idx, len(df) - 2)
        except Exception:
            # If anything goes sideways, be conservative
            asof_idx = min(asof_idx, len(df) - 2)

    asof_idx = max(0, int(asof_idx))
    return df.iloc[: asof_idx + 1].copy()


def _detect_liquidity_sweep(df: pd.DataFrame, levels: dict, *, atr_last: float | None = None, buffer: float = 0.0):
    """Liquidity sweep with confirmation (reclaim + displacement).

    We only count a sweep when ALL are true on the latest bar:
      1) Liquidity grab (wick through a key level)
      2) Reclaim (close back on the 'correct' side of the level)
      3) Displacement (range >= ~1.2x ATR) to filter chop/fakes

    Returns:
      {"type": "...", "level": float(level), "confirmed": bool}
    or None.
    """
    if df is None or len(df) < 2 or not levels:
        return None

    h = float(df["high"].iloc[-1])
    l = float(df["low"].iloc[-1])
    c = float(df["close"].iloc[-1])

    # Displacement filter (keep it mild; still allow if ATR isn't available)
    disp_ok = True
    if atr_last is not None and np.isfinite(float(atr_last)) and float(atr_last) > 0:
        disp_ok = float(h - l) >= 1.2 * float(atr_last)

    def _bull(level: float) -> Optional[dict]:
        # wick below, reclaim above
        if l < level - buffer and c > level + buffer and disp_ok:
            return {"type": "bull_sweep", "level": float(level), "confirmed": True}
        return None

    def _bear(level: float) -> Optional[dict]:
        # wick above, reclaim below
        if h > level + buffer and c < level - buffer and disp_ok:
            return {"type": "bear_sweep", "level": float(level), "confirmed": True}
        return None

    # Priority: prior day hi/lo, then premarket hi/lo
    ph = levels.get("prior_high")
    pl = levels.get("prior_low")
    if ph is not None:
        out = _bear(float(ph))
        if out:
            out["type"] = "bear_sweep_prior_high"
            return out
    if pl is not None:
        out = _bull(float(pl))
        if out:
            out["type"] = "bull_sweep_prior_low"
            return out

    pmah = levels.get("premarket_high")
    pmal = levels.get("premarket_low")
    if pmah is not None:
        out = _bear(float(pmah))
        if out:
            out["type"] = "bear_sweep_premarket_high"
            return out
    if pmal is not None:
        out = _bull(float(pmal))
        if out:
            out["type"] = "bull_sweep_premarket_low"
            return out

    return None


def _orb_three_stage(
    df: pd.DataFrame,
    *,
    orb_high: float | None,
    orb_low: float | None,
    buffer: float,
    lookback_bars: int = 30,
    accept_bars: int = 2,
) -> Dict[str, bool]:
    """ORB as a 3-stage sequence: break -> accept -> retest.

    Bull:
      - break: close crosses above orb_high
      - accept: next `accept_bars` closes stay above orb_high
      - retest: subsequent bar(s) touch orb_high (within buffer) and close back above

    Bear mirrors below orb_low.

    Returns dict with:
      {"bull_orb_seq": bool, "bear_orb_seq": bool, "bull_break": bool, "bear_break": bool}
    """
    out = {"bull_orb_seq": False, "bear_orb_seq": False, "bull_break": False, "bear_break": False}
    if df is None or len(df) < 8:
        return out

    d = df.tail(int(min(max(10, lookback_bars), len(df)))).copy()
    c = d["close"].astype(float)
    h = d["high"].astype(float)
    l = d["low"].astype(float)

    # --- Bull sequence ---
    if orb_high is not None and np.isfinite(float(orb_high)):
        level = float(orb_high)
        broke_idx = None
        for i in range(1, len(d)):
            if c.iloc[i] > level + buffer and c.iloc[i - 1] <= level + buffer:
                broke_idx = i
        if broke_idx is not None:
            out["bull_break"] = True
            # accept: next N closes remain above
            end_acc = min(len(d), broke_idx + 1 + int(accept_bars))
            acc_ok = True
            for j in range(broke_idx + 1, end_acc):
                if c.iloc[j] <= level + buffer:
                    acc_ok = False
                    break
            if acc_ok and end_acc <= len(d) - 1:
                # retest: any later bar tags level (low <= level+buffer) and closes back above
                for k in range(end_acc, len(d)):
                    if l.iloc[k] <= level + buffer and c.iloc[k] > level + buffer:
                        out["bull_orb_seq"] = True
                        break

    # --- Bear sequence ---
    if orb_low is not None and np.isfinite(float(orb_low)):
        level = float(orb_low)
        broke_idx = None
        for i in range(1, len(d)):
            if c.iloc[i] < level - buffer and c.iloc[i - 1] >= level - buffer:
                broke_idx = i
        if broke_idx is not None:
            out["bear_break"] = True
            end_acc = min(len(d), broke_idx + 1 + int(accept_bars))
            acc_ok = True
            for j in range(broke_idx + 1, end_acc):
                if c.iloc[j] >= level - buffer:
                    acc_ok = False
                    break
            if acc_ok and end_acc <= len(d) - 1:
                for k in range(end_acc, len(d)):
                    if h.iloc[k] >= level - buffer and c.iloc[k] < level - buffer:
                        out["bear_orb_seq"] = True
                        break

    return out



def _detect_rsi_divergence(
    df: pd.DataFrame,
    rsi_fast: pd.Series,
    rsi_slow: pd.Series | None = None,
    *,
    lookback: int = 160,
    pivot_lr: int = 3,
    min_price_delta_atr: float = 0.20,
    min_rsi_delta: float = 3.0,
) -> Optional[Dict[str, float | str]]:
    """Pivot-based RSI divergence with RSI-5 timing + RSI-14 validation.

    We use PRICE pivots (swing highs/lows) and compare RSI values at those pivots.
    - RSI-5 provides the timing (fast divergence signal)
    - RSI-14 acts as a validator (should not *contradict* the divergence)

    Bullish divergence:
      price pivot low2 < low1 by >= min_price_delta_atr * ATR
      AND RSI-5 at low2 > RSI-5 at low1 by >= min_rsi_delta
      AND RSI-14 at low2 >= RSI-14 at low1 - 1 (soft validation)

    Bearish divergence:
      price pivot high2 > high1 by >= min_price_delta_atr * ATR
      AND RSI-5 at high2 < RSI-5 at high1 by >= min_rsi_delta
      AND RSI-14 at high2 <= RSI-14 at high1 + 1 (soft validation)

    Returns dict like:
      {"type": "bull"|"bear", "strength": float, ...}
    """
    if df is None or len(df) < 25 or rsi_fast is None or len(rsi_fast) < 25:
        return None

    d = df.tail(int(min(max(60, lookback), len(df)))).copy()
    r5 = rsi_fast.reindex(d.index).ffill()
    if r5.isna().all():
        return None
    r14 = None
    if rsi_slow is not None:
        r14 = rsi_slow.reindex(d.index).ffill()

    # ATR for scaling (fallback to price*0.002 if missing)
    atr_last = None
    try:
        if "atr14" in d.columns and np.isfinite(float(d["atr14"].iloc[-1])):
            atr_last = float(d["atr14"].iloc[-1])
    except Exception:
        atr_last = None
    atr_scale = atr_last if (atr_last is not None and atr_last > 0) else float(d["close"].iloc[-1]) * 0.002

    # Price pivots
    lows_mask = rolling_swing_lows(d["low"], left=int(pivot_lr), right=int(pivot_lr))
    highs_mask = rolling_swing_highs(d["high"], left=int(pivot_lr), right=int(pivot_lr))
    piv_lows = d.loc[lows_mask, ["low"]].tail(6)
    piv_highs = d.loc[highs_mask, ["high"]].tail(6)

    # --- Bull divergence on the last two pivot lows ---
    if len(piv_lows) >= 2:
        a_idx = piv_lows.index[-2]
        b_idx = piv_lows.index[-1]
        p_a = float(d.loc[a_idx, "low"])
        p_b = float(d.loc[b_idx, "low"])
        r_a = float(r5.loc[a_idx])
        r_b = float(r5.loc[b_idx])

        price_ok = (p_b < p_a) and ((p_a - p_b) >= float(min_price_delta_atr) * atr_scale)
        rsi_ok = (r_b > r_a) and ((r_b - r_a) >= float(min_rsi_delta))
        slow_ok = True
        if r14 is not None and not r14.isna().all():
            try:
                s_a = float(r14.loc[a_idx])
                s_b = float(r14.loc[b_idx])
                slow_ok = (s_b >= s_a - 1.0)  # don't contradict
            except Exception:
                slow_ok = True

        if price_ok and rsi_ok and slow_ok:
            strength = float((r_b - r_a) / max(1.0, min_rsi_delta)) + float((p_a - p_b) / max(1e-9, atr_scale))
            return {"type": "bull", "strength": float(strength), "price_a": p_a, "price_b": p_b, "rsi_a": r_a, "rsi_b": r_b}

    # --- Bear divergence on the last two pivot highs ---
    if len(piv_highs) >= 2:
        a_idx = piv_highs.index[-2]
        b_idx = piv_highs.index[-1]
        p_a = float(d.loc[a_idx, "high"])
        p_b = float(d.loc[b_idx, "high"])
        r_a = float(r5.loc[a_idx])
        r_b = float(r5.loc[b_idx])

        price_ok = (p_b > p_a) and ((p_b - p_a) >= float(min_price_delta_atr) * atr_scale)
        rsi_ok = (r_b < r_a) and ((r_a - r_b) >= float(min_rsi_delta))
        slow_ok = True
        if r14 is not None and not r14.isna().all():
            try:
                s_a = float(r14.loc[a_idx])
                s_b = float(r14.loc[b_idx])
                slow_ok = (s_b <= s_a + 1.0)
            except Exception:
                slow_ok = True

        if price_ok and rsi_ok and slow_ok:
            strength = float((r_a - r_b) / max(1.0, min_rsi_delta)) + float((p_b - p_a) / max(1e-9, atr_scale))
            return {"type": "bear", "strength": float(strength), "price_a": p_a, "price_b": p_b, "rsi_a": r_a, "rsi_b": r_b}

    return None





def _is_deep_zone_touch(
    zone_low: float,
    zone_high: float,
    candle_low: float,
    candle_high: float,
    atr_v: float,
    zone_type: str,
) -> bool:
    """Count a later touch only if price penetrates meaningfully into the zone."""
    try:
        zl = float(zone_low); zh = float(zone_high); lo = float(candle_low); hi = float(candle_high); atr = float(atr_v)
    except Exception:
        return False
    zone_width = max(0.0, zh - zl)
    deep_thresh = max(0.30 * zone_width, 0.08 * max(atr, 1e-9))
    zt = str(zone_type).upper()
    if zt == "BULLISH_DEMAND":
        return bool(lo <= (zh - deep_thresh))
    if zt == "BEARISH_SUPPLY":
        return bool(hi >= (zl + deep_thresh))
    return False

def _evaluate_entry_zone_context(
    df: pd.DataFrame,
    *,
    entry_price: float | None,
    direction: str,
    atr_last: float | None,
    lookback: int = 10,
) -> dict:
    """Assess simple recent demand/supply zone context around a proposed entry.

    Uses only existing OHLCV data already fetched by the engine. A candle must show
    meaningful rejection *and* enough displacement/volume to qualify as a zone.
    Returns a small, local context signal that can be used as a score tilt without
    introducing new data dependencies or engine gridlock.
    """
    out = {
        "favorable": False,
        "hostile": False,
        "favorable_type": None,
        "hostile_type": None,
        "favorable_dist": None,
        "hostile_dist": None,
        "favorable_inside": False,
        "hostile_inside": False,
        "zone_score_adj": 0,
        "zone_ref_price": None,
        "zone_quality": None,
        "favorable_quality": None,
        "hostile_quality": None,
    }
    try:
        if entry_price is None or not np.isfinite(float(entry_price)):
            return out
        if df is None or len(df) < 6:
            return out
        ep = float(entry_price)
        atr_v = float(atr_last) if atr_last is not None and np.isfinite(float(atr_last)) and float(atr_last) > 0 else max(1e-6, 0.005 * max(ep, 1.0))
        prox = max(0.18 * atr_v, 0.001 * max(ep, 1.0))

        sub = df.tail(int(max(6, lookback)) + 2).copy()
        vol = sub["volume"].astype(float) if "volume" in sub.columns else None
        vol_sma = vol.rolling(5, min_periods=3).mean() if vol is not None else None
        start = 0
        end = len(sub) - 1
        bull_zones = []
        bear_zones = []
        for i in range(start, end):
            o = float(sub["open"].iloc[i]); h = float(sub["high"].iloc[i]); l = float(sub["low"].iloc[i]); c = float(sub["close"].iloc[i])
            rng = max(1e-9, h - l)
            body = abs(c - o)
            upper = h - max(o, c)
            lower = min(o, c) - l
            close_pos = (c - l) / rng
            next_df = sub.iloc[i + 1 : min(len(sub), i + 3)]
            next_closes = next_df["close"].astype(float) if len(next_df) else pd.Series(dtype=float)
            next_lows = next_df["low"].astype(float) if len(next_df) else pd.Series(dtype=float)
            next_highs = next_df["high"].astype(float) if len(next_df) else pd.Series(dtype=float)

            disp_ok = bool(rng >= 0.60 * atr_v)
            vol_ok = True
            if vol is not None and vol_sma is not None and i < len(vol_sma):
                v = float(vol.iloc[i]) if np.isfinite(vol.iloc[i]) else np.nan
                vs = float(vol_sma.iloc[i]) if np.isfinite(vol_sma.iloc[i]) else np.nan
                if np.isfinite(v) and np.isfinite(vs) and vs > 0:
                    vol_ok = bool(v >= 1.20 * vs)

            bull_shape = bool(
                lower >= max(1.5 * body, 0.35 * rng)
                and close_pos >= 0.45
            )
            bear_shape = bool(
                upper >= max(1.5 * body, 0.35 * rng)
                and close_pos <= 0.55
            )

            bull_cand = bool(
                bull_shape
                and disp_ok
                and vol_ok
                and len(next_closes) > 0
                and float(next_closes.max()) >= (max(o, c) - 0.05 * rng)
                and float(next_lows.min()) >= (l - 0.08 * atr_v)
            )
            bear_cand = bool(
                bear_shape
                and disp_ok
                and vol_ok
                and len(next_closes) > 0
                and float(next_closes.min()) <= (min(o, c) + 0.05 * rng)
                and float(next_highs.max()) <= (h + 0.08 * atr_v)
            )

            if bull_cand:
                zone_low = l
                zone_high = max(o, c)
                dist = 0.0 if zone_low - prox <= ep <= zone_high + prox else min(abs(ep - zone_low), abs(ep - zone_high))
                disp_move = float(next_closes.max() - c) if len(next_closes) > 0 else 0.0
                disp_score = float(min(1.0, max(0.0, disp_move) / max(1e-9, 1.5 * atr_v)))
                v = float(vol.iloc[i]) if (vol is not None and i < len(vol) and np.isfinite(vol.iloc[i])) else np.nan
                vs = float(vol_sma.iloc[i]) if (vol_sma is not None and i < len(vol_sma) and np.isfinite(vol_sma.iloc[i])) else np.nan
                vol_score = float(min(1.0, (v / vs) / 1.5)) if np.isfinite(v) and np.isfinite(vs) and vs > 0 else 0.5
                later_df = sub.iloc[i + 1 :]
                touch_count = int(sum(
                    1 for _, rr in later_df.iterrows()
                    if _is_deep_zone_touch(
                        zone_low,
                        zone_high,
                        float(rr["low"]),
                        float(rr["high"]),
                        atr_v,
                        "BULLISH_DEMAND",
                    )
                )) if len(later_df) else 0
                fresh_score = float(max(0.0, 1.0 - 0.25 * max(0, touch_count - 1)))
                zone_width = max(1e-9, zone_high - zone_low)
                precision_score = float(max(0.0, min(1.0, 1.0 - (zone_width / max(1e-9, 0.80 * atr_v)))))
                hold_score = float(max(0.0, min(1.0, (float(next_closes.iloc[-1]) - zone_high) / max(1e-9, 0.90 * atr_v)))) if len(next_closes) > 0 else 0.0
                reaction_score = float(max(0.0, min(1.0, 0.55 * disp_score + 0.45 * hold_score)))
                zone_quality = float(max(0.0, min(1.0, 0.30 * disp_score + 0.20 * vol_score + 0.20 * fresh_score + 0.20 * reaction_score + 0.10 * precision_score)))
                inside = bool(zone_low <= ep <= zone_high)
                bull_zones.append({"type": "BULLISH_DEMAND", "dist": float(dist), "ref": float(zone_high), "i": i, "quality": zone_quality, "inside": inside})
            if bear_cand:
                zone_low = min(o, c)
                zone_high = h
                dist = 0.0 if zone_low - prox <= ep <= zone_high + prox else min(abs(ep - zone_low), abs(ep - zone_high))
                disp_move = float(c - next_closes.min()) if len(next_closes) > 0 else 0.0
                disp_score = float(min(1.0, max(0.0, disp_move) / max(1e-9, 1.5 * atr_v)))
                v = float(vol.iloc[i]) if (vol is not None and i < len(vol) and np.isfinite(vol.iloc[i])) else np.nan
                vs = float(vol_sma.iloc[i]) if (vol_sma is not None and i < len(vol_sma) and np.isfinite(vol_sma.iloc[i])) else np.nan
                vol_score = float(min(1.0, (v / vs) / 1.5)) if np.isfinite(v) and np.isfinite(vs) and vs > 0 else 0.5
                later_df = sub.iloc[i + 1 :]
                touch_count = int(sum(
                    1 for _, rr in later_df.iterrows()
                    if _is_deep_zone_touch(
                        zone_low,
                        zone_high,
                        float(rr["low"]),
                        float(rr["high"]),
                        atr_v,
                        "BEARISH_SUPPLY",
                    )
                )) if len(later_df) else 0
                fresh_score = float(max(0.0, 1.0 - 0.25 * max(0, touch_count - 1)))
                zone_width = max(1e-9, zone_high - zone_low)
                precision_score = float(max(0.0, min(1.0, 1.0 - (zone_width / max(1e-9, 0.80 * atr_v)))))
                hold_score = float(max(0.0, min(1.0, (zone_low - float(next_closes.iloc[-1])) / max(1e-9, 0.90 * atr_v)))) if len(next_closes) > 0 else 0.0
                reaction_score = float(max(0.0, min(1.0, 0.55 * disp_score + 0.45 * hold_score)))
                zone_quality = float(max(0.0, min(1.0, 0.30 * disp_score + 0.20 * vol_score + 0.20 * fresh_score + 0.20 * reaction_score + 0.10 * precision_score)))
                inside = bool(zone_low <= ep <= zone_high)
                bear_zones.append({"type": "BEARISH_SUPPLY", "dist": float(dist), "ref": float(zone_low), "i": i, "quality": zone_quality, "inside": inside})

        best_bull = min(bull_zones, key=lambda z: (z["dist"], -z["i"])) if bull_zones else None
        best_bear = min(bear_zones, key=lambda z: (z["dist"], -z["i"])) if bear_zones else None

        if str(direction).upper() == "LONG":
            if best_bull and best_bull["dist"] <= prox:
                out["favorable"] = True
                out["favorable_type"] = best_bull["type"]
                out["favorable_dist"] = float(best_bull["dist"])
                out["favorable_quality"] = float(best_bull.get("quality", 0.0))
                out["favorable_inside"] = bool(best_bull.get("inside", False))
                out["zone_ref_price"] = float(best_bull["ref"])
                out["zone_quality"] = float(best_bull.get("quality", 0.0))
            if best_bear and best_bear["dist"] <= prox:
                out["hostile"] = True
                out["hostile_type"] = best_bear["type"]
                out["hostile_dist"] = float(best_bear["dist"])
                out["hostile_quality"] = float(best_bear.get("quality", 0.0))
                out["hostile_inside"] = bool(best_bear.get("inside", False))
                out["zone_ref_price"] = float(best_bear["ref"])
                out["zone_quality"] = float(best_bear.get("quality", 0.0))
        else:
            if best_bear and best_bear["dist"] <= prox:
                out["favorable"] = True
                out["favorable_type"] = best_bear["type"]
                out["favorable_dist"] = float(best_bear["dist"])
                out["favorable_quality"] = float(best_bear.get("quality", 0.0))
                out["favorable_inside"] = bool(best_bear.get("inside", False))
                out["zone_ref_price"] = float(best_bear["ref"])
                out["zone_quality"] = float(best_bear.get("quality", 0.0))
            if best_bull and best_bull["dist"] <= prox:
                out["hostile"] = True
                out["hostile_type"] = best_bull["type"]
                out["hostile_dist"] = float(best_bull["dist"])
                out["hostile_quality"] = float(best_bull.get("quality", 0.0))
                out["hostile_inside"] = bool(best_bull.get("inside", False))
                out["zone_ref_price"] = float(best_bull["ref"])
                out["zone_quality"] = float(best_bull.get("quality", 0.0))
    except Exception:
        return out
    return out

def _compute_atr_pct_series(df: pd.DataFrame, period: int = 14):
    if df is None or len(df) < period + 2:
        return None
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr / close.replace(0, np.nan)


def _apply_atr_score_normalization(score: float, df: pd.DataFrame, lookback: int = 200, period: int = 14):
    atr_pct = _compute_atr_pct_series(df, period=period)
    if atr_pct is None:
        return score, None, None, 1.0
    cur = atr_pct.iloc[-1]
    if pd.isna(cur) or float(cur) <= 0:
        return score, (None if pd.isna(cur) else float(cur)), None, 1.0
    tail = atr_pct.dropna().tail(int(lookback))
    baseline = float(tail.median()) if len(tail) else None
    if baseline is None or baseline <= 0:
        return score, float(cur), baseline, 1.0
    scale = float(baseline / float(cur))
    scale = max(0.75, min(1.35, scale))
    return max(0.0, min(100.0, float(score) * scale)), float(cur), baseline, scale


def _tape_bonus_from_readiness(
    readiness: float,
    *,
    cap: int = 4,
    thresholds: tuple[float, float, float, float] = (4.0, 5.5, 7.0, 8.0),
) -> int:
    try:
        r = float(readiness)
    except Exception:
        return 0
    t1, t2, t3, t4 = [float(x) for x in thresholds]
    if r >= t4:
        return int(min(cap, 4))
    if r >= t3:
        return int(min(cap, 3))
    if r >= t2:
        return int(min(cap, 2))
    if r >= t1:
        return int(min(cap, 1))
    return 0


def _compute_tape_readiness(
    df: pd.DataFrame,
    *,
    direction: str,
    atr_last: float | None,
    release_level: float | None,
    structural_level: float | None = None,
    trigger_near: bool = False,
    baseline_ok: bool = False,
) -> Dict[str, float | bool | None]:
    """Small, behavior-first tape diagnostic for chaotic $1-$5 tape.

    Uses only already-fetched OHLCV + indicator context. It intentionally avoids
    explicit pattern labels and instead scores the behavior behind useful coils:
      - tightening
      - structural hold
      - directional pressure
      - proximity to a meaningful release area
    """
    out: Dict[str, float | bool | None] = {
        "eligible": False,
        "tightening": 0.0,
        "structural_hold": 0.0,
        "pressure": 0.0,
        "release_proximity": 0.0,
        "readiness": 0.0,
        "recent_range_ratio": None,
        "recent_body_ratio": None,
        "release_dist_atr": None,
        "prior_impulse_span_atr": None,
        "prior_impulse_push_atr": None,
        "impulse_floor_ok": False,
        "macd_directional_build": False,
    }
    try:
        if df is None or len(df) < 12 or (not baseline_ok):
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float("nan")
        if not np.isfinite(atr_val) or atr_val <= 0:
            return out
        release = float(release_level) if release_level is not None and np.isfinite(release_level) else float("nan")
        if not np.isfinite(release):
            return out
        structural = float(structural_level) if structural_level is not None and np.isfinite(structural_level) else release
        direction = str(direction or "").upper().strip()

        recent = df.tail(5).copy()
        prior = df.iloc[-10:-5].copy()
        if len(recent) < 4 or len(prior) < 3:
            return out

        recent_range = pd.to_numeric(recent["high"] - recent["low"], errors="coerce").dropna()
        prior_range = pd.to_numeric(prior["high"] - prior["low"], errors="coerce").dropna()
        recent_body = pd.to_numeric((recent["close"] - recent["open"]).abs(), errors="coerce").dropna()
        prior_body = pd.to_numeric((prior["close"] - prior["open"]).abs(), errors="coerce").dropna()
        if len(recent_range) < 3 or len(prior_range) < 3:
            return out

        rr_ratio = float(recent_range.mean() / max(1e-9, float(prior_range.mean())))
        rb_ratio = float(recent_body.mean() / max(1e-9, float(prior_body.mean()))) if len(prior_body) else float("nan")
        out["recent_range_ratio"] = rr_ratio
        out["recent_body_ratio"] = rb_ratio if np.isfinite(rb_ratio) else None

        prior_highs = pd.to_numeric(prior["high"], errors="coerce").dropna()
        prior_lows = pd.to_numeric(prior["low"], errors="coerce").dropna()
        prior_opens = pd.to_numeric(prior["open"], errors="coerce").dropna()
        prior_closes = pd.to_numeric(prior["close"], errors="coerce").dropna()
        if not len(prior_highs) or not len(prior_lows) or not len(prior_opens) or not len(prior_closes):
            return out
        prior_span_atr = float((prior_highs.max() - prior_lows.min()) / max(1e-9, atr_val))
        if direction == "LONG":
            prior_push_atr = float((prior_closes.iloc[-1] - prior_opens.iloc[0]) / max(1e-9, atr_val))
            impulse_floor_ok = bool(prior_span_atr >= 1.10 and prior_push_atr >= 0.35)
        else:
            prior_push_atr = float((prior_opens.iloc[0] - prior_closes.iloc[-1]) / max(1e-9, atr_val))
            impulse_floor_ok = bool(prior_span_atr >= 1.10 and prior_push_atr >= 0.35)
        out["prior_impulse_span_atr"] = float(prior_span_atr)
        out["prior_impulse_push_atr"] = float(prior_push_atr)
        out["impulse_floor_ok"] = bool(impulse_floor_ok)
        if not impulse_floor_ok:
            return out

        tightening = 0.0
        if rr_ratio <= 0.82:
            tightening += 1.0
        if np.isfinite(rb_ratio) and rb_ratio <= 0.88:
            tightening += 0.5
        if len(recent_range) >= 3 and float(recent_range.iloc[-3:].mean()) <= float(recent_range.iloc[:2].mean()) * 0.92:
            tightening += 0.5
        tightening = float(np.clip(tightening, 0.0, 2.0))

        lows = pd.to_numeric(recent["low"], errors="coerce")
        highs = pd.to_numeric(recent["high"], errors="coerce")
        closes = pd.to_numeric(recent["close"], errors="coerce")
        macd_tail = pd.to_numeric(df.get("macd_hist", pd.Series(index=df.index, dtype=float)).tail(4), errors="coerce").dropna()
        rsi5_tail = pd.to_numeric(df.get("rsi5", pd.Series(index=df.index, dtype=float)).tail(3), errors="coerce").dropna()
        rsi14_tail = pd.to_numeric(df.get("rsi14", pd.Series(index=df.index, dtype=float)).tail(3), errors="coerce").dropna()

        structural_hold = 0.0
        pressure = 0.0
        macd_directional_build = False
        if direction == "LONG":
            if len(lows.dropna()) >= 4 and int((lows.diff().fillna(0.0) >= (-0.12 * atr_val)).sum()) >= 4:
                structural_hold += 1.0
            if float(closes.min()) >= float(structural) - 0.35 * atr_val:
                structural_hold += 1.0
            if len(macd_tail) >= 3 and bool(macd_tail.iloc[-1] > macd_tail.iloc[-2] > macd_tail.iloc[-3]):
                pressure += 1.0
                macd_directional_build = True
            recent_span = max(1e-9, float(highs.max() - lows.min()))
            if float(closes.mean()) >= float(lows.min()) + 0.58 * recent_span:
                pressure += 0.5
            retrace_span = max(1e-9, float(df["close"].tail(6).max() - df["close"].tail(6).min()))
            if float(df["close"].tail(3).min()) >= float(df["close"].tail(6).max()) - 0.75 * retrace_span:
                pressure += 0.5
            if macd_directional_build and len(rsi5_tail) and len(rsi14_tail):
                if float(rsi5_tail.iloc[-1]) >= 42.0 and float(rsi14_tail.iloc[-1]) >= 40.0:
                    pressure += 0.25
        else:
            if len(highs.dropna()) >= 4 and int((highs.diff().fillna(0.0) <= (0.12 * atr_val)).sum()) >= 4:
                structural_hold += 1.0
            if float(closes.max()) <= float(structural) + 0.35 * atr_val:
                structural_hold += 1.0
            if len(macd_tail) >= 3 and bool(macd_tail.iloc[-1] < macd_tail.iloc[-2] < macd_tail.iloc[-3]):
                pressure += 1.0
                macd_directional_build = True
            recent_span = max(1e-9, float(highs.max() - lows.min()))
            if float(closes.mean()) <= float(highs.max()) - 0.58 * recent_span:
                pressure += 0.5
            retrace_span = max(1e-9, float(df["close"].tail(6).max() - df["close"].tail(6).min()))
            if float(df["close"].tail(3).max()) <= float(df["close"].tail(6).min()) + 0.75 * retrace_span:
                pressure += 0.5
            if macd_directional_build and len(rsi5_tail) and len(rsi14_tail):
                if float(rsi5_tail.iloc[-1]) <= 58.0 and float(rsi14_tail.iloc[-1]) <= 60.0:
                    pressure += 0.25
        structural_hold = float(np.clip(structural_hold, 0.0, 2.0))
        pressure = float(np.clip(pressure, 0.0, 2.0))
        out["macd_directional_build"] = bool(macd_directional_build)

        last_close = float(pd.to_numeric(df["close"], errors="coerce").iloc[-1])
        release_dist_atr = abs(last_close - release) / max(1e-9, atr_val)
        out["release_dist_atr"] = float(release_dist_atr)
        release_prox = 0.0
        if trigger_near or release_dist_atr <= 0.75:
            release_prox += 1.0
        if release_dist_atr <= 0.40:
            release_prox += 1.0
        release_prox = float(np.clip(release_prox, 0.0, 2.0))

        readiness = 0.0
        pressure_floor_ok = bool(pressure >= 1.0)
        high_readiness_ok = bool((pressure >= 1.5) and macd_directional_build)
        if tightening >= 0.5 and structural_hold >= 0.5 and pressure_floor_ok and release_prox >= 0.5:
            readiness = float(np.clip(tightening + structural_hold + pressure + release_prox, 0.0, 8.0))
            if readiness >= 6.0 and not high_readiness_ok:
                readiness = min(readiness, 5.5)

        out.update({
            "eligible": bool(readiness > 0.0),
            "tightening": float(tightening),
            "structural_hold": float(structural_hold),
            "pressure": float(pressure),
            "release_proximity": float(release_prox),
            "readiness": float(readiness),
        })
        return out
    except Exception:
        return out



def _compute_scalp_reversal_stabilization(
    df: pd.DataFrame,
    *,
    direction: str,
    ref_level: float | None,
    atr_last: float | None,
) -> Dict[str, float | bool | None]:
    """Reversal-specific stabilizer for SCALP PRE assistance.

    Lets SCALP benefit from post-flush stabilization / reclaim lean without
    demanding the same continuation-style pressure required by RIDE breakout
    selection.
    """
    out: Dict[str, float | bool | None] = {
        "stabilizing": False,
        "flush_present": False,
        "cluster_improving": False,
        "reclaim_lean": False,
        "bonus": 0.0,
    }
    try:
        if df is None or len(df) < 8:
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float("nan")
        ref = float(ref_level) if ref_level is not None and np.isfinite(ref_level) else float("nan")
        if not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(ref):
            return out
        recent = df.tail(4).copy()
        prior = df.iloc[-8:-4].copy()
        if len(recent) < 4 or len(prior) < 4:
            return out
        o_r = pd.to_numeric(recent["open"], errors="coerce").dropna()
        h_r = pd.to_numeric(recent["high"], errors="coerce").dropna()
        l_r = pd.to_numeric(recent["low"], errors="coerce").dropna()
        c_r = pd.to_numeric(recent["close"], errors="coerce").dropna()
        o_p = pd.to_numeric(prior["open"], errors="coerce").dropna()
        h_p = pd.to_numeric(prior["high"], errors="coerce").dropna()
        l_p = pd.to_numeric(prior["low"], errors="coerce").dropna()
        c_p = pd.to_numeric(prior["close"], errors="coerce").dropna()
        macd_tail = pd.to_numeric(df.get("macd_hist", pd.Series(index=df.index, dtype=float)).tail(4), errors="coerce").dropna()
        if min(len(o_r), len(h_r), len(l_r), len(c_r), len(o_p), len(h_p), len(l_p), len(c_p)) < 3:
            return out
        direction = str(direction or "").upper().strip()
        if direction == "LONG":
            flush_present = bool(((o_p.iloc[0] - l_p.min()) / max(1e-9, atr_val) >= 0.50) or (c_p.iloc[-1] < o_p.iloc[0] - 0.30 * atr_val))
            lows_stable = bool(l_r.min() >= l_p.min() - 0.10 * atr_val)
            closes_improving = bool(c_r.iloc[-1] >= c_r.iloc[0] - 0.04 * atr_val and c_r.mean() >= l_r.min() + 0.52 * max(1e-9, float(h_r.max() - l_r.min())))
            reclaim_lean = bool(c_r.iloc[-1] >= ref - 0.34 * atr_val or h_r.max() >= ref - 0.15 * atr_val)
            macd_improving = bool(len(macd_tail) >= 3 and macd_tail.iloc[-1] >= macd_tail.iloc[-2] >= macd_tail.iloc[-3])
        else:
            flush_present = bool(((h_p.max() - o_p.iloc[0]) / max(1e-9, atr_val) >= 0.50) or (c_p.iloc[-1] > o_p.iloc[0] + 0.30 * atr_val))
            lows_stable = bool(h_r.max() <= h_p.max() + 0.10 * atr_val)
            closes_improving = bool(c_r.iloc[-1] <= c_r.iloc[0] + 0.04 * atr_val and c_r.mean() <= h_r.max() - 0.52 * max(1e-9, float(h_r.max() - l_r.min())))
            reclaim_lean = bool(c_r.iloc[-1] <= ref + 0.34 * atr_val or l_r.min() <= ref + 0.15 * atr_val)
            macd_improving = bool(len(macd_tail) >= 3 and macd_tail.iloc[-1] <= macd_tail.iloc[-2] <= macd_tail.iloc[-3])
        bonus = 0.0
        if flush_present:
            bonus += 0.30
        if lows_stable:
            bonus += 0.25
        if closes_improving:
            bonus += 0.25
        if reclaim_lean:
            bonus += 0.20
        if macd_improving:
            bonus += 0.15
        bonus = float(min(1.0, bonus))
        out.update({
            "stabilizing": bool(flush_present and lows_stable and closes_improving),
            "flush_present": bool(flush_present),
            "cluster_improving": bool(closes_improving and macd_improving),
            "reclaim_lean": bool(reclaim_lean),
            "bonus": float(bonus),
        })
        return out
    except Exception:
        return out



def _detect_scalp_reversal_trigger(
    df: pd.DataFrame,
    *,
    direction: str,
    atr_last: float | None,
    ref_level: float | None = None,
) -> Dict[str, float | bool | str | None]:
    """Structure-first SCALP reversal trigger.

    This is intentionally not a full confirmation model. It identifies the
    candle/structure moment a trader would act on after an exhaustion move:
    hammer/pin, engulfing/outside reversal, failed breakdown/breakout,
    strong rejection close, absorption stall, or first micro higher-low/lower-high.
    Momentum indicators are supporting context, not mandatory gates.
    """
    out: Dict[str, float | bool | str | None] = {
        "trigger": False,
        "trigger_type": None,
        "score": 0.0,
        "exhaustion_present": False,
        "soft_validation": False,
        "entry_timing": None,
        "entry_anchor": None,
        "structure_ref": None,
        "near_ref": False,
    }
    try:
        if df is None or len(df) < 7:
            return out
        atr = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float('nan')
        if not np.isfinite(atr) or atr <= 0:
            atr = float(max(1e-9, pd.to_numeric(df['high'].tail(8), errors='coerce').max() - pd.to_numeric(df['low'].tail(8), errors='coerce').min()) / 4.0)
        if not np.isfinite(atr) or atr <= 0:
            return out
        d = df.tail(8).copy()
        o = pd.to_numeric(d['open'], errors='coerce').reset_index(drop=True)
        h = pd.to_numeric(d['high'], errors='coerce').reset_index(drop=True)
        l = pd.to_numeric(d['low'], errors='coerce').reset_index(drop=True)
        c = pd.to_numeric(d['close'], errors='coerce').reset_index(drop=True)
        v = pd.to_numeric(d['volume'], errors='coerce').reset_index(drop=True) if 'volume' in d else pd.Series(dtype=float)
        if min(o.notna().sum(), h.notna().sum(), l.notna().sum(), c.notna().sum()) < 7:
            return out
        direction = str(direction or '').upper().strip()
        i = len(c) - 1
        body = abs(float(c.iloc[i]) - float(o.iloc[i]))
        rng = max(1e-9, float(h.iloc[i]) - float(l.iloc[i]))
        prev_body = abs(float(c.iloc[i-1]) - float(o.iloc[i-1]))
        lower_wick = min(float(o.iloc[i]), float(c.iloc[i])) - float(l.iloc[i])
        upper_wick = float(h.iloc[i]) - max(float(o.iloc[i]), float(c.iloc[i]))
        close_loc_long = (float(c.iloc[i]) - float(l.iloc[i])) / rng
        close_loc_short = (float(h.iloc[i]) - float(c.iloc[i])) / rng
        prior_low = float(l.iloc[:-1].min())
        prior_high = float(h.iloc[:-1].max())
        recent_low = float(l.iloc[-4:].min())
        recent_high = float(h.iloc[-4:].max())
        ref = float(ref_level) if ref_level is not None and np.isfinite(ref_level) else float('nan')
        near_ref = bool(np.isfinite(ref) and abs(float(c.iloc[i]) - ref) <= 0.75 * atr)

        # Soft indicator context: helpful, never sufficient alone.
        macd_tail = pd.to_numeric(df.get('macd_hist', pd.Series(index=df.index, dtype=float)).tail(4), errors='coerce').dropna()
        rsi_tail = pd.to_numeric(df.get('rsi5', pd.Series(index=df.index, dtype=float)).tail(4), errors='coerce').dropna()
        if direction == 'LONG':
            macd_ok = bool(len(macd_tail) >= 3 and macd_tail.iloc[-1] >= macd_tail.iloc[-2] >= macd_tail.iloc[-3])
            rsi_ok = bool(len(rsi_tail) >= 3 and rsi_tail.iloc[-1] >= rsi_tail.iloc[-2] >= rsi_tail.iloc[-3])
        else:
            macd_ok = bool(len(macd_tail) >= 3 and macd_tail.iloc[-1] <= macd_tail.iloc[-2] <= macd_tail.iloc[-3])
            rsi_ok = bool(len(rsi_tail) >= 3 and rsi_tail.iloc[-1] <= rsi_tail.iloc[-2] <= rsi_tail.iloc[-3])
        vol_absorb = False
        if len(v) >= 5 and v.notna().sum() >= 5:
            v_last = float(v.iloc[-1]); v_prev = float(v.iloc[-2]); v_base = float(v.iloc[:-2].replace(0, pd.NA).dropna().median()) if not v.iloc[:-2].replace(0, pd.NA).dropna().empty else 0.0
            if v_base > 0:
                vol_absorb = bool(v_prev >= 1.15 * v_base and v_last >= 0.55 * v_base)
        soft_validation = bool(macd_ok or rsi_ok or vol_absorb)

        # Require a real directional move into the reversal trigger. A hammer,
        # engulfing candle, or failed break is only actionable for SCALP after
        # actual pressure has pushed price far enough to create reversal edge.
        # This keeps the trigger from buying/selling random candles in chop.
        def _directional_exhaustion(dirn: str) -> bool:
            try:
                win_start = max(0, i - 6)
                pre_o = o.iloc[win_start:i]
                pre_h = h.iloc[win_start:i]
                pre_l = l.iloc[win_start:i]
                pre_c = c.iloc[win_start:i]
                if len(pre_c) < 5:
                    return False
                last_px = max(1e-9, abs(float(c.iloc[i])))
                min_move = max(0.0030 * last_px, 0.33 * atr)
                ranges = (pre_h - pre_l).replace(0, np.nan)
                bodies = (pre_c - pre_o).abs()
                body_q = (bodies / ranges).fillna(0.0)
                if str(dirn).upper() == 'LONG':
                    directional_travel = max(float(pre_c.iloc[0]) - float(pre_l.min()), float(pre_c.iloc[0]) - float(pre_c.iloc[-1]))
                    red_candles = int((pre_c < pre_o).sum())
                    strong_red = int(((pre_c < pre_o) & (body_q >= 0.38)).sum())
                    lower_low_steps = int((pre_l.diff().dropna() < -0.03 * atr).sum())
                    sweep_reclaim = bool(float(l.iloc[i]) <= float(pre_l.min()) - 0.05 * atr and float(c.iloc[i]) >= float(pre_l.min()) + 0.08 * atr)
                    pressure_ok = bool((red_candles >= 3 and strong_red >= 1) or strong_red >= 2 or lower_low_steps >= 2 or sweep_reclaim)
                    return bool(directional_travel >= min_move and pressure_ok)
                else:
                    directional_travel = max(float(pre_h.max()) - float(pre_c.iloc[0]), float(pre_c.iloc[-1]) - float(pre_c.iloc[0]))
                    green_candles = int((pre_c > pre_o).sum())
                    strong_green = int(((pre_c > pre_o) & (body_q >= 0.38)).sum())
                    higher_high_steps = int((pre_h.diff().dropna() > 0.03 * atr).sum())
                    sweep_reject = bool(float(h.iloc[i]) >= float(pre_h.max()) + 0.05 * atr and float(c.iloc[i]) <= float(pre_h.max()) - 0.08 * atr)
                    pressure_ok = bool((green_candles >= 3 and strong_green >= 1) or strong_green >= 2 or higher_high_steps >= 2 or sweep_reject)
                    return bool(directional_travel >= min_move and pressure_ok)
            except Exception:
                return False

        trigger_types: list[str] = []
        score = 0.0
        if direction == 'LONG':
            exhaustion = bool(_directional_exhaustion('LONG'))
            hammer = bool(lower_wick >= max(1.6 * max(body, 1e-9), 0.34 * rng) and close_loc_long >= 0.55 and upper_wick <= 0.55 * rng)
            rejection_close = bool(lower_wick >= 0.30 * rng and close_loc_long >= 0.62 and float(c.iloc[i]) >= float(o.iloc[i]) - 0.08 * atr)
            engulfing = bool(float(c.iloc[i]) > float(o.iloc[i]) and float(c.iloc[i-1]) < float(o.iloc[i-1]) and float(c.iloc[i]) >= float(o.iloc[i-1]) - 0.03 * atr and body >= 0.75 * max(prev_body, 1e-9))
            outside_rev = bool(float(l.iloc[i]) <= float(l.iloc[i-1]) - 0.03 * atr and float(c.iloc[i]) > float(h.iloc[i-1]) - 0.10 * atr and close_loc_long >= 0.58)
            failed_break = bool(float(l.iloc[i]) <= prior_low + 0.03 * atr and float(c.iloc[i]) >= prior_low + 0.12 * atr)
            absorption = bool(exhaustion and float(l.iloc[-1]) >= float(l.iloc[-2]) - 0.06 * atr and float(c.iloc[-1]) >= float(l.iloc[-2]) + 0.32 * atr and (vol_absorb or lower_wick >= 0.24 * rng))
            micro_hl = bool(float(l.iloc[-1]) >= float(l.iloc[-2]) - 0.06 * atr and float(l.iloc[-2]) >= float(l.iloc[-3]) - 0.12 * atr and float(c.iloc[-1]) > float(c.iloc[-2]) + 0.03 * atr)
            # Structural reversal catches the PLUG/CRWG style setup: a real flush,
            # then selling stops making progress and buyers step in over multiple bars.
            # This is intentionally separate from hammer/engulfing candles so SCALP
            # can catch gradual bases, not only violent one-candle reversals.
            _base_range = max(1e-9, recent_high - recent_low)
            _higher_low_cluster = bool(
                (float(l.iloc[-1]) >= float(l.iloc[-2]) - 0.08 * atr and float(l.iloc[-2]) >= float(l.iloc[-3]) - 0.12 * atr)
                or float(l.iloc[-1]) >= float(l.iloc[-3]) + 0.05 * atr
            )
            _higher_close_cluster = bool(
                (float(c.iloc[-1]) > float(c.iloc[-2]) + 0.02 * atr and float(c.iloc[-2]) >= float(c.iloc[-3]) - 0.05 * atr)
                or float(c.iloc[-1]) > float(c.iloc[-3]) + 0.10 * atr
            )
            _recovered_from_low = bool(
                float(c.iloc[-1]) >= recent_low + 0.30 * atr
                or ((float(c.iloc[-1]) - recent_low) / _base_range) >= 0.45
            )
            _buyer_body_support = bool(float(c.iloc[-1]) > float(o.iloc[-1]) or float(c.iloc[-2]) > float(o.iloc[-2]))
            structural_reversal = bool(_higher_low_cluster and _higher_close_cluster and _recovered_from_low and _buyer_body_support)
            checks = [
                (hammer,'HAMMER_PIN',2.0),
                (rejection_close,'REJECTION_CLOSE',2.0),
                (engulfing,'BULL_ENGULF',3.0),
                (outside_rev,'OUTSIDE_REV',3.0),
                (failed_break,'FAILED_BREAKDOWN',4.0),
                (micro_hl,'MICRO_HIGHER_LOW',3.0),
                (structural_reversal,'STRUCTURAL_REVERSAL',3.0),
                (absorption,'ABSORPTION_STALL',1.0),
            ]
            structure_ref = min(recent_low, float(l.iloc[i]))
            entry_anchor = float(c.iloc[i])
        else:
            exhaustion = bool(_directional_exhaustion('SHORT'))
            shooting = bool(upper_wick >= max(1.6 * max(body, 1e-9), 0.34 * rng) and close_loc_short >= 0.55 and lower_wick <= 0.55 * rng)
            rejection_close = bool(upper_wick >= 0.30 * rng and close_loc_short >= 0.62 and float(c.iloc[i]) <= float(o.iloc[i]) + 0.08 * atr)
            engulfing = bool(float(c.iloc[i]) < float(o.iloc[i]) and float(c.iloc[i-1]) > float(o.iloc[i-1]) and float(c.iloc[i]) <= float(o.iloc[i-1]) + 0.03 * atr and body >= 0.75 * max(prev_body, 1e-9))
            outside_rev = bool(float(h.iloc[i]) >= float(h.iloc[i-1]) + 0.03 * atr and float(c.iloc[i]) < float(l.iloc[i-1]) + 0.10 * atr and close_loc_short >= 0.58)
            failed_break = bool(float(h.iloc[i]) >= prior_high - 0.03 * atr and float(c.iloc[i]) <= prior_high - 0.12 * atr)
            absorption = bool(exhaustion and float(h.iloc[-1]) <= float(h.iloc[-2]) + 0.06 * atr and float(c.iloc[-1]) <= float(h.iloc[-2]) - 0.32 * atr and (vol_absorb or upper_wick >= 0.24 * rng))
            micro_hl = bool(float(h.iloc[-1]) <= float(h.iloc[-2]) + 0.06 * atr and float(h.iloc[-2]) <= float(h.iloc[-3]) + 0.12 * atr and float(c.iloc[-1]) < float(c.iloc[-2]) - 0.03 * atr)
            # Bearish structural reversal mirrors the long side: real upside pressure,
            # then buyers stop making progress and sellers step in over multiple bars.
            _base_range = max(1e-9, recent_high - recent_low)
            _lower_high_cluster = bool(
                (float(h.iloc[-1]) <= float(h.iloc[-2]) + 0.08 * atr and float(h.iloc[-2]) <= float(h.iloc[-3]) + 0.12 * atr)
                or float(h.iloc[-1]) <= float(h.iloc[-3]) - 0.05 * atr
            )
            _lower_close_cluster = bool(
                (float(c.iloc[-1]) < float(c.iloc[-2]) - 0.02 * atr and float(c.iloc[-2]) <= float(c.iloc[-3]) + 0.05 * atr)
                or float(c.iloc[-1]) < float(c.iloc[-3]) - 0.10 * atr
            )
            _rejected_from_high = bool(
                float(c.iloc[-1]) <= recent_high - 0.30 * atr
                or ((recent_high - float(c.iloc[-1])) / _base_range) >= 0.45
            )
            _seller_body_support = bool(float(c.iloc[-1]) < float(o.iloc[-1]) or float(c.iloc[-2]) < float(o.iloc[-2]))
            structural_reversal = bool(_lower_high_cluster and _lower_close_cluster and _rejected_from_high and _seller_body_support)
            checks = [
                (shooting,'SHOOTING_PIN',2.0),
                (rejection_close,'REJECTION_CLOSE',2.0),
                (engulfing,'BEAR_ENGULF',3.0),
                (outside_rev,'OUTSIDE_REV',3.0),
                (failed_break,'FAILED_BREAKOUT',4.0),
                (micro_hl,'MICRO_LOWER_HIGH',3.0),
                (structural_reversal,'STRUCTURAL_REVERSAL',3.0),
                (absorption,'ABSORPTION_STALL',1.0),
            ]
            structure_ref = max(recent_high, float(h.iloc[i]))
            entry_anchor = float(c.iloc[i])

        trigger_points = 0.0
        absorption_only = True
        for ok, name, pts in checks:
            if ok:
                trigger_types.append(name)
                trigger_points += float(pts)
                if name != 'ABSORPTION_STALL':
                    absorption_only = False
        support_points = (1.0 if soft_validation else 0.0) + (0.5 if near_ref else 0.0)
        weighted_points = float(trigger_points + support_points)
        # Weighted trigger model:
        #   Tier 1: failed break / engulfing / outside reversal / micro structure flip.
        #   Tier 2: hammer-pin / rejection close, valid when pressure confirms.
        #   Tier 3: absorption stall is supporting context only, never standalone.
        trigger = bool(exhaustion and not absorption_only and (trigger_points >= 3.0 or weighted_points >= 3.0))
        score = float(np.clip(weighted_points / 5.0, 0.0, 1.0))
        out.update({
            "trigger": bool(trigger),
            "trigger_type": "+".join(trigger_types[:3]) if trigger_types else None,
            "score": float(score),
            "trigger_points": float(trigger_points),
            "weighted_points": float(weighted_points),
            "exhaustion_present": bool(exhaustion),
            "soft_validation": bool(soft_validation),
            "entry_timing": "CLOSE_OR_NEXT_HOLD" if trigger else None,
            "entry_anchor": float(entry_anchor),
            "structure_ref": float(structure_ref),
            "near_ref": bool(near_ref),
        })
        return out
    except Exception:
        return out

def _compute_release_rejection_penalty(
    df: pd.DataFrame,
    *,
    direction: str,
    atr_last: float | None,
    release_level: float | None,
) -> Dict[str, float | bool | None]:
    out: Dict[str, float | bool | None] = {"penalty": 0.0, "stuffing": False, "wick_ratio": 0.0, "close_finish": 0.0}
    try:
        if df is None or len(df) < 3:
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float("nan")
        rel = float(release_level) if release_level is not None and np.isfinite(release_level) else float("nan")
        if not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(rel):
            return out
        recent = df.tail(3).copy()
        o = pd.to_numeric(recent["open"], errors="coerce")
        h = pd.to_numeric(recent["high"], errors="coerce")
        l = pd.to_numeric(recent["low"], errors="coerce")
        c = pd.to_numeric(recent["close"], errors="coerce")
        if min(o.notna().sum(), h.notna().sum(), l.notna().sum(), c.notna().sum()) < 3:
            return out
        direction = str(direction or "").upper().strip()
        ranges = (h - l).replace(0, np.nan)
        if direction == "LONG":
            wick = (h - np.maximum(o, c)) / ranges
            close_finish = (c - l) / ranges
            near_release = bool(float(h.max()) >= rel - 0.18 * atr_val)
            stuffing = bool(near_release and float(wick.fillna(0).mean()) >= 0.34 and float(close_finish.fillna(0.5).mean()) <= 0.62)
        else:
            wick = (np.minimum(o, c) - l) / ranges
            close_finish = (h - c) / ranges
            near_release = bool(float(l.min()) <= rel + 0.18 * atr_val)
            stuffing = bool(near_release and float(wick.fillna(0).mean()) >= 0.34 and float(close_finish.fillna(0.5).mean()) <= 0.62)
        penalty = 0.0
        if near_release and float(wick.fillna(0).mean()) >= 0.26:
            penalty += 0.5
        if stuffing:
            penalty += 0.5
        out.update({
            "penalty": float(min(1.0, penalty)),
            "stuffing": bool(stuffing),
            "wick_ratio": float(wick.fillna(0).mean()),
            "close_finish": float(close_finish.fillna(0.5).mean()),
        })
        return out
    except Exception:
        return out


def _compute_breakout_urgency(
    df: pd.DataFrame,
    *,
    direction: str,
    atr_last: float | None,
    release_level: float | None,
) -> Dict[str, float | bool | None]:
    out: Dict[str, float | bool | None] = {"score": 0.0, "urgent": False}
    try:
        if df is None or len(df) < 4:
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float("nan")
        rel = float(release_level) if release_level is not None and np.isfinite(release_level) else float("nan")
        if not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(rel):
            return out
        direction = str(direction or "").upper().strip()
        recent = df.tail(4).copy()
        c = pd.to_numeric(recent["close"], errors="coerce").dropna()
        h = pd.to_numeric(recent["high"], errors="coerce").dropna()
        l = pd.to_numeric(recent["low"], errors="coerce").dropna()
        macd = pd.to_numeric(df.get("macd_hist", pd.Series(index=df.index, dtype=float)).tail(4), errors="coerce").dropna()
        if min(len(c), len(h), len(l)) < 3:
            return out
        score = 0.0
        if direction == "LONG":
            if len(macd) >= 3 and macd.iloc[-1] > macd.iloc[-2] > macd.iloc[-3]:
                score += 0.5
            recent_span = max(1e-9, float(h.max() - l.min()))
            if float(c.iloc[-2:].mean()) >= float(l.min()) + 0.68 * recent_span:
                score += 0.5
            if float(c.iloc[-1]) >= rel - 0.22 * atr_val:
                score += 0.5
            if float(l.tail(2).min()) >= rel - 0.55 * atr_val:
                score += 0.5
        else:
            if len(macd) >= 3 and macd.iloc[-1] < macd.iloc[-2] < macd.iloc[-3]:
                score += 0.5
            recent_span = max(1e-9, float(h.max() - l.min()))
            if float(c.iloc[-2:].mean()) <= float(h.max()) - 0.68 * recent_span:
                score += 0.5
            if float(c.iloc[-1]) <= rel + 0.22 * atr_val:
                score += 0.5
            if float(h.tail(2).max()) <= rel + 0.55 * atr_val:
                score += 0.5
        score = float(min(2.0, score))
        out.update({"score": score, "urgent": bool(score >= 1.5)})
        return out
    except Exception:
        return out


def _compute_pullback_unlikelihood(
    df: pd.DataFrame,
    *,
    direction: str,
    atr_last: float | None,
    accept_line: float | None,
) -> Dict[str, float | bool | None]:
    out: Dict[str, float | bool | None] = {"score": 0.0, "unlikely": False}
    try:
        if df is None or len(df) < 6:
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float("nan")
        accept = float(accept_line) if accept_line is not None and np.isfinite(accept_line) else float("nan")
        if not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(accept):
            return out
        direction = str(direction or "").upper().strip()
        c6 = pd.to_numeric(df["close"].tail(6), errors="coerce").dropna()
        l4 = pd.to_numeric(df["low"].tail(4), errors="coerce").dropna()
        h4 = pd.to_numeric(df["high"].tail(4), errors="coerce").dropna()
        if min(len(c6), len(l4), len(h4)) < 4:
            return out
        score = 0.0
        if direction == "LONG":
            retrace_depth = float(c6.max() - l4.min()) / max(1e-9, atr_val)
            if retrace_depth <= 0.65:
                score += 0.75
            if float(l4.min()) >= accept - 0.18 * atr_val:
                score += 0.75
            if float(c6.tail(3).mean()) >= float(c6.max()) - 0.28 * atr_val:
                score += 0.5
        else:
            retrace_depth = float(h4.max() - c6.min()) / max(1e-9, atr_val)
            if retrace_depth <= 0.65:
                score += 0.75
            if float(h4.max()) <= accept + 0.18 * atr_val:
                score += 0.75
            if float(c6.tail(3).mean()) <= float(c6.min()) + 0.28 * atr_val:
                score += 0.5
        score = float(min(2.0, score))
        out.update({"score": score, "unlikely": bool(score >= 1.25)})
        return out
    except Exception:
        return out

def _compute_breakout_acceptance_quality(
    df: pd.DataFrame,
    *,
    direction: str,
    breakout_ref: float | None,
    atr_last: float | None,
    buffer: float = 0.0,
) -> Dict[str, float | bool | None]:
    out: Dict[str, float | bool | None] = {
        "accepted": False,
        "clean_accept": False,
        "rejection": False,
        "touch": False,
        "wick_ratio": 0.0,
        "close_finish": 0.5,
        "last_close_vs_ref": 0.0,
    }
    try:
        if df is None or len(df) < 2:
            return out
        direction = str(direction or "").upper().strip()
        ref = float(breakout_ref) if breakout_ref is not None and np.isfinite(breakout_ref) else float("nan")
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float("nan")
        if not np.isfinite(ref) or not np.isfinite(atr_val) or atr_val <= 0:
            return out
        recent = df.tail(2).copy()
        o = pd.to_numeric(recent["open"], errors="coerce")
        h = pd.to_numeric(recent["high"], errors="coerce")
        l = pd.to_numeric(recent["low"], errors="coerce")
        c = pd.to_numeric(recent["close"], errors="coerce")
        if min(o.notna().sum(), h.notna().sum(), l.notna().sum(), c.notna().sum()) < 2:
            return out
        ranges = (h - l).replace(0, np.nan)
        buf = float(buffer or 0.0)
        if direction == "LONG":
            wick = (h - np.maximum(o, c)) / ranges
            close_finish = (c - l) / ranges
            touch = bool((h >= ref - buf).any())
            last_close_ok = bool(float(c.iloc[-1]) >= ref - max(buf, 0.03 * atr_val))
            avg_close_ok = bool(float(c.mean()) >= ref - 0.01 * atr_val)
            clean_accept = bool(float(c.iloc[-1]) >= ref + 0.02 * atr_val and float(close_finish.iloc[-1]) >= 0.58)
            rejection = bool(
                touch and (
                    (float(wick.fillna(0).iloc[-1]) >= 0.33 and float(c.iloc[-1]) < ref + 0.02 * atr_val)
                    or (float(h.iloc[-1]) >= ref + 0.08 * atr_val and float(c.iloc[-1]) <= ref - 0.02 * atr_val)
                    or (float(wick.fillna(0).mean()) >= 0.30 and float(close_finish.fillna(0.5).mean()) <= 0.58 and float(c.mean()) < ref + 0.01 * atr_val)
                )
            )
            accepted = bool(touch and last_close_ok and avg_close_ok and not rejection)
            last_close_vs_ref = float((float(c.iloc[-1]) - ref) / atr_val)
        else:
            wick = (np.minimum(o, c) - l) / ranges
            close_finish = (h - c) / ranges
            touch = bool((l <= ref + buf).any())
            last_close_ok = bool(float(c.iloc[-1]) <= ref + max(buf, 0.03 * atr_val))
            avg_close_ok = bool(float(c.mean()) <= ref + 0.01 * atr_val)
            clean_accept = bool(float(c.iloc[-1]) <= ref - 0.02 * atr_val and float(close_finish.iloc[-1]) >= 0.58)
            rejection = bool(
                touch and (
                    (float(wick.fillna(0).iloc[-1]) >= 0.33 and float(c.iloc[-1]) > ref - 0.02 * atr_val)
                    or (float(l.iloc[-1]) <= ref - 0.08 * atr_val and float(c.iloc[-1]) >= ref + 0.02 * atr_val)
                    or (float(wick.fillna(0).mean()) >= 0.30 and float(close_finish.fillna(0.5).mean()) <= 0.58 and float(c.mean()) > ref - 0.01 * atr_val)
                )
            )
            accepted = bool(touch and last_close_ok and avg_close_ok and not rejection)
            last_close_vs_ref = float((ref - float(c.iloc[-1])) / atr_val)
        out.update({
            "accepted": bool(accepted),
            "clean_accept": bool(clean_accept and accepted),
            "rejection": bool(rejection),
            "touch": bool(touch),
            "wick_ratio": float(wick.fillna(0).mean()),
            "close_finish": float(close_finish.fillna(0.5).mean()),
            "last_close_vs_ref": float(last_close_vs_ref),
        })
        return out
    except Exception:
        return out


def _compute_breakout_extension_state(
    df: pd.DataFrame,
    *,
    direction: str,
    atr_last: float | None,
    accept_line: float | None,
    ref_vwap: float | None,
) -> Dict[str, float | bool | None]:
    out: Dict[str, float | bool | None] = {
        "penalty": 0.0,
        "extended": False,
        "exhausted": False,
        "dist_accept_atr": 0.0,
        "dist_vwap_atr": 0.0,
        "momentum_fade": False,
        "stalling": False,
    }
    try:
        if df is None or len(df) < 6:
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float("nan")
        accept = float(accept_line) if accept_line is not None and np.isfinite(accept_line) else float("nan")
        vwap = float(ref_vwap) if ref_vwap is not None and np.isfinite(ref_vwap) else float("nan")
        if not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(accept):
            return out
        direction = str(direction or "").upper().strip()
        recent = df.tail(6).copy()
        c = pd.to_numeric(recent["close"], errors="coerce").dropna()
        h = pd.to_numeric(recent["high"], errors="coerce").dropna()
        l = pd.to_numeric(recent["low"], errors="coerce").dropna()
        macd = pd.to_numeric(df.get("macd_hist", pd.Series(index=df.index, dtype=float)).tail(5), errors="coerce").dropna()
        if min(len(c), len(h), len(l)) < 5:
            return out

        if direction == "LONG":
            last_px = float(c.iloc[-1])
            dist_accept_atr = max(0.0, (last_px - accept) / atr_val)
            dist_vwap_atr = max(0.0, (last_px - vwap) / atr_val) if np.isfinite(vwap) else max(0.0, dist_accept_atr)
            recent_high = float(h.max())
            prev_high = float(h.iloc[:-2].max()) if len(h) > 2 else recent_high
            close_slip = bool(last_px <= recent_high - 0.35 * atr_val)
            stall = bool((recent_high - prev_high) <= 0.12 * atr_val and close_slip)
            fade = bool(len(macd) >= 3 and (float(macd.iloc[-1]) < float(macd.iloc[-2]) <= float(macd.iloc[-3])))
            extended = bool(dist_accept_atr >= 0.80 or dist_vwap_atr >= 1.20)
            exhausted = bool(extended and (stall or fade) and dist_accept_atr >= 0.70)
        else:
            last_px = float(c.iloc[-1])
            dist_accept_atr = max(0.0, (accept - last_px) / atr_val)
            dist_vwap_atr = max(0.0, (vwap - last_px) / atr_val) if np.isfinite(vwap) else max(0.0, dist_accept_atr)
            recent_low = float(l.min())
            prev_low = float(l.iloc[:-2].min()) if len(l) > 2 else recent_low
            close_slip = bool(last_px >= recent_low + 0.35 * atr_val)
            stall = bool((prev_low - recent_low) <= 0.12 * atr_val and close_slip)
            fade = bool(len(macd) >= 3 and (float(macd.iloc[-1]) > float(macd.iloc[-2]) >= float(macd.iloc[-3])))
            extended = bool(dist_accept_atr >= 0.80 or dist_vwap_atr >= 1.20)
            exhausted = bool(extended and (stall or fade) and dist_accept_atr >= 0.70)

        penalty = 0.0
        if extended:
            penalty += 0.5
        if max(dist_accept_atr, dist_vwap_atr) >= 1.55:
            penalty += 0.5
        if stall:
            penalty += 0.5
        if fade:
            penalty += 0.5
        if exhausted:
            penalty += 0.25
        out.update({
            "penalty": float(min(1.5, penalty)),
            "extended": bool(extended),
            "exhausted": bool(exhausted),
            "dist_accept_atr": float(dist_accept_atr),
            "dist_vwap_atr": float(dist_vwap_atr),
            "momentum_fade": bool(fade),
            "stalling": bool(stall),
        })
        return out
    except Exception:
        return out




def _anchor_recent_interaction_score(
    df: pd.DataFrame,
    *,
    direction: str,
    anchor: float | None,
    atr_last: float | None,
    lookback: int = 8,
) -> Dict[str, float | int]:
    out: Dict[str, float | int] = {"score": 0.0, "touches": 0, "defended": 0, "wick_quality": 0.0, "close_quality": 0.0}
    try:
        if df is None or len(df) < 3:
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float("nan")
        anch = float(anchor) if anchor is not None and np.isfinite(anchor) else float("nan")
        if not np.isfinite(anch) or not np.isfinite(atr_val) or atr_val <= 0:
            return out
        direction = str(direction or '').upper().strip()
        recent = df.tail(int(max(3, min(lookback, len(df))))).copy()
        h = pd.to_numeric(recent['high'], errors='coerce')
        l = pd.to_numeric(recent['low'], errors='coerce')
        c = pd.to_numeric(recent['close'], errors='coerce')
        o = pd.to_numeric(recent['open'], errors='coerce')
        band = max(0.08 * atr_val, 1e-9)
        touches = 0
        defended = 0
        wick_scores = []
        close_scores = []
        for hh, ll, cc, oo in zip(h.tolist(), l.tolist(), c.tolist(), o.tolist()):
            if not all(np.isfinite(v) for v in [hh, ll, cc, oo]):
                continue
            touched = bool((ll <= anch + band) and (hh >= anch - band))
            if not touched:
                continue
            touches += 1
            rng = max(1e-9, hh - ll)
            if direction == 'LONG':
                wick_pen = max(0.0, anch - ll) / max(1e-9, 0.28 * atr_val)
                wick_q = float(np.clip(1.0 - wick_pen, 0.0, 1.0))
                close_q = float(np.clip((cc - anch) / max(1e-9, 0.30 * atr_val), -1.0, 1.0))
                defended_bar = bool(cc >= anch - 0.06 * atr_val and cc >= oo - 0.10 * rng)
            else:
                wick_pen = max(0.0, hh - anch) / max(1e-9, 0.28 * atr_val)
                wick_q = float(np.clip(1.0 - wick_pen, 0.0, 1.0))
                close_q = float(np.clip((anch - cc) / max(1e-9, 0.30 * atr_val), -1.0, 1.0))
                defended_bar = bool(cc <= anch + 0.06 * atr_val and cc <= oo + 0.10 * rng)
            wick_scores.append(wick_q)
            close_scores.append(max(0.0, close_q))
            if defended_bar:
                defended += 1
        if touches <= 0:
            return out
        wick_quality = float(np.mean(wick_scores)) if wick_scores else 0.0
        close_quality = float(np.mean(close_scores)) if close_scores else 0.0
        score = float(min(1.0, 0.28 * min(touches, 3) + 0.32 * min(defended, 3) / 3.0 + 0.20 * wick_quality + 0.20 * close_quality))
        out.update({
            'score': score,
            'touches': int(touches),
            'defended': int(defended),
            'wick_quality': float(wick_quality),
            'close_quality': float(close_quality),
        })
        return out
    except Exception:
        return out


def _compute_multibar_extension_profile(
    df: pd.DataFrame,
    *,
    direction: str,
    atr_last: float | None,
    accept_line: float | None,
) -> Dict[str, float | bool]:
    out: Dict[str, float | bool] = {"penalty": 0.0, "extended": False, "stalling": False, "fading": False, "path_stretched": False}
    try:
        if df is None or len(df) < 5:
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float('nan')
        accept = float(accept_line) if accept_line is not None and np.isfinite(accept_line) else float('nan')
        if not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(accept):
            return out
        direction = str(direction or '').upper().strip()
        recent = df.tail(5).copy()
        c = pd.to_numeric(recent['close'], errors='coerce').dropna()
        o = pd.to_numeric(recent['open'], errors='coerce').dropna()
        h = pd.to_numeric(recent['high'], errors='coerce').dropna()
        l = pd.to_numeric(recent['low'], errors='coerce').dropna()
        v = pd.to_numeric(recent.get('volume', pd.Series(index=recent.index, dtype=float)), errors='coerce').dropna()
        if min(len(c), len(o), len(h), len(l)) < 4:
            return out
        if direction == 'LONG':
            dists = np.maximum(0.0, (c.astype(float).values - accept) / atr_val)
            progresses = np.diff(c.astype(float).values)
            body_sign = (c.astype(float).values - o.astype(float).values)
            wick_reject = ((h.astype(float).values - np.maximum(c.astype(float).values, o.astype(float).values)) / np.maximum(1e-9, (h.astype(float).values - l.astype(float).values)))
        else:
            dists = np.maximum(0.0, (accept - c.astype(float).values) / atr_val)
            progresses = np.diff(-c.astype(float).values)
            body_sign = (o.astype(float).values - c.astype(float).values)
            wick_reject = ((np.minimum(c.astype(float).values, o.astype(float).values) - l.astype(float).values) / np.maximum(1e-9, (h.astype(float).values - l.astype(float).values)))
        avg_dist = float(np.mean(dists[-3:])) if len(dists) >= 3 else float(np.mean(dists))
        path_stretched = bool(avg_dist >= 0.82 or (len(dists) >= 2 and float(np.max(dists[-2:])) >= 1.05))
        prog_tail = progresses[-3:] if len(progresses) >= 3 else progresses
        stalling = bool(len(prog_tail) >= 2 and np.isfinite(prog_tail).all() and float(np.mean(prog_tail)) <= 0.06 * atr_val)
        fading = bool(len(body_sign) >= 3 and float(np.mean(body_sign[-2:])) <= float(np.mean(body_sign[:2])) * 0.70)
        if len(v) >= 4:
            vol_tail = v.astype(float).values
            fading = bool(fading or (np.mean(vol_tail[-2:]) <= 0.82 * np.mean(vol_tail[:2]) and avg_dist >= 0.70))
        rejection_rising = bool(len(wick_reject) >= 3 and float(np.mean(wick_reject[-2:])) >= max(0.38, float(np.mean(wick_reject[:2])) + 0.08))
        penalty = 0.0
        if path_stretched:
            penalty += 0.35
        if stalling:
            penalty += 0.30
        if fading:
            penalty += 0.25
        if rejection_rising:
            penalty += 0.25
        out.update({
            'penalty': float(min(1.2, penalty)),
            'extended': bool(avg_dist >= 0.70),
            'stalling': bool(stalling or rejection_rising),
            'fading': bool(fading),
            'path_stretched': bool(path_stretched),
        })
        return out
    except Exception:
        return out


def _assess_scalp_weak_tape_turn(
    df: pd.DataFrame,
    *,
    direction: str,
    trigger_line: float | None,
    atr_last: float | None,
) -> Dict[str, float | bool]:
    out: Dict[str, float | bool] = {"score": 0.0, "ok": False, "stall": False, "rejection": False}
    try:
        if df is None or len(df) < 4:
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float('nan')
        trigger = float(trigger_line) if trigger_line is not None and np.isfinite(trigger_line) else float('nan')
        if not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(trigger):
            return out
        direction = str(direction or '').upper().strip()
        recent = df.tail(4).copy()
        c = pd.to_numeric(recent['close'], errors='coerce').dropna()
        o = pd.to_numeric(recent['open'], errors='coerce').dropna()
        h = pd.to_numeric(recent['high'], errors='coerce').dropna()
        l = pd.to_numeric(recent['low'], errors='coerce').dropna()
        if min(len(c), len(o), len(h), len(l)) < 4:
            return out
        ranges = np.maximum(1e-9, (h.values - l.values))
        if direction == 'LONG':
            lower_wicks = (np.minimum(c.values, o.values) - l.values) / ranges
            close_progress = np.diff(c.values)
            stall = bool(np.mean(close_progress[-2:]) >= -0.03 * atr_val)
            reclaim_fail = int(np.sum((h.values >= trigger - 0.03 * atr_val) & (c.values <= trigger - 0.04 * atr_val)))
            rejection = bool(reclaim_fail >= 2)
            score = float(0.50 * stall + 0.30 * (np.mean(lower_wicks[-2:]) >= 0.28) + 0.20 * (c.values[-1] >= c.values[-2] - 0.02 * atr_val))
        else:
            upper_wicks = (h.values - np.maximum(c.values, o.values)) / ranges
            close_progress = np.diff(-c.values)
            stall = bool(np.mean(close_progress[-2:]) >= -0.03 * atr_val)
            reclaim_fail = int(np.sum((l.values <= trigger + 0.03 * atr_val) & (c.values >= trigger + 0.04 * atr_val)))
            rejection = bool(reclaim_fail >= 2)
            score = float(0.50 * stall + 0.30 * (np.mean(upper_wicks[-2:]) >= 0.28) + 0.20 * (c.values[-1] <= c.values[-2] + 0.02 * atr_val))
        out.update({'score': float(np.clip(score, 0.0, 1.0)), 'ok': bool(score >= 0.55 and not rejection), 'stall': bool(stall), 'rejection': bool(rejection)})
        return out
    except Exception:
        return out


def _classify_ride_structure_phase_info(
    *,
    direction: str,
    df: pd.DataFrame,
    accept_line: float | None,
    break_trigger: float | None,
    atr_last: float | None,
) -> dict[str, object]:
    """RIDE structure phase classification with confidence and operator-facing detail.

    Trader-edge refinement: detect phase emergence earlier while preserving the
    route_phase contract used downstream by RIDE auto-exec/payload logic.
    """
    out: dict[str, object] = {
        'route_phase': 'UNSET',
        'detail_phase': 'UNSET',
        'confidence': 0.0,
        'interpretation': 'No clean continuation phase established',
    }
    try:
        if df is None or len(df) < 5:
            return out
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) else float('nan')
        accept = float(accept_line) if accept_line is not None and np.isfinite(accept_line) else float('nan')
        br = float(break_trigger) if break_trigger is not None and np.isfinite(break_trigger) else float('nan')
        if not np.isfinite(atr_val) or atr_val <= 0 or not np.isfinite(accept) or not np.isfinite(br):
            return out
        direction = str(direction or '').upper().strip()
        recent = df.tail(5).copy()
        short_ctx = df.tail(min(len(df), 12)).copy()
        struct_ctx = df.tail(min(len(df), 24)).copy()
        o = pd.to_numeric(recent['open'], errors='coerce').dropna()
        c = pd.to_numeric(recent['close'], errors='coerce').dropna()
        l = pd.to_numeric(recent['low'], errors='coerce').dropna()
        h = pd.to_numeric(recent['high'], errors='coerce').dropna()
        v_short = pd.to_numeric(short_ctx.get('volume', pd.Series(index=short_ctx.index, data=np.nan)), errors='coerce')
        c_short = pd.to_numeric(short_ctx['close'], errors='coerce').dropna()
        l_short = pd.to_numeric(short_ctx['low'], errors='coerce').dropna()
        h_short = pd.to_numeric(short_ctx['high'], errors='coerce').dropna()
        l_struct = pd.to_numeric(struct_ctx['low'], errors='coerce').dropna()
        h_struct = pd.to_numeric(struct_ctx['high'], errors='coerce').dropna()
        if min(len(o), len(c), len(l), len(h)) < 5:
            return out

        ranges = np.maximum(1e-9, (h.values - l.values))
        body_frac = np.abs(c.values - o.values) / ranges
        last_close = float(c.iloc[-1])
        prev_close = float(c.iloc[-2]) if len(c) >= 2 else last_close
        range_atr = float((float(h.max()) - float(l.min())) / atr_val)
        short_range_atr = float((float(h_short.max()) - float(l_short.min())) / atr_val) if min(len(h_short), len(l_short)) >= 5 else range_atr
        squeeze_ratio = float(np.clip(range_atr / max(1e-9, short_range_atr), 0.0, 3.0))
        body_q = float(np.clip(np.mean(body_frac[-3:]), 0.0, 1.0))
        vol_tail = v_short.dropna().tail(min(8, len(v_short.dropna())))
        if len(vol_tail) >= 5:
            vol_recent = float(np.nanmean(vol_tail.tail(3)))
            vol_prior = float(np.nanmean(vol_tail.head(max(1, len(vol_tail) - 3))))
            volume_drying = bool(vol_recent <= vol_prior * 1.05)
            volume_dead = bool(vol_recent <= max(1.0, vol_prior) * 0.35)
        else:
            volume_drying = False
            volume_dead = False

        if direction == 'LONG':
            upper_wicks = (h.values - np.maximum(c.values, o.values)) / ranges
            lower_wicks = (np.minimum(c.values, o.values) - l.values) / ranges
            defended = bool(float(l.tail(3).min()) >= accept - 0.16 * atr_val)
            near_accept_defense = bool(float(l.tail(3).min()) <= accept + 0.22 * atr_val and last_close >= accept - 0.06 * atr_val)
            compressed = bool(range_atr <= 1.10 and last_close <= br + 0.20 * atr_val)
            dist_break = float((last_close - br) / atr_val)
            dist_accept = float((last_close - accept) / atr_val)
            close_progress = np.diff(c.values)
            close_stack = float(np.clip(np.mean(close_progress[-3:] >= -0.02 * atr_val), 0.0, 1.0))
            reject = float(np.clip(np.mean(upper_wicks[-3:]), 0.0, 1.0))
            support_wick = float(np.clip(np.mean(lower_wicks[-3:]), 0.0, 1.0))
            broke_recent = bool(np.any(c.tail(3).values >= br + 0.02 * atr_val))
            currently_beyond_break = bool(last_close >= br + 0.02 * atr_val)
            failed_back_through = bool(broke_recent and last_close < br - 0.08 * atr_val)
            failed = bool(dist_break >= 0.45 and (last_close <= float(c.max()) - 0.30 * atr_val or reject >= 0.48))
            struct_extension = float((last_close - float(l_struct.min())) / atr_val) if len(l_struct) >= 5 else dist_accept
            giveback = float((float(h.tail(5).max()) - last_close) / atr_val)
            trend_vals = l_short.tail(min(6, len(l_short))).values
            trend_step = float(np.clip(np.mean(np.diff(trend_vals) >= -0.03 * atr_val), 0.0, 1.0)) if len(trend_vals) >= 4 else close_stack
            micro_vals = l_short.tail(min(5, len(l_short))).values
            rising_structure = float(np.clip(np.mean(np.diff(micro_vals) >= -0.04 * atr_val), 0.0, 1.0)) if len(micro_vals) >= 4 else trend_step
            upper_half_hold = float(np.clip(np.mean(c.tail(4).values >= (l.tail(4).values + 0.52 * np.maximum(1e-9, (h.tail(4).values - l.tail(4).values)))), 0.0, 1.0)) if len(c) >= 4 else close_stack
            pressure_build = bool((rising_structure >= 0.66 or trend_step >= 0.66) and upper_half_hold >= 0.50)
            distance_shrinking = bool(len(c_short) >= 4 and abs(float(c_short.iloc[-1]) - br) <= abs(float(c_short.iloc[-4]) - br) + 0.04 * atr_val)
            stalling_extension = bool(dist_break >= 0.22 and (giveback >= 0.18 or reject >= 0.34 or (last_close <= prev_close + 0.01 * atr_val and body_q <= 0.44)) and not failed)
        else:
            lower_wicks = (np.minimum(c.values, o.values) - l.values) / ranges
            upper_wicks = (h.values - np.maximum(c.values, o.values)) / ranges
            defended = bool(float(h.tail(3).max()) <= accept + 0.16 * atr_val)
            near_accept_defense = bool(float(h.tail(3).max()) >= accept - 0.22 * atr_val and last_close <= accept + 0.06 * atr_val)
            compressed = bool(range_atr <= 1.10 and last_close >= br - 0.20 * atr_val)
            dist_break = float((br - last_close) / atr_val)
            dist_accept = float((accept - last_close) / atr_val)
            close_progress = np.diff(-c.values)
            close_stack = float(np.clip(np.mean(close_progress[-3:] >= -0.02 * atr_val), 0.0, 1.0))
            reject = float(np.clip(np.mean(lower_wicks[-3:]), 0.0, 1.0))
            support_wick = float(np.clip(np.mean(upper_wicks[-3:]), 0.0, 1.0))
            broke_recent = bool(np.any(c.tail(3).values <= br - 0.02 * atr_val))
            currently_beyond_break = bool(last_close <= br - 0.02 * atr_val)
            failed_back_through = bool(broke_recent and last_close > br + 0.08 * atr_val)
            failed = bool(dist_break >= 0.45 and (last_close >= float(c.min()) + 0.30 * atr_val or reject >= 0.48))
            struct_extension = float((float(h_struct.max()) - last_close) / atr_val) if len(h_struct) >= 5 else dist_accept
            giveback = float((last_close - float(l.tail(5).min())) / atr_val)
            trend_vals = h_short.tail(min(6, len(h_short))).values
            trend_step = float(np.clip(np.mean(np.diff(trend_vals) <= 0.03 * atr_val), 0.0, 1.0)) if len(trend_vals) >= 4 else close_stack
            micro_vals = h_short.tail(min(5, len(h_short))).values
            rising_structure = float(np.clip(np.mean(np.diff(micro_vals) <= 0.04 * atr_val), 0.0, 1.0)) if len(micro_vals) >= 4 else trend_step
            upper_half_hold = float(np.clip(np.mean(c.tail(4).values <= (h.tail(4).values - 0.52 * np.maximum(1e-9, (h.tail(4).values - l.tail(4).values)))), 0.0, 1.0)) if len(c) >= 4 else close_stack
            pressure_build = bool((rising_structure >= 0.66 or trend_step >= 0.66) and upper_half_hold >= 0.50)
            distance_shrinking = bool(len(c_short) >= 4 and abs(float(c_short.iloc[-1]) - br) <= abs(float(c_short.iloc[-4]) - br) + 0.04 * atr_val)
            stalling_extension = bool(dist_break >= 0.22 and (giveback >= 0.18 or reject >= 0.34 or (last_close >= prev_close - 0.01 * atr_val and body_q <= 0.44)) and not failed)

        compression_build = bool(defended and (compressed or squeeze_ratio <= 0.74) and 0.08 <= dist_break <= 0.72 and close_stack >= 0.52 and reject <= 0.32 and struct_extension <= 1.45 and trend_step >= 0.50)
        pre_compression = bool((not volume_dead) and (compressed or squeeze_ratio <= 0.86 or short_range_atr <= 1.34) and -0.35 <= dist_break <= 0.82 and pressure_build and distance_shrinking and reject <= 0.38 and struct_extension <= 1.42 and trend_step >= 0.48)
        acceptance_attempt = bool(near_accept_defense and dist_accept >= -0.12 and dist_break <= 0.52 and close_stack >= 0.45 and support_wick >= 0.18 and reject <= 0.40 and not failed_back_through)
        re_compression = bool(defended and 0.18 <= dist_break <= 0.82 and (squeeze_ratio <= 0.78 or range_atr <= 0.95) and close_stack >= 0.48 and trend_step >= 0.52 and reject <= 0.36 and 0.85 <= struct_extension <= 1.65)
        true_break_hold = bool(broke_recent and currently_beyond_break and defended and not failed_back_through and close_stack >= 0.52 and reject <= 0.36 and -0.03 <= dist_break <= 0.72)
        true_extension = bool(dist_break >= 0.62 and struct_extension >= 1.20 and (range_atr >= 1.12 or squeeze_ratio >= 0.88 or reject >= 0.22 or body_q <= 0.46 or giveback >= 0.14 or stalling_extension))
        try:
            prior_range_expanded = bool(short_range_atr >= 1.10 or range_atr >= 0.95)
            recovery_build = bool(
                prior_range_expanded and (not volume_dead)
                and near_accept_defense and dist_accept >= -0.18 and dist_break <= 0.95
                and close_stack >= 0.48 and support_wick >= 0.16 and reject <= 0.42
                and trend_step >= 0.46 and rising_structure >= 0.50
                and struct_extension <= 1.75 and not failed_back_through
            )
        except Exception:
            recovery_build = False

        if failed:
            detail_phase = 'FAILED_EXTENSION'; conf = float(np.clip(0.55 + 0.25 * reject + 0.20 * min(1.0, abs(dist_break)), 0.0, 0.98)); interp = 'Continuation weakened after extension and lost control'
        elif stalling_extension and true_extension:
            detail_phase = 'STALLING_EXTENSION'; conf = float(np.clip(0.50 + 0.16 * min(1.0, giveback) + 0.14 * reject + 0.10 * max(0.0, 0.48 - body_q), 0.0, 0.95)); interp = 'Move is extended and beginning to stall; avoid new breakout chase'
        elif true_extension:
            detail_phase = 'EXTEND_THEN_PULLBACK'; conf = float(np.clip(0.52 + 0.14 * min(1.2, dist_break) + 0.10 * min(1.0, struct_extension / 1.8) + 0.10 * max(0.0, reject - 0.18) + 0.08 * max(0.0, 0.52 - body_q), 0.0, 0.96)); interp = 'Trend remains healthy but the move is stretched enough to favor a reset before continuation'
        elif pre_compression and not broke_recent:
            detail_phase = 'PRE_COMPRESSION'; conf = float(np.clip(0.46 + 0.14 * rising_structure + 0.12 * upper_half_hold + 0.10 * max(0.0, 0.92 - squeeze_ratio) + 0.08 * close_stack + 0.06 * float(volume_drying), 0.0, 0.92)); interp = 'Pressure is starting to coil before the breakout becomes obvious'
        elif compression_build:
            detail_phase = 'COMPRESSION_BUILDUP'; conf = float(np.clip(0.52 + 0.16 * close_stack + 0.10 * trend_step + 0.10 * max(0.0, 0.90 - squeeze_ratio) + 0.08 * max(0.0, 0.34 - reject), 0.0, 0.96)); interp = 'Pressure is building above support and may resolve through continuation breakout'
        elif acceptance_attempt and not broke_recent:
            detail_phase = 'ACCEPTANCE_ATTEMPT'; conf = float(np.clip(0.48 + 0.16 * close_stack + 0.12 * support_wick + 0.10 * trend_step + 0.08 * max(0.0, 0.42 - reject), 0.0, 0.93)); interp = 'Key level is being defended before full acceptance confirmation'
        elif recovery_build and not true_extension:
            detail_phase = 'RECOVERY_BUILD'; conf = float(np.clip(0.46 + 0.14 * close_stack + 0.12 * support_wick + 0.10 * trend_step + 0.08 * rising_structure + 0.06 * float(not volume_dead), 0.0, 0.92)); interp = 'Post-flush recovery is stabilizing into continuation pressure'
        elif true_break_hold:
            detail_phase = 'BREAK_AND_HOLD'; conf = float(np.clip(0.58 + 0.20 * close_stack + 0.12 * body_q + 0.10 * (1.10 - min(1.10, range_atr)), 0.0, 0.98)); interp = 'Fresh breakout has printed and is holding structure'
        elif defended and dist_break <= 0.32 and close_stack >= 0.58 and reject <= 0.34:
            detail_phase = 'EARLY_ACCEPTANCE'; conf = float(np.clip(0.54 + 0.22 * close_stack + 0.12 * body_q + 0.10 * (0.34 - reject), 0.0, 0.97)); interp = 'Price is beginning to accept the key level and remains early in the move'
        elif defended and dist_break <= 1.05 and close_stack >= 0.64 and reject <= 0.30 and body_q >= 0.38:
            detail_phase = 'PERSISTENT_CONTINUATION'; conf = float(np.clip(0.58 + 0.18 * close_stack + 0.10 * body_q + 0.08 * trend_step + 0.08 * (0.30 - reject), 0.0, 0.99)); interp = 'Trend is persistent and may not offer a full pullback'
        elif re_compression:
            detail_phase = 'RE_COMPRESSION'; conf = float(np.clip(0.50 + 0.14 * close_stack + 0.12 * trend_step + 0.10 * max(0.0, 0.86 - squeeze_ratio) + 0.08 * float(volume_drying), 0.0, 0.94)); interp = 'Mature acceptance is re-coiling and may produce another continuation attempt'
        elif defended and dist_break <= 0.62 and close_stack >= 0.40 and reject <= 0.42:
            detail_phase = 'MATURE_ACCEPTANCE'; conf = float(np.clip(0.48 + 0.18 * close_stack + 0.10 * body_q + 0.08 * (0.42 - reject), 0.0, 0.92)); interp = 'Price is accepted above the key level but the move is more mature'
        elif pre_compression or acceptance_attempt or recovery_build:
            detail_phase = 'EARLY_BUILD'; conf = float(np.clip(0.38 + 0.12 * float(pressure_build) + 0.10 * close_stack + 0.08 * trend_step + 0.06 * float(not volume_dead), 0.0, 0.84)); interp = 'Continuation structure is beginning but has not reached a tradable phase yet'
        else:
            detail_phase = 'UNSTRUCTURED'; conf = float(np.clip(0.34 + 0.10 * defended + 0.08 * close_stack + 0.06 * trend_step, 0.0, 0.82)); interp = 'No clean continuation phase established'

        route_map = {
            'BREAK_AND_HOLD': 'BREAK_AND_HOLD',
            'EARLY_ACCEPTANCE': 'ACCEPT_AND_GO',
            'ACCEPTANCE_ATTEMPT': 'ACCEPT_AND_GO',
            'PRE_COMPRESSION': 'ACCEPT_AND_GO',
            'COMPRESSION_BUILDUP': 'ACCEPT_AND_GO',
            'RE_COMPRESSION': 'ACCEPT_AND_GO',
            'RECOVERY_BUILD': 'ACCEPT_AND_GO',
            'PERSISTENT_CONTINUATION': 'ACCEPT_AND_GO',
            'MATURE_ACCEPTANCE': 'ACCEPT_AND_GO',
            'EXTEND_THEN_PULLBACK': 'EXTEND_THEN_PULLBACK',
            'STALLING_EXTENSION': 'EXTEND_THEN_PULLBACK',
            'FAILED_EXTENSION': 'FAILED_EXTENSION',
            'EARLY_BUILD': 'UNSET',
            'UNSTRUCTURED': 'UNSET',
        }
        out.update({'route_phase': route_map.get(detail_phase, 'UNSET'), 'detail_phase': detail_phase, 'confidence': float(np.clip(conf, 0.0, 0.99)), 'interpretation': interp})
        return out
    except Exception:
        return out

def _classify_ride_structure_phase(
    *,
    direction: str,
    df: pd.DataFrame,
    accept_line: float | None,
    break_trigger: float | None,
    atr_last: float | None,
) -> str:
    try:
        return str(_classify_ride_structure_phase_info(
            direction=direction,
            df=df,
            accept_line=accept_line,
            break_trigger=break_trigger,
            atr_last=atr_last,
        ).get('route_phase') or 'UNSET')
    except Exception:
        return 'UNSET'




def _classify_macd_momentum_state(
    hist_series: pd.Series | None,
    *,
    atr_last: float | None,
    direction: str,
) -> dict[str, object]:
    """Classify MACD histogram into a direction-aware momentum state.

    This is a momentum overlay for RIDE. It is intentionally not a structural
    phase classifier; it describes how alive the move still is inside the phase.
    """
    out: dict[str, object] = {
        "raw_state": "NEUTRAL_NOISE",
        "aligned_state": "NEUTRAL_NOISE",
        "comment": "Momentum neutral/noisy",
        "aligned": False,
        "norm_slope": 0.0,
        "norm_accel": 0.0,
        "norm_mag": 0.0,
        "consistency": 0.0,
        "recent_peak_ratio": 1.0,
        "crossed_zero": False,
    }
    try:
        if hist_series is None:
            return out
        hs = pd.to_numeric(hist_series, errors='coerce').dropna().tail(6)
        if len(hs) < 4:
            return out
        h0, h1, h2, h3 = float(hs.iloc[-1]), float(hs.iloc[-2]), float(hs.iloc[-3]), float(hs.iloc[-4])
        atr_val = float(atr_last) if atr_last is not None and np.isfinite(atr_last) and float(atr_last) > 0 else 1.0
        slope4 = float(h0 - h3)
        accel = float((h0 - h1) - (h1 - h2))
        norm_slope = float(slope4 / max(1e-9, atr_val))
        norm_accel = float(accel / max(1e-9, atr_val))
        norm_mag = float(abs(h0) / max(1e-9, atr_val))
        T_small = 0.012
        T_strong = 0.026
        T_accel = 0.004
        T_mag = 0.004
        expanding_up = bool(h0 > h1 and h1 >= h2)
        contracting_up = bool(h0 < h1 and h1 <= h2)
        expanding_dn = bool(h0 < h1 and h1 <= h2)
        contracting_dn = bool(h0 > h1 and h1 >= h2)
        up_steps = [float(h0 > h1), float(h1 > h2), float(h2 > h3)]
        dn_steps = [float(h0 < h1), float(h1 < h2), float(h2 < h3)]
        crossed_up = bool(h1 <= 0.0 and h0 > 0.0)
        crossed_dn = bool(h1 >= 0.0 and h0 < 0.0)
        recent_peak = float(np.max(np.abs(hs.values))) if len(hs) else abs(h0)
        recent_peak_ratio = float(abs(h0) / max(1e-9, recent_peak))
        raw_state = 'NEUTRAL_NOISE'
        if abs(h0) <= T_mag and abs(norm_slope) <= T_small:
            raw_state = 'NEUTRAL_NOISE'
        elif h0 > 0:
            if crossed_up and norm_slope > T_small and norm_accel >= -0.5 * T_accel:
                raw_state = 'IGNITING_UP'
            elif norm_slope > T_strong and norm_accel >= -T_accel and expanding_up:
                raw_state = 'ACCELERATING_UP'
            elif norm_slope < -T_small and contracting_up:
                raw_state = 'TAPERING_UP'
            else:
                raw_state = 'SUSTAINING_UP'
            if raw_state == 'SUSTAINING_UP' and recent_peak_ratio < 0.72 and norm_slope < 0.0:
                raw_state = 'TAPERING_UP'
        else:
            if crossed_dn and norm_slope < -T_small and norm_accel <= 0.5 * T_accel:
                raw_state = 'IGNITING_DOWN'
            elif norm_slope < -T_strong and norm_accel <= T_accel and expanding_dn:
                raw_state = 'ACCELERATING_DOWN'
            elif norm_slope > T_small and contracting_dn:
                raw_state = 'TAPERING_DOWN'
            else:
                raw_state = 'SUSTAINING_DOWN'
            if raw_state == 'SUSTAINING_DOWN' and recent_peak_ratio < 0.72 and norm_slope > 0.0:
                raw_state = 'TAPERING_DOWN'

        direction = str(direction or '').upper().strip()
        aligned_state = 'NEUTRAL_NOISE'
        aligned = False
        if direction == 'LONG':
            if raw_state.endswith('_UP'):
                aligned = True
                if raw_state.startswith('IGNITING'):
                    aligned_state = 'IGNITING'
                elif raw_state.startswith('ACCELERATING'):
                    aligned_state = 'ACCELERATING'
                elif raw_state.startswith('SUSTAINING'):
                    aligned_state = 'SUSTAINING'
                elif raw_state.startswith('TAPERING'):
                    aligned_state = 'TAPERING'
            elif raw_state.endswith('_DOWN'):
                aligned_state = 'ROLLING_OVER' if crossed_dn or norm_slope < -T_strong else 'COUNTER_TREND'
        else:
            if raw_state.endswith('_DOWN'):
                aligned = True
                if raw_state.startswith('IGNITING'):
                    aligned_state = 'IGNITING'
                elif raw_state.startswith('ACCELERATING'):
                    aligned_state = 'ACCELERATING'
                elif raw_state.startswith('SUSTAINING'):
                    aligned_state = 'SUSTAINING'
                elif raw_state.startswith('TAPERING'):
                    aligned_state = 'TAPERING'
            elif raw_state.endswith('_UP'):
                aligned_state = 'ROLLING_OVER' if crossed_up or norm_slope > T_strong else 'COUNTER_TREND'
        comments = {
            'IGNITING': 'Momentum ignition building',
            'ACCELERATING': 'Momentum accelerating',
            'SUSTAINING': 'Momentum sustaining',
            'TAPERING': 'Momentum tapering; reset risk rising',
            'ROLLING_OVER': 'Momentum rolling over; reset likely',
            'COUNTER_TREND': 'Counter-trend momentum active',
            'NEUTRAL_NOISE': 'Momentum neutral/noisy',
        }
        consistency = float(max(sum(up_steps), sum(dn_steps)) / 3.0)
        out.update({
            'raw_state': raw_state,
            'aligned_state': aligned_state,
            'comment': comments.get(aligned_state, comments['NEUTRAL_NOISE']),
            'aligned': bool(aligned),
            'norm_slope': norm_slope,
            'norm_accel': norm_accel,
            'norm_mag': norm_mag,
            'consistency': consistency,
            'recent_peak_ratio': recent_peak_ratio,
            'crossed_zero': bool(crossed_up or crossed_dn),
        })
        return out
    except Exception:
        return out


def _ride_macd_phase_utility(
    *,
    direction: str,
    detail_phase: str | None,
    macd_info: dict[str, object] | None,
    entry_mode: str | None,
) -> dict[str, object]:
    """Momentum-state utility layered on top of the structural phase."""
    out: dict[str, object] = {
        'score_adj': 0.0,
        'hard_block': False,
        'soft_caution': False,
        'watch_early_build': False,
        'breakout_bias_bonus': 0.0,
        'comment': '',
    }
    try:
        phase = str(detail_phase or 'UNSET').upper().strip()
        mode = str(entry_mode or '').upper().strip()
        state = str((macd_info or {}).get('aligned_state') or 'NEUTRAL_NOISE').upper().strip()
        comment = str((macd_info or {}).get('comment') or '')
        score_adj = 0.0
        hard_block = False
        soft_caution = False
        watch_early_build = False
        breakout_bias_bonus = 0.0

        if phase == 'BREAK_AND_HOLD':
            if state == 'ACCELERATING':
                score_adj += 5.0; breakout_bias_bonus += 0.12
            elif state == 'SUSTAINING':
                score_adj += 2.0
            elif state == 'TAPERING':
                score_adj -= 5.0; soft_caution = True
            elif state == 'ROLLING_OVER':
                score_adj -= 12.0; hard_block = True
        elif phase in ('EARLY_ACCEPTANCE', 'ACCEPTANCE_ATTEMPT'):
            if state == 'IGNITING':
                score_adj += 4.0; breakout_bias_bonus += 0.12
            elif state == 'ACCELERATING':
                score_adj += 6.0; breakout_bias_bonus += 0.10
            elif state == 'SUSTAINING':
                score_adj += 2.0
            elif state == 'TAPERING':
                score_adj -= 4.0; soft_caution = True
            elif state == 'ROLLING_OVER':
                score_adj -= 10.0; hard_block = True
        elif phase in ('COMPRESSION_BUILDUP', 'PRE_COMPRESSION', 'RE_COMPRESSION', 'RECOVERY_BUILD'):
            if state == 'IGNITING':
                score_adj += 5.0; breakout_bias_bonus += 0.16
            elif state == 'ACCELERATING':
                score_adj += 6.0; breakout_bias_bonus += 0.14
            elif state == 'SUSTAINING':
                score_adj += 2.0; breakout_bias_bonus += 0.05
            elif state == 'TAPERING':
                score_adj -= 3.0; soft_caution = True
            elif state == 'ROLLING_OVER':
                score_adj -= 9.0; hard_block = True
        elif phase == 'PERSISTENT_CONTINUATION':
            if state == 'ACCELERATING':
                score_adj += 5.0; breakout_bias_bonus += 0.10
            elif state == 'SUSTAINING':
                score_adj += 2.0
            elif state == 'TAPERING':
                score_adj -= 4.0; soft_caution = True
            elif state == 'ROLLING_OVER':
                score_adj -= 10.0; hard_block = True
        elif phase == 'MATURE_ACCEPTANCE':
            if state == 'ACCELERATING':
                score_adj += 2.0
            elif state == 'SUSTAINING':
                score_adj += 1.0
            elif state == 'TAPERING':
                score_adj -= 6.0; soft_caution = True
            elif state == 'ROLLING_OVER':
                score_adj -= 12.0; hard_block = True
        elif phase in ('EXTEND_THEN_PULLBACK', 'STALLING_EXTENSION'):
            if state == 'ACCELERATING':
                score_adj += 1.0
            elif state == 'TAPERING':
                score_adj -= 7.0; soft_caution = True
            elif state == 'ROLLING_OVER':
                score_adj -= 14.0; hard_block = True
        elif phase == 'FAILED_EXTENSION':
            if state in ('TAPERING', 'ROLLING_OVER'):
                score_adj -= 14.0; hard_block = True
            elif state == 'ACCELERATING':
                score_adj -= 2.0; soft_caution = True
        elif phase in ('UNSTRUCTURED', 'EARLY_BUILD'):
            if state in ('IGNITING', 'ACCELERATING'):
                watch_early_build = True
                score_adj += 1.0
            elif state in ('TAPERING', 'ROLLING_OVER'):
                score_adj -= 4.0

        if state == 'COUNTER_TREND':
            score_adj -= 8.0
            if phase in ('BREAK_AND_HOLD', 'EARLY_ACCEPTANCE', 'ACCEPTANCE_ATTEMPT', 'PERSISTENT_CONTINUATION', 'COMPRESSION_BUILDUP', 'PRE_COMPRESSION', 'RE_COMPRESSION', 'RECOVERY_BUILD') and mode == 'BREAKOUT':
                hard_block = True
        out.update({
            'score_adj': float(score_adj),
            'hard_block': bool(hard_block),
            'soft_caution': bool(soft_caution),
            'watch_early_build': bool(watch_early_build),
            'breakout_bias_bonus': float(np.clip(breakout_bias_bonus, 0.0, 0.18)),
            'comment': comment,
        })
        return out
    except Exception:
        return out

def _ride_continuation_strength_adjustment(
    structure_phase: str | None,
    entry_mode: str | None,
    tape_breakout_urgency: float | None,
    tape_pullback_unlikelihood: float | None,
) -> float:
    """Score tilt for continuation quality.

    Protection-first by design: weak continuation is punished harder than strong
    continuation is rewarded.
    """
    try:
        phase = str(structure_phase or 'UNSET').upper().strip()
        mode = str(entry_mode or '').upper().strip()
        urg = float(tape_breakout_urgency) if tape_breakout_urgency is not None and np.isfinite(tape_breakout_urgency) else 0.0
        unlk = float(tape_pullback_unlikelihood) if tape_pullback_unlikelihood is not None and np.isfinite(tape_pullback_unlikelihood) else 0.0

        continuation_phases = {'BREAK_AND_HOLD', 'ACCEPT_AND_GO', 'EXTEND_THEN_PULLBACK'}
        breakout_friendly_phases = {'BREAK_AND_HOLD', 'ACCEPT_AND_GO'}
        if phase not in continuation_phases or mode not in {'BREAKOUT', 'PULLBACK'}:
            return 0.0

        adj = 0.0
        # Weak continuation: structure may look okay, but the tape is not showing
        # enough urgency and a true no-pullback continuation is unlikely.
        if urg < 0.90 and unlk < 0.95:
            weak_u = float(np.clip((0.90 - urg) / 0.45, 0.0, 1.0))
            weak_p = float(np.clip((0.95 - unlk) / 0.55, 0.0, 1.0))
            weak_score = 0.55 * weak_u + 0.45 * weak_p
            base_pen = 6.0 + 6.0 * weak_score
            if mode == 'BREAKOUT':
                base_pen += 1.5
            if phase == 'EXTEND_THEN_PULLBACK' and mode == 'BREAKOUT':
                base_pen += 1.0
            adj -= base_pen

        # Strong continuation: reward only when urgency and pullback-unlikelihood
        # are both elevated in a breakout-friendly structural phase.
        elif phase in breakout_friendly_phases and urg >= 1.35 and unlk >= 1.20:
            strong_u = float(np.clip((urg - 1.35) / 0.65, 0.0, 1.0))
            strong_p = float(np.clip((unlk - 1.20) / 0.80, 0.0, 1.0))
            strong_score = 0.60 * strong_u + 0.40 * strong_p
            adj += 4.0 + 4.0 * strong_score
            if mode == 'BREAKOUT':
                adj += 0.5

        return float(adj)

    except Exception:
        return 0.0


def _scalp_phase3_profile_adjustment(
    *,
    adx_ctx: dict[str, object],
    long_cluster: int,
    short_cluster: int,
    long_trend_ok: bool,
    short_trend_ok: bool,
    rsi14: float | None,
    last_price: float | None,
    ref_vwap: float | None,
    atr_last: float | None,
    bullish_candle_quality: float | None,
    bearish_candle_quality: float | None,
) -> dict[str, object]:
    """Internal SCALP refinement for volatile small-cap transition vs. late-spike risk.

    Keeps outward payloads unchanged. Returns modest directional adjustments only.
    """
    try:
        regime = str(adx_ctx.get('regime') or 'unknown')
        dom = str(adx_ctx.get('dominant_side') or '').upper().strip()
        slope = float(adx_ctx.get('adx_slope') or 0.0)
        spread = float(adx_ctx.get('di_spread') or 0.0)
        dom_bars = int(adx_ctx.get('dominance_bars') or 0)
        rsi14_v = float(rsi14) if rsi14 is not None and np.isfinite(rsi14) else float('nan')
        atr_v = float(atr_last) if atr_last is not None and np.isfinite(atr_last) and float(atr_last) > 0 else float('nan')
        last_v = float(last_price) if last_price is not None and np.isfinite(last_price) else float('nan')
        vwap_v = float(ref_vwap) if ref_vwap is not None and np.isfinite(ref_vwap) else float('nan')
        long_ext = float((last_v - vwap_v) / atr_v) if np.isfinite(last_v) and np.isfinite(vwap_v) and np.isfinite(atr_v) else 0.0
        short_ext = float((vwap_v - last_v) / atr_v) if np.isfinite(last_v) and np.isfinite(vwap_v) and np.isfinite(atr_v) else 0.0

        bull_cq = float(bullish_candle_quality) if bullish_candle_quality is not None and np.isfinite(bullish_candle_quality) else 0.5
        bear_cq = float(bearish_candle_quality) if bearish_candle_quality is not None and np.isfinite(bearish_candle_quality) else 0.5

        long_adj = 0.0
        short_adj = 0.0
        long_note = None
        short_note = None

        # Transition/reflex logic: reward only when the opposite trend is aging or losing control.
        if long_cluster >= 3 and dom == 'SHORT':
            if regime in ('mature_trend', 'exhausting') and bull_cq >= 0.52:
                long_adj += 4.0 if bull_cq >= 0.64 else 3.0
                long_note = 'ADX transition favors LONG reversal against tired DI-'
            elif regime in ('coiling', 'emerging') and slope >= -0.35 and bull_cq >= 0.56:
                long_adj += 3.0
                long_note = 'ADX transition favors LONG reversal as DI- weakens'
            elif regime in ('strengthening', 'healthy_trend') and spread >= 6.0 and dom_bars >= 2 and not long_trend_ok:
                long_adj -= 3.0 if bull_cq < 0.50 else 2.0
                long_note = 'Countertrend LONG still fighting healthy DI-'

        if short_cluster >= 3 and dom == 'LONG':
            if regime in ('mature_trend', 'exhausting') and bear_cq >= 0.52:
                short_adj += 4.0 if bear_cq >= 0.64 else 3.0
                short_note = 'ADX transition favors SHORT reversal against tired DI+'
            elif regime in ('coiling', 'emerging') and slope >= -0.35 and bear_cq >= 0.56:
                short_adj += 3.0
                short_note = 'ADX transition favors SHORT reversal as DI+ weakens'
            elif regime in ('strengthening', 'healthy_trend') and spread >= 6.0 and dom_bars >= 2 and not short_trend_ok:
                short_adj -= 3.0 if bear_cq < 0.50 else 2.0
                short_note = 'Countertrend SHORT still fighting healthy DI+'

        # Same-direction ignition: reward fresh authority, not late extension.
        if dom == 'LONG' and regime in ('emerging', 'strengthening') and slope >= 0.45 and spread >= 4.5:
            if long_cluster >= 2 and long_ext <= 1.20 and bull_cq >= 0.52:
                long_adj += 2.5 if bull_cq >= 0.66 else 2.0
                long_note = long_note or 'ADX ignition supports fresh LONG expansion'
        elif dom == 'SHORT' and regime in ('emerging', 'strengthening') and slope >= 0.45 and spread >= 4.5:
            if short_cluster >= 2 and short_ext <= 1.20 and bear_cq >= 0.52:
                short_adj += 2.5 if bear_cq >= 0.66 else 2.0
                short_note = short_note or 'ADX ignition supports fresh SHORT expansion'

        # Late-spike caution: common failure mode in $1-$5 volatile names.
        if dom == 'LONG' and regime in ('healthy_trend', 'mature_trend', 'exhausting') and dom_bars >= 2:
            if long_ext >= 1.35 and slope <= 0.20 and np.isfinite(rsi14_v) and rsi14_v >= 67:
                late_pen = 4.0 if bull_cq < 0.42 else 3.0
                long_adj -= late_pen
                long_note = 'Late LONG extension risk vs VWAP/ATR'
        if dom == 'SHORT' and regime in ('healthy_trend', 'mature_trend', 'exhausting') and dom_bars >= 2:
            if short_ext >= 1.35 and slope <= 0.20 and np.isfinite(rsi14_v) and rsi14_v <= 33:
                late_pen = 4.0 if bear_cq < 0.42 else 3.0
                short_adj -= late_pen
                short_note = 'Late SHORT extension risk vs VWAP/ATR'

        return {
            'long_adj': float(np.clip(long_adj, -4.0, 5.0)),
            'short_adj': float(np.clip(short_adj, -4.0, 5.0)),
            'long_note': long_note,
            'short_note': short_note,
        }
    except Exception:
        return {'long_adj': 0.0, 'short_adj': 0.0, 'long_note': None, 'short_note': None}





def _indicator_pressure_states(df: pd.DataFrame, adx_ctx: dict[str, object] | None = None) -> dict[str, object]:
    """Trader-style pressure-state reader reused by SCALP and RIDE.

    This intentionally uses existing columns (RSI, MACD hist, ADX/DI, volume, OHLC)
    and returns internal state only. It does not change any payload contract.
    """
    out: dict[str, object] = {
        "di_transfer_long": False,
        "di_transfer_short": False,
        "rsi_pressure_long": False,
        "rsi_pressure_short": False,
        "macd_pressure_long": False,
        "macd_pressure_short": False,
        "volume_absorption_long": False,
        "volume_absorption_short": False,
        "volume_dryup_coil": False,
        "directional_expansion_long": False,
        "directional_expansion_short": False,
        "fast_trigger_pressure_long": False,
        "fast_trigger_pressure_short": False,
        "structure_window_long": False,
        "structure_window_short": False,
        "long_pressure_score": 0.0,
        "short_pressure_score": 0.0,
    }
    try:
        if df is None or len(df) < 8:
            return out
        d = df.copy()
        close = pd.to_numeric(d.get("close"), errors="coerce")
        open_ = pd.to_numeric(d.get("open"), errors="coerce")
        high = pd.to_numeric(d.get("high"), errors="coerce")
        low = pd.to_numeric(d.get("low"), errors="coerce")
        vol = pd.to_numeric(d.get("volume"), errors="coerce") if "volume" in d else pd.Series(dtype=float)
        rsi5 = pd.to_numeric(d.get("rsi5"), errors="coerce") if "rsi5" in d else pd.Series(dtype=float)
        rsi14 = pd.to_numeric(d.get("rsi14"), errors="coerce") if "rsi14" in d else pd.Series(dtype=float)
        hist = pd.to_numeric(d.get("macd_hist"), errors="coerce") if "macd_hist" in d else pd.Series(dtype=float)
        pdi = pd.to_numeric(d.get("plus_di14"), errors="coerce") if "plus_di14" in d else pd.Series(dtype=float)
        mdi = pd.to_numeric(d.get("minus_di14"), errors="coerce") if "minus_di14" in d else pd.Series(dtype=float)

        # DI control transfer: the old controlling side is still visible, but its spread is narrowing
        # while the opposite DI starts curling. This is often the first readable reversal pressure shift.
        if len(pdi.dropna()) >= 5 and len(mdi.dropna()) >= 5:
            p = pdi.ffill().tail(5); m = mdi.ffill().tail(5)
            spread_now_long = float(m.iloc[-1] - p.iloc[-1])
            spread_prev_long = float(m.iloc[-4] - p.iloc[-4])
            spread_now_short = float(p.iloc[-1] - m.iloc[-1])
            spread_prev_short = float(p.iloc[-4] - m.iloc[-4])
            p_slope = float(p.iloc[-1] - p.iloc[-4])
            m_slope = float(m.iloc[-1] - m.iloc[-4])
            adx_slope = float((adx_ctx or {}).get("adx_slope") or 0.0)
            out["di_transfer_long"] = bool(spread_now_long > 0 and spread_now_long < spread_prev_long - 1.0 and p_slope > 0.35 and m_slope <= 1.0 and adx_slope <= 1.25)
            out["di_transfer_short"] = bool(spread_now_short > 0 and spread_now_short < spread_prev_short - 1.0 and m_slope > 0.35 and p_slope <= 1.0 and adx_slope <= 1.25)

        # RSI pressure shift: RSI improving before price fully confirms, or diverging vs a marginal new extreme.
        if len(rsi5.dropna()) >= 5 and len(rsi14.dropna()) >= 5 and len(close.dropna()) >= 5:
            r5 = rsi5.ffill().tail(5); r14 = rsi14.ffill().tail(5); c5 = close.ffill().tail(5)
            rsi5_slope = float(r5.iloc[-1] - r5.iloc[-4])
            rsi14_slope = float(r14.iloc[-1] - r14.iloc[-4])
            price_slope = float(c5.iloc[-1] - c5.iloc[-4])
            out["rsi_pressure_long"] = bool((rsi5_slope > 1.2 and rsi14_slope >= -0.4) or (c5.iloc[-1] <= c5.iloc[:-1].min() * 1.003 and r5.iloc[-1] > r5.iloc[:-1].min() + 1.5))
            out["rsi_pressure_short"] = bool((rsi5_slope < -1.2 and rsi14_slope <= 0.4) or (c5.iloc[-1] >= c5.iloc[:-1].max() * 0.997 and r5.iloc[-1] < r5.iloc[:-1].max() - 1.5))

        # MACD pressure shift: histogram improving while still negative for long, weakening while still positive for short.
        if len(hist.dropna()) >= 5:
            h = hist.ffill().tail(5)
            h_slope = float(h.iloc[-1] - h.iloc[-4])
            h_accel = float((h.iloc[-1] - h.iloc[-2]) - (h.iloc[-2] - h.iloc[-3]))
            out["macd_pressure_long"] = bool((h.iloc[-1] < 0 and h_slope > 0 and h.iloc[-1] > h.iloc[-2]) or (h_slope > 0 and h_accel >= -abs(h_slope) * 0.40))
            out["macd_pressure_short"] = bool((h.iloc[-1] > 0 and h_slope < 0 and h.iloc[-1] < h.iloc[-2]) or (h_slope < 0 and h_accel <= abs(h_slope) * 0.40))

        # Volume character: absorption (panic/flush fails to continue) vs clean dry-up coil.
        if len(vol.dropna()) >= 8 and len(close.dropna()) >= 8 and len(high.dropna()) >= 8 and len(low.dropna()) >= 8:
            v = vol.ffill().tail(8); c = close.ffill().tail(8); o = open_.ffill().tail(8); h = high.ffill().tail(8); l = low.ffill().tail(8)
            med = float(v.iloc[:-1].median()) if len(v) > 2 else float(v.median())
            rng = (h - l).replace(0.0, np.nan)
            close_loc = ((c - l) / rng).fillna(0.5)
            body = ((c - o).abs() / rng).fillna(0.0)
            prior_spike = bool(v.iloc[-4:-1].max() >= 1.55 * max(1.0, med))
            last_quieter = bool(v.iloc[-1] <= 1.20 * max(1.0, med))
            out["volume_absorption_long"] = bool(prior_spike and last_quieter and l.iloc[-1] >= l.iloc[-4:-1].min() - 0.12 * float((h-l).tail(8).median()) and close_loc.iloc[-1] >= 0.48)
            out["volume_absorption_short"] = bool(prior_spike and last_quieter and h.iloc[-1] <= h.iloc[-4:-1].max() + 0.12 * float((h-l).tail(8).median()) and close_loc.iloc[-1] <= 0.52)
            range_now = float((h.tail(3).max() - l.tail(3).min()))
            range_prev = float((h.iloc[:5].max() - l.iloc[:5].min()))
            vol_dry = bool(v.tail(3).mean() <= 0.92 * max(1.0, v.iloc[:5].mean()))
            out["volume_dryup_coil"] = bool(range_prev > 0 and range_now <= 0.72 * range_prev and vol_dry)
            out["directional_expansion_long"] = bool(v.iloc[-1] >= 1.18 * max(1.0, med) and c.iloc[-1] > o.iloc[-1] and close_loc.iloc[-1] >= 0.62 and body.iloc[-1] >= 0.38)
            out["directional_expansion_short"] = bool(v.iloc[-1] >= 1.18 * max(1.0, med) and c.iloc[-1] < o.iloc[-1] and close_loc.iloc[-1] <= 0.38 and body.iloc[-1] >= 0.38)

        # Multi-speed 5m trigger read: keep larger context, but let the final trigger
        # be dominated by the most recent 3/5/8-bar pressure shift. This helps SCALP
        # see the turn forming and helps RIDE see ignition before the full breakout print.
        try:
            if len(close.dropna()) >= 8:
                c8 = close.ffill().tail(8); o8 = open_.ffill().tail(8); h8 = high.ffill().tail(8); l8 = low.ffill().tail(8)
                c3 = c8.tail(3); h3 = h8.tail(3); l3 = l8.tail(3)
                if len(c3) >= 3:
                    if len(rsi5.dropna()) >= 3:
                        r3 = rsi5.ffill().tail(3)
                        r_fast_long = bool(float(r3.iloc[-1] - r3.iloc[0]) > 0.7 and float(r3.iloc[-1]) >= float(r3.iloc[-2]) - 0.2)
                        r_fast_short = bool(float(r3.iloc[-1] - r3.iloc[0]) < -0.7 and float(r3.iloc[-1]) <= float(r3.iloc[-2]) + 0.2)
                    else:
                        r_fast_long = r_fast_short = False
                    if len(hist.dropna()) >= 3:
                        mh3 = hist.ffill().tail(3)
                        macd_fast_long = bool(float(mh3.iloc[-1]) > float(mh3.iloc[-2]) >= float(mh3.iloc[0]) - abs(float(mh3.iloc[0])) * 0.10)
                        macd_fast_short = bool(float(mh3.iloc[-1]) < float(mh3.iloc[-2]) <= float(mh3.iloc[0]) + abs(float(mh3.iloc[0])) * 0.10)
                    else:
                        macd_fast_long = macd_fast_short = False
                    rng3 = float(max(1e-9, h3.max() - l3.min()))
                    close_upper = bool(float(c3.iloc[-1] - l3.min()) / rng3 >= 0.55)
                    close_lower = bool(float(h3.max() - c3.iloc[-1]) / rng3 >= 0.55)
                    lows_flat_rising = bool(float(l3.iloc[-1]) >= float(l3.iloc[0]) - 0.18 * rng3)
                    highs_flat_falling = bool(float(h3.iloc[-1]) <= float(h3.iloc[0]) + 0.18 * rng3)
                    out["fast_trigger_pressure_long"] = bool((r_fast_long or macd_fast_long) and close_upper and lows_flat_rising)
                    out["fast_trigger_pressure_short"] = bool((r_fast_short or macd_fast_short) and close_lower and highs_flat_falling)
                if len(c8) >= 8:
                    rng8 = float(max(1e-9, h8.max() - l8.min()))
                    out["structure_window_long"] = bool(float(l8.tail(3).min()) >= float(l8.head(5).min()) - 0.12 * rng8 and float(c8.iloc[-1]) >= float(c8.tail(5).median()))
                    out["structure_window_short"] = bool(float(h8.tail(3).max()) <= float(h8.head(5).max()) + 0.12 * rng8 and float(c8.iloc[-1]) <= float(c8.tail(5).median()))
        except Exception:
            pass

        long_score = (
            1.0 * bool(out["di_transfer_long"]) +
            1.0 * bool(out["rsi_pressure_long"]) +
            1.0 * bool(out["macd_pressure_long"]) +
            1.0 * bool(out["volume_absorption_long"]) +
            0.7 * bool(out["volume_dryup_coil"]) +
            0.8 * bool(out["directional_expansion_long"]) +
            0.8 * bool(out["fast_trigger_pressure_long"]) +
            0.5 * bool(out["structure_window_long"])
        )
        short_score = (
            1.0 * bool(out["di_transfer_short"]) +
            1.0 * bool(out["rsi_pressure_short"]) +
            1.0 * bool(out["macd_pressure_short"]) +
            1.0 * bool(out["volume_absorption_short"]) +
            0.7 * bool(out["volume_dryup_coil"]) +
            0.8 * bool(out["directional_expansion_short"]) +
            0.8 * bool(out["fast_trigger_pressure_short"]) +
            0.5 * bool(out["structure_window_short"])
        )
        out["long_pressure_score"] = float(long_score)
        out["short_pressure_score"] = float(short_score)
        return out
    except Exception:
        return out


def _ride_indicator_pressure_adjustment(
    *,
    direction: str,
    detail_phase: str | None,
    entry_mode: str | None,
    adx_ctx: dict[str, object] | None,
    macd_info: dict[str, object] | None,
    pressure_states: dict[str, object] | None,
) -> tuple[float, str | None]:
    """Phase-aware RIDE pressure adjustment using existing ADX/DI + MACD + volume states."""
    try:
        side = str(direction or "").upper().strip()
        phase = str(detail_phase or "UNSET").upper().strip()
        mode = str(entry_mode or "").upper().strip()
        adx = adx_ctx or {}
        ps = pressure_states or {}
        macd = macd_info or {}
        regime = str(adx.get("regime") or "unknown")
        dom = str(adx.get("dominant_side") or "").upper().strip()
        slope = float(adx.get("adx_slope") or 0.0)
        spread = float(adx.get("di_spread") or 0.0)
        state = str(macd.get("aligned_state") or "NEUTRAL_NOISE").upper().strip()
        aligned = bool(dom == side and side in ("LONG", "SHORT"))
        early_phase = phase in {"PRE_COMPRESSION", "COMPRESSION_BUILDUP", "ACCEPTANCE_ATTEMPT", "RE_COMPRESSION", "RECOVERY_BUILD", "EARLY_BUILD"}
        breakout_phase = phase in {"BREAK_AND_HOLD", "EARLY_ACCEPTANCE", "PERSISTENT_CONTINUATION"}
        late_phase = phase in {"MATURE_ACCEPTANCE", "EXTEND_THEN_PULLBACK", "STALLING_EXTENSION", "FAILED_EXTENSION"}
        side_score = float(ps.get("long_pressure_score") if side == "LONG" else ps.get("short_pressure_score") or 0.0)
        if side == "LONG":
            di_transfer = bool(ps.get("di_transfer_long"))
            directional_expansion = bool(ps.get("directional_expansion_long"))
        else:
            di_transfer = bool(ps.get("di_transfer_short"))
            directional_expansion = bool(ps.get("directional_expansion_short"))
        dry_coil = bool(ps.get("volume_dryup_coil"))
        mod = 0.0
        note = None
        # Early edge: fresh ADX/DI pressure + MACD ignition into a coil is worth acting sooner.
        if early_phase and mode in {"BREAKOUT", "PULLBACK", "MOMENTUM"}:
            fresh_adx = bool((regime in {"coiling", "emerging", "strengthening"} and slope >= 0.10) or (aligned and spread >= 4.0))
            fresh_macd = bool(state in {"IGNITING", "ACCELERATING"} or (state == "SUSTAINING" and dry_coil))
            if fresh_adx and fresh_macd and (dry_coil or directional_expansion or side_score >= 2.0):
                mod += 3.0 if side_score >= 2.5 else 2.0
                note = "Early pressure build: ADX/DI + MACD + volume support anticipation"
            elif di_transfer and state in {"IGNITING", "ACCELERATING"}:
                mod += 1.5
                note = "DI control transfer with MACD ignition"
        # Confirmed continuation: reward only if pressure is still alive.
        if breakout_phase and mode == "BREAKOUT" and aligned and state in {"IGNITING", "ACCELERATING", "SUSTAINING"} and directional_expansion:
            mod += 1.5
            note = note or "Directional expansion confirms breakout pressure"
        # Late edge protection: high/late ADX plus tapering MACD should not be chased.
        if late_phase or mode in {"BREAKOUT", "MOMENTUM"}:
            if state in {"TAPERING", "ROLLING_OVER", "COUNTER_TREND"} and (regime in {"mature_trend", "healthy_trend", "exhausting"} or slope <= -0.25):
                pen = 3.5 if phase in {"STALLING_EXTENSION", "EXTEND_THEN_PULLBACK", "FAILED_EXTENSION"} else 2.0
                mod -= pen
                note = "Late-trend pressure fading: MACD/ADX warns against chase"
            elif regime == "exhausting" and mode in {"BREAKOUT", "MOMENTUM"}:
                mod -= 2.0
                note = "ADX exhaustion weakens breakout/momentum entry"
        mod = float(np.clip(mod, -5.0, 4.0))
        if abs(mod) < 1.0:
            note = None
        return mod, note
    except Exception:
        return 0.0, None

def _ride_phase3_continuation_adjustment(
    *,
    direction: str,
    adx_ctx: dict[str, object],
    structure_phase: str | None,
    entry_mode: str | None,
    legitimacy: float,
    vwap_score: float,
    pivot_score: float,
    orb_score: float,
) -> tuple[float, str | None]:
    """Internal continuation durability refinement for volatile small-cap continuation.

    Keeps contracts stable and only nudges score in places where carry quality meaningfully differs.
    """
    try:
        side = str(direction or '').upper().strip()
        regime = str(adx_ctx.get('regime') or 'unknown')
        dom = str(adx_ctx.get('dominant_side') or '').upper().strip()
        slope = float(adx_ctx.get('adx_slope') or 0.0)
        spread = float(adx_ctx.get('di_spread') or 0.0)
        dom_bars = int(adx_ctx.get('dominance_bars') or 0)
        phase = str(structure_phase or 'UNSET').upper().strip()
        mode = str(entry_mode or '').upper().strip()
        aligned = bool(dom == side and side in ('LONG', 'SHORT'))
        support_count = int(float(vwap_score) >= 0.45) + int(float(pivot_score) >= 0.45) + int(float(orb_score) >= 0.45)
        legit = float(legitimacy) if legitimacy is not None and np.isfinite(legitimacy) else 0.0
        legit_strong = bool(legit >= 0.62)
        legit_elite = bool(legit >= 0.78)
        legit_weak = bool(legit < 0.50)

        mod = 0.0
        note = None

        if mode == 'BREAKOUT':
            if regime in ('dead_chop', 'coiling'):
                base_pen = 4.0 if support_count < 3 else 2.0
                mod -= (base_pen + 1.0) if legit_weak else base_pen
                note = 'Breakout carry weak in low-ADX chop/compression'
            elif aligned and regime == 'emerging' and slope >= 0.50 and support_count >= 2:
                if legit_strong:
                    mod += 3.0 if legit_elite else 2.5
                    note = 'Emerging ADX and impulse legitimacy support breakout carry'
                else:
                    mod += 1.0
                    note = 'Emerging ADX improving, but breakout legitimacy is still building'
            elif aligned and regime in ('strengthening', 'healthy_trend') and support_count >= 2:
                core = 3.75 if phase in ('BREAK_AND_HOLD', 'ACCEPT_AND_GO') else 2.75
                if legit_elite:
                    core += 0.75
                elif not legit_strong:
                    core -= 1.0
                mod += core
                note = 'Healthy continuation structure supports breakout carry'
            elif regime == 'mature_trend':
                if aligned and support_count >= 2 and slope >= -0.15 and phase in ('BREAK_AND_HOLD', 'ACCEPT_AND_GO') and legit >= 0.58:
                    mod += 1.5 if legit_strong else 1.0
                    note = 'Mature trend still orderly enough for breakout carry'
                else:
                    mod -= 2.5 if legit_weak else 2.0
                    note = 'Mature breakout trend may be running out of room'
            elif regime == 'exhausting':
                mod -= 5.0 if (phase != 'EXTEND_THEN_PULLBACK' or legit_weak) else 2.5
                note = 'Exhausting ADX weakens breakout carry'
        elif mode == 'PULLBACK':
            if regime in ('dead_chop', 'coiling'):
                base_pen = 3.0 if support_count < 2 else 1.5
                mod -= (base_pen + 0.5) if legit_weak else base_pen
                note = 'Pullback continuation lacks directional authority'
            elif aligned and regime in ('emerging', 'strengthening', 'healthy_trend'):
                core = 2.5 if phase in ('ACCEPT_AND_GO', 'EXTEND_THEN_PULLBACK', 'BREAK_AND_HOLD') else 1.5
                if legit_elite:
                    core += 0.5
                elif not legit_strong:
                    core -= 0.75
                mod += core
                note = 'Healthy trend context supports pullback continuation'
            elif regime == 'mature_trend':
                if aligned and support_count >= 2 and slope >= -0.35 and legit >= 0.54:
                    mod += 1.25 if legit_strong else 1.0
                    note = 'Mature trend still supports disciplined pullback'
                else:
                    mod -= 2.0 if legit_weak else 1.5
                    note = 'Mature pullback continuation needs more caution'
            elif regime == 'exhausting':
                mod -= 3.5 if (dom_bars >= 2 and legit_weak) else (3.0 if dom_bars >= 2 else 2.0)
                note = 'Exhausting ADX weakens pullback carry'

        if not aligned and regime in ('strengthening', 'healthy_trend') and spread >= 6.0 and dom_bars >= 2:
            mod -= 2.5 if legit_weak else 2.0
            note = 'Continuation is leaning against still-healthy DI control'
        if mode == 'BREAKOUT' and legit < 0.55:
            mod -= 1.0 if legit >= 0.45 else 1.5
        elif mode == 'PULLBACK' and legit < 0.50:
            mod -= 0.5 if legit >= 0.42 else 1.0

        mod = float(np.clip(mod, -5.0, 5.0))
        if abs(mod) < 1.0:
            note = None
        return mod, note
    except Exception:
        return 0.0, None


def _assess_coiled_continuation(
    direction: str,
    df: pd.DataFrame,
    accept_line: float,
    break_trigger: float,
    atr_last: float | None,
) -> dict[str, object]:
    """Identify compressed higher-low / lower-high continuation where a real pullback is unlikely.

    This is meant to catch small-cap stair-step continuations that are not impulsive *yet*
    but are clearly coiling near the break trigger.
    """
    try:
        if atr_last is None or not np.isfinite(atr_last) or float(atr_last) <= 0 or len(df) < 4:
            return {"score": 0.0, "coiled": False, "compression": 0.0, "trend_persist": 0.0, "accept_hold": 0.0, "trigger_prox": 0.0}
        n = int(min(5, len(df)))
        recent = df.tail(n).copy()
        h = pd.to_numeric(recent['high'], errors='coerce').astype(float)
        l = pd.to_numeric(recent['low'], errors='coerce').astype(float)
        c = pd.to_numeric(recent['close'], errors='coerce').astype(float)
        if h.isna().all() or l.isna().all() or c.isna().all():
            return {"score": 0.0, "coiled": False, "compression": 0.0, "trend_persist": 0.0, "accept_hold": 0.0, "trigger_prox": 0.0}
        atr = float(atr_last)
        bar_ranges = (h - l).clip(lower=0.0)
        avg_bar_range_atr = float(bar_ranges.mean() / max(1e-9, atr))
        total_range_atr = float((float(h.max()) - float(l.min())) / max(1e-9, atr))
        compression = float(np.clip(1.0 - max(0.0, avg_bar_range_atr - 0.42) / 0.70, 0.0, 1.0))
        compression = float(np.clip(0.60 * compression + 0.40 * np.clip(1.0 - max(0.0, total_range_atr - 1.05) / 0.95, 0.0, 1.0), 0.0, 1.0))

        if direction == 'LONG':
            hl1 = float(l.iloc[-1] - l.iloc[-2]) if len(l) >= 2 else 0.0
            hl2 = float(l.iloc[-2] - l.iloc[-3]) if len(l) >= 3 else 0.0
            trend_persist = 0.0
            if hl1 >= -0.05 * atr:
                trend_persist += 0.55
            if hl2 >= -0.08 * atr:
                trend_persist += 0.45
            accept_hold = float(np.clip((float(l.tail(min(4, len(l))).min()) - (float(accept_line) - 0.18 * atr)) / max(1e-9, 0.28 * atr), 0.0, 1.0))
            trigger_prox = float(np.clip((float(c.iloc[-1]) - (float(break_trigger) - 0.45 * atr)) / max(1e-9, 0.45 * atr), 0.0, 1.0))
            close_pos = float(np.clip((float(c.iloc[-1]) - float(l.min())) / max(1e-9, float(h.max()) - float(l.min())), 0.0, 1.0))
        else:
            hl1 = float(h.iloc[-2] - h.iloc[-1]) if len(h) >= 2 else 0.0
            hl2 = float(h.iloc[-3] - h.iloc[-2]) if len(h) >= 3 else 0.0
            trend_persist = 0.0
            if hl1 >= -0.05 * atr:
                trend_persist += 0.55
            if hl2 >= -0.08 * atr:
                trend_persist += 0.45
            accept_hold = float(np.clip(((float(accept_line) + 0.18 * atr) - float(h.tail(min(4, len(h))).max())) / max(1e-9, 0.28 * atr), 0.0, 1.0))
            trigger_prox = float(np.clip(((float(break_trigger) + 0.45 * atr) - float(c.iloc[-1])) / max(1e-9, 0.45 * atr), 0.0, 1.0))
            close_pos = float(np.clip((float(h.max()) - float(c.iloc[-1])) / max(1e-9, float(h.max()) - float(l.min())), 0.0, 1.0))

        score = float(np.clip(0.34 * compression + 0.28 * trend_persist + 0.22 * accept_hold + 0.10 * trigger_prox + 0.06 * close_pos, 0.0, 1.0))
        coiled = bool(compression >= 0.55 and trend_persist >= 0.55 and accept_hold >= 0.45 and trigger_prox >= 0.30 and score >= 0.60)
        return {
            "score": score,
            "coiled": coiled,
            "compression": compression,
            "trend_persist": float(np.clip(trend_persist, 0.0, 1.0)),
            "accept_hold": accept_hold,
            "trigger_prox": trigger_prox,
        }
    except Exception:
        return {"score": 0.0, "coiled": False, "compression": 0.0, "trend_persist": 0.0, "accept_hold": 0.0, "trigger_prox": 0.0}




def _assess_compression_breakout(
    direction: str,
    df: pd.DataFrame,
    atr_last: float | None,
    *,
    lookback: int = 6,
    break_trigger: float | None = None,
) -> dict[str, float | bool]:
    """Detect tight-range compression followed by a real directional release.

    Used by both SCALP and RIDE so the system can recognize stored-pressure breakouts
    without requiring the move to look like a classic reclaim or a full pullback cycle.
    """
    try:
        if atr_last is None or not np.isfinite(atr_last) or float(atr_last) <= 0 or len(df) < max(lookback * 2 + 2, 10):
            return {"score": 0.0, "ready": False, "compression": 0.0, "breakout": 0.0, "volume": 0.0, "momentum": 0.0, "close_quality": 0.0}
        direction = str(direction).upper()
        atr = float(atr_last)
        n = int(max(4, lookback))
        recent = df.tail(n + 1).copy()
        prior = df.tail(n * 2 + 1).head(n).copy()
        if recent.empty or prior.empty:
            return {"score": 0.0, "ready": False, "compression": 0.0, "breakout": 0.0, "volume": 0.0, "momentum": 0.0, "close_quality": 0.0}
        h_recent = pd.to_numeric(recent.get('high'), errors='coerce').astype(float)
        l_recent = pd.to_numeric(recent.get('low'), errors='coerce').astype(float)
        c_recent = pd.to_numeric(recent.get('close'), errors='coerce').astype(float)
        o_recent = pd.to_numeric(recent.get('open'), errors='coerce').astype(float)
        v_recent = pd.to_numeric(recent.get('volume'), errors='coerce').astype(float) if 'volume' in recent else pd.Series(dtype=float)
        h_prior = pd.to_numeric(prior.get('high'), errors='coerce').astype(float)
        l_prior = pd.to_numeric(prior.get('low'), errors='coerce').astype(float)
        if h_recent.isna().all() or l_recent.isna().all() or c_recent.isna().all() or h_prior.isna().all() or l_prior.isna().all():
            return {"score": 0.0, "ready": False, "compression": 0.0, "breakout": 0.0, "volume": 0.0, "momentum": 0.0, "close_quality": 0.0}
        bar_ranges_recent = (h_recent - l_recent).clip(lower=0.0)
        bar_ranges_prior = (h_prior - l_prior).clip(lower=0.0)
        avg_recent_atr = float(bar_ranges_recent.tail(n).mean() / max(1e-9, atr))
        avg_prior_atr = float(bar_ranges_prior.tail(n).mean() / max(1e-9, atr)) if len(bar_ranges_prior) else avg_recent_atr
        total_recent_atr = float((float(h_recent.tail(n).max()) - float(l_recent.tail(n).min())) / max(1e-9, atr))
        total_prior_atr = float((float(h_prior.tail(n).max()) - float(l_prior.tail(n).min())) / max(1e-9, atr)) if len(h_prior) else total_recent_atr
        compression_q = float(np.clip(
            0.55 * np.clip((max(0.0, avg_prior_atr - avg_recent_atr) + 0.18) / 0.75, 0.0, 1.0)
            + 0.45 * np.clip((max(0.0, total_prior_atr - total_recent_atr) + 0.25) / 1.05, 0.0, 1.0),
            0.0,
            1.0,
        ))

        last_close = float(c_recent.iloc[-1])
        last_high = float(h_recent.iloc[-1])
        last_low = float(l_recent.iloc[-1])
        last_open = float(o_recent.iloc[-1]) if len(o_recent) else last_close
        prev_highs = h_recent.iloc[:-1] if len(h_recent) > 1 else h_recent
        prev_lows = l_recent.iloc[:-1] if len(l_recent) > 1 else l_recent
        if direction == 'LONG':
            trigger = float(break_trigger) if break_trigger is not None and np.isfinite(float(break_trigger)) else float(prev_highs.max())
            breakout_dist = float(last_close - trigger)
            breakout_q = float(np.clip((breakout_dist + 0.10 * atr) / max(1e-9, 0.38 * atr), 0.0, 1.0))
            rng = max(1e-9, last_high - last_low)
            close_quality = float(np.clip(0.55 * ((last_close - min(last_open, last_close)) / rng) + 0.45 * ((last_close - last_low) / rng), 0.0, 1.0))
        else:
            trigger = float(break_trigger) if break_trigger is not None and np.isfinite(float(break_trigger)) else float(prev_lows.min())
            breakout_dist = float(trigger - last_close)
            breakout_q = float(np.clip((breakout_dist + 0.10 * atr) / max(1e-9, 0.38 * atr), 0.0, 1.0))
            rng = max(1e-9, last_high - last_low)
            close_quality = float(np.clip(0.55 * ((max(last_open, last_close) - last_close) / rng) + 0.45 * ((last_high - last_close) / rng), 0.0, 1.0))

        volume_q = 0.5
        if len(v_recent) >= 4 and v_recent.notna().sum() >= 4:
            last_vol = float(v_recent.iloc[-1])
            baseline_vol = float(v_recent.iloc[:-1].median()) if len(v_recent.iloc[:-1]) else last_vol
            if baseline_vol > 0:
                volume_q = float(np.clip((last_vol / baseline_vol - 0.85) / 0.75, 0.0, 1.0))
            else:
                volume_q = 0.0
        momentum_q = 0.5
        if 'macd_hist' in df.columns:
            mh = pd.to_numeric(df['macd_hist'].tail(4), errors='coerce').dropna().astype(float)
            if len(mh) >= 3:
                if direction == 'LONG':
                    momentum_q = float(np.clip(0.55 * float(mh.iloc[-1] > mh.iloc[-2] > mh.iloc[-3]) + 0.45 * float(mh.iloc[-1] > 0), 0.0, 1.0))
                else:
                    momentum_q = float(np.clip(0.55 * float(mh.iloc[-1] < mh.iloc[-2] < mh.iloc[-3]) + 0.45 * float(mh.iloc[-1] < 0), 0.0, 1.0))
        score = float(np.clip(0.32 * compression_q + 0.26 * breakout_q + 0.18 * volume_q + 0.14 * momentum_q + 0.10 * close_quality, 0.0, 1.0))
        ready = bool(compression_q >= 0.55 and breakout_q >= 0.55 and volume_q >= 0.40 and momentum_q >= 0.40 and close_quality >= 0.48 and score >= 0.62)
        return {"score": score, "ready": ready, "compression": compression_q, "breakout": breakout_q, "volume": volume_q, "momentum": momentum_q, "close_quality": close_quality}
    except Exception:
        return {"score": 0.0, "ready": False, "compression": 0.0, "breakout": 0.0, "volume": 0.0, "momentum": 0.0, "close_quality": 0.0}

def compute_scalp_signal(
    symbol: str,
    ohlcv: pd.DataFrame,
    rsi_fast: pd.Series,
    rsi_slow: pd.Series,
    macd_hist: pd.Series,
    *,
    mode: str = "Cleaner signals",
    pro_mode: bool = False,

    # Time / bar guards
    allow_opening: bool = True,
    allow_midday: bool = False,
    allow_power: bool = True,
    allow_premarket: bool = False,
    allow_afterhours: bool = False,
    use_last_closed_only: bool = False,
    bar_closed_guard: bool = True,
    interval: str = "1min",

    # VWAP / Fib / HTF
    lookback_bars: int = 180,
    vwap_logic: str = "session",
    session_vwap_include_premarket: bool = False,
    session_vwap_include_afterhours: bool = False,
    fib_lookback_bars: int = 120,
    htf_bias: Optional[Dict[str, object]] = None,   # {bias, score, details}
    htf_strict: bool = False,

    # Liquidity / ORB / execution model
    killzone_preset: str = "Custom (use toggles)",
    liquidity_weighting: float = 0.55,
    orb_minutes: int = 15,
    entry_model: str = "VWAP reclaim limit",
    slippage_mode: str = "Off",
    fixed_slippage_cents: float = 0.02,
    atr_fraction_slippage: float = 0.15,

    # Score normalization
    target_atr_pct: float | None = None,
    tape_mode_enabled: bool = False,
    **_ignored: object,
) -> SignalResult:
    if len(ohlcv) < 60:
        return SignalResult(symbol, "NEUTRAL", 0, "Not enough data", None, None, None, None, None, None, "OFF", {})

    # --- Interval parsing ---
    # interval is typically like "1min", "5min", "15min", "30min", "60min"
    interval_mins = 1
    try:
        s = str(interval).lower().strip()
        if s.endswith("min"):
            interval_mins = int(float(s.replace("min", "").strip()))
        elif s.endswith("m"):
            interval_mins = int(float(s.replace("m", "").strip()))
        else:
            interval_mins = int(float(s))
    except Exception:
        interval_mins = 1

    # --- Killzone presets ---
    # Presets can optionally override the time-of-day allow toggles.
    kz = (killzone_preset or "Custom (use toggles)").strip()
    if kz == "Opening Drive":
        allow_opening, allow_midday, allow_power, allow_premarket, allow_afterhours = True, False, False, False, False
    elif kz == "Lunch Chop":
        allow_opening, allow_midday, allow_power, allow_premarket, allow_afterhours = False, True, False, False, False
    elif kz == "Power Hour":
        allow_opening, allow_midday, allow_power, allow_premarket, allow_afterhours = False, False, True, False, False
    elif kz == "Pre-market":
        allow_opening, allow_midday, allow_power, allow_premarket, allow_afterhours = False, False, False, True, False

    # --- Snapshot / bar-closed guards ---
    try:
        df_asof = _asof_slice(ohlcv.copy(), interval_mins=interval_mins, use_last_closed_only=use_last_closed_only, bar_closed_guard=bar_closed_guard)
    except Exception:
        df_asof = ohlcv.copy()

    cfg = PRESETS.get(mode, PRESETS["Cleaner signals"])

    df = df_asof.copy().tail(int(lookback_bars)).copy()
    # --- Attach indicator series onto df for downstream helpers that expect columns ---
    # Some callers pass RSI/MACD as separate Series; downstream logic may reference df["rsi5"]/df["rsi14"]/df["macd_hist"].
    # Align by index when possible; otherwise fall back to tail-alignment by length.
    def _attach_series(_df: pd.DataFrame, col: str, s) -> None:
        if s is None:
            return
        try:
            if isinstance(s, pd.Series):
                # Prefer index alignment
                if _df.index.equals(s.index):
                    _df[col] = s
                else:
                    _df[col] = s.reindex(_df.index)
                    # If reindex produced all-NaN (e.g., different tz), tail-align values
                    if _df[col].isna().all() and len(s) >= len(_df):
                        _df[col] = pd.Series(s.values[-len(_df):], index=_df.index)
            else:
                # list/np array
                arr = list(s)
                if len(arr) >= len(_df):
                    _df[col] = pd.Series(arr[-len(_df):], index=_df.index)
        except Exception:
            # Last resort: do nothing
            return

    _attach_series(df, "rsi5", rsi_fast)
    _attach_series(df, "rsi14", rsi_slow)
    _attach_series(df, "macd_hist", macd_hist)
    # Session VWAP windows are session-dependent. If the user enables scanning PM/AH but keeps
    # session VWAP restricted to RTH, VWAP-based logic becomes NaN during those windows.
    # As a product guardrail, automatically extend session VWAP to the scanned session(s).
    auto_vwap_fix = False
    if vwap_logic == "session":
        if allow_premarket and not session_vwap_include_premarket:
            session_vwap_include_premarket = True
            auto_vwap_fix = True
        if allow_afterhours and not session_vwap_include_afterhours:
            session_vwap_include_afterhours = True
            auto_vwap_fix = True

    df["vwap_cum"] = calc_vwap(df)
    df["vwap_sess"] = calc_session_vwap(
        df,
        include_premarket=session_vwap_include_premarket,
        include_afterhours=session_vwap_include_afterhours,
    )
    df["atr14"] = calc_atr(df, 14)
    df["ema20"] = calc_ema(df["close"], 20)
    df["ema50"] = calc_ema(df["close"], 50)

    # Pro: Trend strength (ADX) + direction (DI+/DI-)
    adx14 = plus_di = minus_di = None
    adx_ctx = {"adx": None, "plus_di": None, "minus_di": None, "di_spread": None, "adx_slope": 0.0, "adx_accel": 0.0, "dominant_side": None, "dominance_bars": 0, "regime": "unknown"}
    try:
        adx_s, pdi_s, mdi_s = calc_adx(df, 14)
        df["adx14"] = adx_s
        df["plus_di14"] = pdi_s
        df["minus_di14"] = mdi_s
        adx_ctx = calc_adx_context(adx_s, pdi_s, mdi_s)
        adx14 = float(adx_ctx.get("adx")) if adx_ctx.get("adx") is not None and np.isfinite(float(adx_ctx.get("adx"))) else None
        plus_di = float(adx_ctx.get("plus_di")) if adx_ctx.get("plus_di") is not None and np.isfinite(float(adx_ctx.get("plus_di"))) else None
        minus_di = float(adx_ctx.get("minus_di")) if adx_ctx.get("minus_di") is not None and np.isfinite(float(adx_ctx.get("minus_di"))) else None
    except Exception:
        adx14 = plus_di = minus_di = None

    # Keep a stable local alias for weak-tape gating and any upstream helpers
    # that expect the most recent ADX reading by this name.
    adx_last = float(adx14) if isinstance(adx14, (int, float)) and np.isfinite(adx14) else None

    rsi_fast = rsi_fast.reindex(df.index).ffill()
    rsi_slow = rsi_slow.reindex(df.index).ffill()
    macd_hist = macd_hist.reindex(df.index).ffill()

    close = df["close"]
    vol = df["volume"]
    vwap_use = df["vwap_sess"] if vwap_logic == "session" else df["vwap_cum"]
    df["vwap_use"] = vwap_use  # unify VWAP ref for downstream TP/expected-excursion logic

    last_ts = df.index[-1]
    # Feed freshness diagnostics (ET): this helps catch the "AsOf is yesterday" case.
    try:
        now_et = pd.Timestamp.now(tz="America/New_York")
        ts_et = last_ts.tz_convert("America/New_York") if last_ts.tzinfo is not None else last_ts.tz_localize("America/New_York")
        data_age_min = float((now_et - ts_et).total_seconds() / 60.0)
        extras_feed = {"data_age_min": data_age_min, "data_date": str(ts_et.date())}
    except Exception:
        extras_feed = {"data_age_min": None, "data_date": None}
    session = classify_session(last_ts)
    phase = classify_liquidity_phase(last_ts)

    # IMPORTANT PRODUCT RULE:
    # Time-of-day toggles should NOT *block* scoring/alerts.
    # They are preference hints used for liquidity weighting and optional UI filtering.
    # A great setup is a great setup regardless of clock-time.
    allowed = (
        (session == "OPENING" and allow_opening)
        or (session == "MIDDAY" and allow_midday)
        or (session == "POWER" and allow_power)
        or (session == "PREMARKET" and allow_premarket)
        or (session == "AFTERHOURS" and allow_afterhours)
    )
    last_bar_price = float(close.iloc[-1])
    try:
        live_last_price = float(_ignored.get("live_last_price")) if _ignored.get("live_last_price") is not None else float("nan")
    except Exception:
        live_last_price = float("nan")
    last_price = float(live_last_price) if np.isfinite(live_last_price) and float(live_last_price) > 0 else float(last_bar_price)

    # --- Safety: define reference VWAP early so it is always in-scope ---
    # The PRE-alert logic and entry/TP models reference `ref_vwap`. In some code paths
    # (depending on toggles/returns), `ref_vwap` can otherwise be referenced before it
    # is assigned, causing UnboundLocalError.
    try:
        _rv = vwap_use.iloc[-1]
        ref_vwap: float | None = float(_rv) if _rv is not None and np.isfinite(_rv) else None
    except Exception:
        ref_vwap = None

    atr_last = float(df["atr14"].iloc[-1]) if np.isfinite(df["atr14"].iloc[-1]) else 0.0
    buffer = 0.25 * atr_last if atr_last else 0.0
    atr_pct = (atr_last / last_price) if last_price else 0.0

    # Liquidity weighting: scale contributions based on the current liquidity phase.
    # liquidity_weighting in [0..1] controls how strongly we care about time-of-day liquidity.
    #  - OPENING / POWER: boost
    #  - MIDDAY: discount
    #  - PREMARKET / AFTERHOURS: heavier discount
    base = 1.0
    if phase in ("OPENING", "POWER"):
        base = 1.15
    elif phase in ("MIDDAY",):
        base = 0.85
    elif phase in ("PREMARKET", "AFTERHOURS"):
        base = 0.75
    try:
        w = max(0.0, min(1.0, float(liquidity_weighting)))
    except Exception:
        w = 0.55
    liquidity_mult = 1.0 + w * (base - 1.0)

    extras: Dict[str, Any] = {
        "vwap_logic": vwap_logic,
        "session_vwap_include_premarket": bool(session_vwap_include_premarket),
        "session_vwap_include_afterhours": bool(session_vwap_include_afterhours),
        "auto_vwap_session_fix": bool(auto_vwap_fix),
        "vwap_session": float(df["vwap_sess"].iloc[-1]) if np.isfinite(df["vwap_sess"].iloc[-1]) else None,
        "vwap_cumulative": float(df["vwap_cum"].iloc[-1]) if np.isfinite(df["vwap_cum"].iloc[-1]) else None,
        "ema20": float(df["ema20"].iloc[-1]) if np.isfinite(df["ema20"].iloc[-1]) else None,
        "ema50": float(df["ema50"].iloc[-1]) if np.isfinite(df["ema50"].iloc[-1]) else None,
        "adx14": adx14,
        "plus_di14": plus_di,
        "minus_di14": minus_di,
        "atr14": atr_last,
        "atr_pct": atr_pct,
        "adx14": adx14,
        "plus_di14": plus_di,
        "minus_di14": minus_di,
        "liquidity_phase": phase,
        "liquidity_mult": liquidity_mult,
        **_vwap_basis_metadata(
            engine=mode,
            vwap_logic=vwap_logic,
            session_vwap_include_premarket=bool(session_vwap_include_premarket),
            session_vwap_include_afterhours=bool(session_vwap_include_afterhours),
        ),
        "fib_lookback_bars": int(fib_lookback_bars),
        "htf_bias": htf_bias,
        "htf_strict": bool(htf_strict),
        "target_atr_pct": (float(target_atr_pct) if target_atr_pct is not None else None),
        # Diagnostics: whether the current session is inside the user's preferred windows.
        # This is NEVER used to block actionability.
        "time_filter_allowed": bool(allowed),
    }

    # Attach feed diagnostics (age/date) to every result.
    try:
        extras.update(extras_feed)
    except Exception:
        pass

    # merge feed freshness fields
    extras.update(extras_feed)

    # Trader pressure states: existing ADX/DI + RSI + MACD + volume interpreted as pressure transfer,
    # not separate payloads or duplicated indicators. Used only for internal scoring/entry timing.
    pressure_states = _indicator_pressure_states(df, adx_ctx)
    extras["pressure_states"] = pressure_states

    # Do not early-return when outside preferred windows.
    # We keep scoring normally and simply annotate the result.

    # VWAP event
    was_below_vwap = (close.shift(3) < vwap_use.shift(3)).iloc[-1] or (close.shift(5) < vwap_use.shift(5)).iloc[-1]
    reclaim_vwap = (close.iloc[-1] > vwap_use.iloc[-1]) and (close.shift(1).iloc[-1] <= vwap_use.shift(1).iloc[-1])

    was_above_vwap = (close.shift(3) > vwap_use.shift(3)).iloc[-1] or (close.shift(5) > vwap_use.shift(5)).iloc[-1]
    reject_vwap = (close.iloc[-1] < vwap_use.iloc[-1]) and (close.shift(1).iloc[-1] >= vwap_use.shift(1).iloc[-1])

    # RSI + MACD events
    rsi5 = float(rsi_fast.iloc[-1])
    rsi14 = float(rsi_slow.iloc[-1])

    # Pro: RSI divergence (RSI-5 vs price pivots)
    rsi_div = None
    if pro_mode:
        try:
            rsi_div = _detect_rsi_divergence(df, rsi_fast, rsi_slow, lookback=int(min(220, max(80, lookback_bars))))
        except Exception:
            rsi_div = None
    extras["rsi_divergence"] = rsi_div

    rsi_snap = (rsi5 >= 30 and float(rsi_fast.shift(1).iloc[-1]) < 30) or (rsi5 >= 25 and float(rsi_fast.shift(1).iloc[-1]) < 25)
    rsi_downshift = (rsi5 <= 70 and float(rsi_fast.shift(1).iloc[-1]) > 70) or (rsi5 <= 75 and float(rsi_fast.shift(1).iloc[-1]) > 75)

    macd_turn_up = (macd_hist.iloc[-1] > macd_hist.shift(1).iloc[-1]) and (macd_hist.shift(1).iloc[-1] > macd_hist.shift(2).iloc[-1])
    macd_turn_down = (macd_hist.iloc[-1] < macd_hist.shift(1).iloc[-1]) and (macd_hist.shift(1).iloc[-1] < macd_hist.shift(2).iloc[-1])

    def _macd_build_bonus(hist: pd.Series, side: str, structure_ok: bool) -> tuple[int, str | None]:
        """Small capped early-build momentum bonus using existing MACD histogram only.

        This is intentionally subordinate to the existing MACD turn logic. It is only meant
        to help borderline reversal setups surface a little earlier when momentum is clearly
        improving and local structure is already at least minimally credible.
        """
        try:
            h0 = float(hist.iloc[-1]); h1 = float(hist.shift(1).iloc[-1]); h2 = float(hist.shift(2).iloc[-1]); h3 = float(hist.shift(3).iloc[-1])
        except Exception:
            return 0, None
        if not (np.isfinite(h0) and np.isfinite(h1) and np.isfinite(h2) and np.isfinite(h3)):
            return 0, None
        if not bool(structure_ok):
            return 0, None

        if str(side).upper() == "LONG":
            rising3 = bool(h0 > h1 > h2)
            rising4 = bool(rising3 and h2 > h3)
            if h0 > 0 and h1 > 0 and rising3:
                return 6, "MACD build expanding positive"
            if h0 > 0 and h1 <= 0 and h0 > h1:
                return 5, "MACD energy flip positive"
            if h0 <= 0 and rising4:
                return 4, "MACD red bars shrinking"
            if h0 <= 0 and rising3:
                return 3, "MACD build improving"
            return 0, None

        falling3 = bool(h0 < h1 < h2)
        falling4 = bool(falling3 and h2 < h3)
        if h0 < 0 and h1 < 0 and falling3:
            return 6, "MACD build expanding negative"
        if h0 < 0 and h1 >= 0 and h0 < h1:
            return 5, "MACD energy flip negative"
        if h0 >= 0 and falling4:
            return 4, "MACD green bars shrinking"
        if h0 >= 0 and falling3:
            return 3, "MACD build weakening"
        return 0, None

    # Volume confirmation (quality-aware for volatile $1-$5 names)
    # Raw relative volume alone is not enough in small caps because one panic spike or wicky churn
    # candle can satisfy a median-volume check without providing healthy participation.
    vol_med = vol.rolling(30, min_periods=10).median().iloc[-1]
    vol_last = float(vol.iloc[-1]) if len(vol) else 0.0
    rel_vol = (vol_last / float(vol_med)) if (np.isfinite(vol_med) and float(vol_med) > 0) else 0.0

    try:
        _o_v = float(df["open"].iloc[-1])
        _h_v = float(df["high"].iloc[-1])
        _l_v = float(df["low"].iloc[-1])
        _c_v = float(df["close"].iloc[-1])
        _rng_v = max(_h_v - _l_v, 1e-9)
        _body_frac_v = abs(_c_v - _o_v) / _rng_v
        _close_loc_v = (_c_v - _l_v) / _rng_v
    except Exception:
        _o_v = _c_v = last_price
        _body_frac_v = 0.0
        _close_loc_v = 0.5

    dollar_flow = (close.astype(float) * vol.astype(float)).rolling(30, min_periods=10).median().iloc[-1]
    last_dollar_flow = float(last_price) * max(vol_last, 0.0)
    dollar_flow_ok = bool(last_dollar_flow >= max(120000.0, 0.65 * float(dollar_flow))) if np.isfinite(dollar_flow) else bool(last_dollar_flow >= 120000.0)

    mult = float(cfg["vol_multiplier"])
    base_relvol_ok = bool(np.isfinite(vol_med) and vol_last >= mult * float(vol_med))
    wicky_chaos = bool(_body_frac_v < 0.22)
    bull_participation = bool((_c_v >= _o_v and _body_frac_v >= 0.35 and _close_loc_v >= 0.58) or (_c_v > _o_v and reclaim_vwap and macd_turn_up))
    bear_participation = bool((_c_v <= _o_v and _body_frac_v >= 0.35 and _close_loc_v <= 0.42) or (_c_v < _o_v and reject_vwap and macd_turn_down))

    vol_ok_long = bool(base_relvol_ok and dollar_flow_ok and bull_participation and not (wicky_chaos and _close_loc_v < 0.60))
    vol_ok_short = bool(base_relvol_ok and dollar_flow_ok and bear_participation and not (wicky_chaos and _close_loc_v > 0.40))
    # Generic volume event remains for diagnostics/backward compatibility, but live directional scoring
    # should use the side-aligned booleans above.
    vol_ok = bool(vol_ok_long or vol_ok_short)
    extras["rel_volume"] = float(rel_vol)
    extras["dollar_flow_ok"] = bool(dollar_flow_ok)
    extras["wicky_volume_bar"] = bool(wicky_chaos)
    extras["vol_ok_long"] = bool(vol_ok_long)
    extras["vol_ok_short"] = bool(vol_ok_short)

    # Swings
    swing_low_mask = rolling_swing_lows(df["low"], left=3, right=3)
    recent_swing_lows = df.loc[swing_low_mask, "low"].tail(6)
    recent_swing_low = float(recent_swing_lows.iloc[-1]) if len(recent_swing_lows) else float(df["low"].tail(12).min())

    swing_high_mask = rolling_swing_highs(df["high"], left=3, right=3)
    recent_swing_highs = df.loc[swing_high_mask, "high"].tail(6)
    recent_swing_high = float(recent_swing_highs.iloc[-1]) if len(recent_swing_highs) else float(df["high"].tail(12).max())

    # Trend context (EMA)
    trend_long_ok = bool((close.iloc[-1] >= df["ema20"].iloc[-1]) and (df["ema20"].iloc[-1] >= df["ema50"].iloc[-1]))
    trend_short_ok = bool((close.iloc[-1] <= df["ema20"].iloc[-1]) and (df["ema20"].iloc[-1] <= df["ema50"].iloc[-1]))
    extras["trend_long_ok"] = trend_long_ok
    extras["trend_short_ok"] = trend_short_ok

    # Fib context (scoring + fib-anchored take profits)
    seg = df.tail(int(min(max(60, fib_lookback_bars), len(df))))
    hi = float(seg["high"].max())
    lo = float(seg["low"].min())
    rng = hi - lo

    fib_name = fib_level = fib_dist = None
    fib_near_long = fib_near_short = False
    fib_bias = "range"
    retr = _fib_retracement_levels(hi, lo) if rng > 0 else []
    fib_name, fib_level, fib_dist = _closest_level(last_price, retr)

    if rng > 0:
        pos = (last_price - lo) / rng
        if pos >= 0.60:
            fib_bias = "up"
        elif pos <= 0.40:
            fib_bias = "down"
        else:
            fib_bias = "range"

    if fib_level is not None and fib_dist is not None:
        # Volatility-aware proximity: tighter when ATR is small, wider when ATR is large.
        # For scalping, we don't want "near fib" firing when price is far away in ATR terms.
        prox = None
        if atr_last is not None and np.isfinite(float(atr_last)) and float(atr_last) > 0:
            prox = max(0.35 * float(atr_last), 0.0015 * float(last_price))
        else:
            prox = 0.002 * float(last_price)
        near = float(fib_dist) <= max(float(buffer), float(prox))
        if near:
            if fib_bias == "up":
                fib_near_long = True
            elif fib_bias == "down":
                fib_near_short = True

    extras["fib_hi"] = hi if rng > 0 else None
    extras["fib_lo"] = lo if rng > 0 else None
    extras["fib_bias"] = fib_bias
    extras["fib_closest"] = {"name": fib_name, "level": fib_level, "dist": fib_dist}
    extras["fib_near_long"] = fib_near_long
    extras["fib_near_short"] = fib_near_short

    # Liquidity sweeps + ORB context
    # Use session-aware levels (prior day high/low, premarket high/low, ORB high/low) when possible.
    try:
        levels = _session_liquidity_levels(df, interval_mins=interval_mins, orb_minutes=int(orb_minutes))
    except Exception:
        levels = {}

    extras["liq_levels"] = levels

    # Fallback swing-based levels (always available)
    prior_swing_high = float(recent_swing_highs.iloc[-1]) if len(recent_swing_highs) else float(df["high"].tail(30).max())
    prior_swing_low = float(recent_swing_lows.iloc[-1]) if len(recent_swing_lows) else float(df["low"].tail(30).min())

    # Sweep definition:
    # - Primary: wick through a key level, then close back inside (ICT-style)
    # - Secondary fallback: take + reclaim against recent swing
    bull_sweep = False
    bear_sweep = False
    if pro_mode and levels:
        sweep = _detect_liquidity_sweep(df, levels, atr_last=atr_last, buffer=buffer)
        extras["liquidity_sweep"] = sweep
        if isinstance(sweep, dict) and sweep.get("type"):
            stype = str(sweep.get("type")).lower()
            bull_sweep = stype.startswith("bull")
            bear_sweep = stype.startswith("bear")
    else:
        # Fallback sweeps should still require meaningful displacement so a tiny poke-through
        # does not masquerade as a real liquidity raid.
        fallback_disp_ok = True
        try:
            if atr_last is not None and np.isfinite(float(atr_last)) and float(atr_last) > 0:
                fallback_disp_ok = float(df["high"].iloc[-1] - df["low"].iloc[-1]) >= 0.8 * float(atr_last)
        except Exception:
            fallback_disp_ok = True
        bull_sweep = bool((df["low"].iloc[-1] < prior_swing_low) and (df["close"].iloc[-1] > prior_swing_low) and fallback_disp_ok)
        bear_sweep = bool((df["high"].iloc[-1] > prior_swing_high) and (df["close"].iloc[-1] < prior_swing_high) and fallback_disp_ok)

    extras["bull_liquidity_sweep"] = bool(bull_sweep)
    extras["bear_liquidity_sweep"] = bool(bear_sweep)

    # ORB bias (upgraded): 3-stage sequence (break → accept → retest)
    orb_high = levels.get("orb_high")
    orb_low = levels.get("orb_low")
    extras["orb_high"] = orb_high
    extras["orb_low"] = orb_low

    orb_seq = _orb_three_stage(
        df,
        orb_high=float(orb_high) if orb_high is not None else None,
        orb_low=float(orb_low) if orb_low is not None else None,
        buffer=float(buffer),
        lookback_bars=int(max(24, orb_minutes * 3)),  # ~last ~2 hours on 5m, ~6 bars on 1m
        accept_bars=2,
    )
    orb_bull = bool(orb_seq.get("bull_orb_seq"))
    orb_bear = bool(orb_seq.get("bear_orb_seq"))
    # keep break-only flags for diagnostics/UI
    extras["orb_bull_break"] = bool(orb_seq.get("bull_break"))
    extras["orb_bear_break"] = bool(orb_seq.get("bear_break"))
    extras["orb_bull_seq"] = orb_bull
    extras["orb_bear_seq"] = orb_bear

    # Compression breakout detection: multi-speed on 5m bars.
    # Fast window catches fresh ignition, standard catches clean squeeze release,
    # and slow window catches broader pressure coils without letting context dominate the trigger.
    _scalp_comp_long_windows = {
        "fast": _assess_compression_breakout("LONG", df, atr_last, lookback=3, break_trigger=float(orb_high) if orb_high is not None else None),
        "standard": _assess_compression_breakout("LONG", df, atr_last, lookback=5, break_trigger=float(orb_high) if orb_high is not None else None),
        "slow": _assess_compression_breakout("LONG", df, atr_last, lookback=8, break_trigger=float(orb_high) if orb_high is not None else None),
    }
    _scalp_comp_short_windows = {
        "fast": _assess_compression_breakout("SHORT", df, atr_last, lookback=3, break_trigger=float(orb_low) if orb_low is not None else None),
        "standard": _assess_compression_breakout("SHORT", df, atr_last, lookback=5, break_trigger=float(orb_low) if orb_low is not None else None),
        "slow": _assess_compression_breakout("SHORT", df, atr_last, lookback=8, break_trigger=float(orb_low) if orb_low is not None else None),
    }
    scalp_compression_long = max(_scalp_comp_long_windows.values(), key=lambda x: float(x.get("score") or 0.0))
    scalp_compression_short = max(_scalp_comp_short_windows.values(), key=lambda x: float(x.get("score") or 0.0))
    compression_breakout_long = bool(any(bool(v.get("ready") or False) for v in _scalp_comp_long_windows.values()))
    compression_breakout_short = bool(any(bool(v.get("ready") or False) for v in _scalp_comp_short_windows.values()))
    extras["compression_breakout_long"] = bool(compression_breakout_long)
    extras["compression_breakout_short"] = bool(compression_breakout_short)
    extras["compression_breakout_long_score"] = float(scalp_compression_long.get("score") or 0.0)
    extras["compression_breakout_short_score"] = float(scalp_compression_short.get("score") or 0.0)
    extras["compression_breakout_long_window"] = str(max(_scalp_comp_long_windows, key=lambda k: float(_scalp_comp_long_windows[k].get("score") or 0.0)))
    extras["compression_breakout_short_window"] = str(max(_scalp_comp_short_windows, key=lambda k: float(_scalp_comp_short_windows[k].get("score") or 0.0)))

    # FVG + OB + Breaker
    bull_fvg, bear_fvg = detect_fvg(df.tail(60))
    extras["bull_fvg"] = bull_fvg
    extras["bear_fvg"] = bear_fvg

    ob_bull = find_order_block(df, df["atr14"], side="bull", lookback=35)
    ob_bear = find_order_block(df, df["atr14"], side="bear", lookback=35)
    extras["bull_ob"] = ob_bull
    extras["bear_ob"] = ob_bear
    bull_ob_retest = bool(ob_bull[0] is not None and in_zone(last_price, ob_bull[0], ob_bull[1], buffer=buffer))
    bear_ob_retest = bool(ob_bear[0] is not None and in_zone(last_price, ob_bear[0], ob_bear[1], buffer=buffer))
    extras["bull_ob_retest"] = bull_ob_retest
    extras["bear_ob_retest"] = bear_ob_retest

    brk_bull = find_breaker_block(df, df["atr14"], side="bull", lookback=60)
    brk_bear = find_breaker_block(df, df["atr14"], side="bear", lookback=60)
    extras["bull_breaker"] = brk_bull
    extras["bear_breaker"] = brk_bear
    bull_breaker_retest = bool(brk_bull[0] is not None and in_zone(last_price, brk_bull[0], brk_bull[1], buffer=buffer))
    bear_breaker_retest = bool(brk_bear[0] is not None and in_zone(last_price, brk_bear[0], brk_bear[1], buffer=buffer))
    extras["bull_breaker_retest"] = bull_breaker_retest
    extras["bear_breaker_retest"] = bear_breaker_retest

    displacement = bool(atr_last and float(df["high"].iloc[-1] - df["low"].iloc[-1]) >= 1.5 * atr_last)
    extras["displacement"] = displacement

    # Directional displacement should only strengthen the side that actually owns the impulse.
    # For volatile $1-$5 names this is intentionally conservative: we want a real body, a strong
    # close location, and at least some directional continuity vs the prior close so a chaotic
    # wide-range doji does not reward both sides.
    try:
        _o = float(df["open"].iloc[-1])
        _h = float(df["high"].iloc[-1])
        _l = float(df["low"].iloc[-1])
        _c = float(df["close"].iloc[-1])
        _pc = float(df["close"].iloc[-2]) if len(df) >= 2 else _c
        _rng = max(_h - _l, 1e-9)
        _body_frac = abs(_c - _o) / _rng
        _close_loc = (_c - _l) / _rng
    except Exception:
        _body_frac = 0.0
        _close_loc = 0.5
        _pc = last_price
        _o = _c = last_price

    bull_displacement = bool(
        displacement and (
            ((_c > _o) and (_body_frac >= 0.45) and (_close_loc >= 0.68) and (_c >= _pc))
            or ((_c > _o) and reclaim_vwap and macd_turn_up and (_close_loc >= 0.60))
        )
    )
    bear_displacement = bool(
        displacement and (
            ((_c < _o) and (_body_frac >= 0.45) and (_close_loc <= 0.32) and (_c <= _pc))
            or ((_c < _o) and reject_vwap and macd_turn_down and (_close_loc <= 0.40))
        )
    )

    # In ambiguous wide-range bars, do not reward both sides. Let the rest of the engine decide.
    if bull_displacement and bear_displacement:
        bull_displacement = False
        bear_displacement = False

    extras["bull_displacement"] = bull_displacement
    extras["bear_displacement"] = bear_displacement
    extras["displacement_body_frac"] = float(_body_frac)
    extras["displacement_close_loc"] = float(_close_loc)

    # Early-ignition candle quality: used for controlled first-leg PRE overrides.
    bull_ignition_candle_quality = float(np.clip(0.55 * float(_body_frac) + 0.45 * float(_close_loc), 0.0, 1.0))
    bear_close_loc = float(np.clip(1.0 - float(_close_loc), 0.0, 1.0))
    bear_ignition_candle_quality = float(np.clip(0.55 * float(_body_frac) + 0.45 * bear_close_loc, 0.0, 1.0))
    extras["bull_ignition_candle_quality"] = float(bull_ignition_candle_quality)
    extras["bear_ignition_candle_quality"] = float(bear_ignition_candle_quality)

    # HTF bias overlay
    htf_b = None
    if isinstance(htf_bias, dict):
        htf_b = htf_bias.get("bias")
    extras["htf_bias_value"] = htf_b

    # --- Scoring (raw) ---
    contrib: Dict[str, Dict[str, int]] = {"LONG": {}, "SHORT": {}}

    def _add(side: str, key: str, pts: int, why: str | None = None):
        nonlocal long_points, short_points
        if side == "LONG":
            long_points += int(pts)
            contrib["LONG"][key] = contrib["LONG"].get(key, 0) + int(pts)
            if why:
                long_reasons.append(why)
        else:
            short_points += int(pts)
            contrib["SHORT"][key] = contrib["SHORT"].get(key, 0) + int(pts)
            if why:
                short_reasons.append(why)

    micro_hl_pre = bool(df["low"].tail(12).iloc[-1] > df["low"].tail(12).min())
    micro_lh_pre = bool(df["high"].tail(12).iloc[-1] < df["high"].tail(12).max())
    macd_build_long_pts, macd_build_long_why = _macd_build_bonus(macd_hist, "LONG", micro_hl_pre)
    macd_build_short_pts, macd_build_short_why = _macd_build_bonus(macd_hist, "SHORT", micro_lh_pre)
    extras["macd_build_long_bonus"] = int(macd_build_long_pts)
    extras["macd_build_short_bonus"] = int(macd_build_short_pts)

    long_points = 0
    long_reasons: List[str] = []
    if was_below_vwap and reclaim_vwap:
        _add("LONG", "vwap_event", 35, f"VWAP reclaim ({vwap_logic})")
    if rsi_snap and rsi14 < 60:
        _add("LONG", "rsi_snap", 20, "RSI-5 snapback (RSI-14 ok)")
    if macd_turn_up:
        _add("LONG", "macd_turn", 20, "MACD hist turning up")
    if macd_build_long_pts > 0:
        _add("LONG", "macd_build", int(macd_build_long_pts), macd_build_long_why)
    if vol_ok_long:
        _add("LONG", "volume", int(round(15 * liquidity_mult)), "Volume confirmation")
    if micro_hl_pre:
        _add("LONG", "micro_structure", 10, "Higher-low micro structure")
    if compression_breakout_long:
        _add("LONG", "compression_breakout", 12, "Compression breakout release")

    short_points = 0
    short_reasons: List[str] = []
    if was_above_vwap and reject_vwap:
        _add("SHORT", "vwap_event", 35, f"VWAP rejection ({vwap_logic})")
    if rsi_downshift and rsi14 > 40:
        _add("SHORT", "rsi_downshift", 20, "RSI-5 downshift (RSI-14 ok)")
    if macd_turn_down:
        _add("SHORT", "macd_turn", 20, "MACD hist turning down")
    if macd_build_short_pts > 0:
        _add("SHORT", "macd_build", int(macd_build_short_pts), macd_build_short_why)
    if vol_ok_short:
        _add("SHORT", "volume", int(round(15 * liquidity_mult)), "Volume confirmation")
    if micro_lh_pre:
        _add("SHORT", "micro_structure", 10, "Lower-high micro structure")
    if compression_breakout_short:
        _add("SHORT", "compression_breakout", 12, "Compression breakdown release")

    # Trader-pressure scoring: modest, capped nudges that help SCALP detect pressure shift before
    # full VWAP/RSI/MACD confirmation. They do not replace the existing confirmation framework.
    try:
        _ps_long = float((pressure_states or {}).get("long_pressure_score") or 0.0)
        _ps_short = float((pressure_states or {}).get("short_pressure_score") or 0.0)
        if _ps_long >= 2.8:
            _add("LONG", "pressure_shift", 6, "Pressure shift: DI/RSI/MACD/volume")
        elif _ps_long >= 2.0:
            _add("LONG", "pressure_shift", 4, "Early pressure shift")
        if _ps_short >= 2.8:
            _add("SHORT", "pressure_shift", 6, "Pressure shift: DI/RSI/MACD/volume")
        elif _ps_short >= 2.0:
            _add("SHORT", "pressure_shift", 4, "Early pressure shift")
    except Exception:
        pass

    # Fib scoring (volatility-aware, cluster-gated)
    # Fib/FVG should only matter when clustered with structure + volatility context.
    micro_hl = bool(df["low"].tail(12).iloc[-1] > df["low"].tail(12).min())
    micro_lh = bool(df["high"].tail(12).iloc[-1] < df["high"].tail(12).max())
    long_structure_ok = bool((was_below_vwap and reclaim_vwap) or micro_hl or orb_bull)
    short_structure_ok = bool((was_above_vwap and reject_vwap) or micro_lh or orb_bear)
    vol_context_ok_long = bool(vol_ok_long or bull_displacement or displacement)
    vol_context_ok_short = bool(vol_ok_short or bear_displacement or displacement)

    if fib_near_long and fib_name is not None and long_structure_ok and vol_context_ok_long:
        add = 15 if ("0.5" in fib_name or "0.618" in fib_name) else 8
        _add("LONG", "fib", add, f"Fib cluster ({fib_name})")
    if fib_near_short and fib_name is not None and short_structure_ok and vol_context_ok_short:
        add = 15 if ("0.5" in fib_name or "0.618" in fib_name) else 8
        _add("SHORT", "fib", add, f"Fib cluster ({fib_name})")


    # Pro structure scoring
    if pro_mode:
        if isinstance(rsi_div, dict) and rsi_div.get("type") == "bull":
            _add("LONG", "rsi_divergence", 22, "RSI bullish divergence")
        if isinstance(rsi_div, dict) and rsi_div.get("type") == "bear":
            _add("SHORT", "rsi_divergence", 22, "RSI bearish divergence")
        if bull_sweep:
            _add("LONG", "liquidity_sweep", int(round(20 * liquidity_mult)), "Liquidity sweep (low)")
        if bear_sweep:
            _add("SHORT", "liquidity_sweep", int(round(20 * liquidity_mult)), "Liquidity sweep (high)")
        if orb_bull:
            _add("LONG", "orb", int(round(12 * liquidity_mult)), f"ORB seq (break→accept→retest, {orb_minutes}m)")
        if orb_bear:
            _add("SHORT", "orb", int(round(12 * liquidity_mult)), f"ORB seq (break→accept→retest, {orb_minutes}m)")
        if bull_ob_retest:
            _add("LONG", "order_block", 15, "Bullish order block retest")
        if bear_ob_retest:
            _add("SHORT", "order_block", 15, "Bearish order block retest")
                # FVG only matters when price is actually interacting with the gap AND structure/vol context agrees.
        if bull_fvg is not None and isinstance(bull_fvg, (tuple, list)) and len(bull_fvg) == 2:
            z0, z1 = float(min(bull_fvg)), float(max(bull_fvg))
            near_fvg = (last_price >= z0 - buffer) and (last_price <= z1 + buffer)
            if near_fvg and long_structure_ok and vol_context_ok_long:
                _add("LONG", "fvg", 10, "Bullish FVG cluster")
        if bear_fvg is not None and isinstance(bear_fvg, (tuple, list)) and len(bear_fvg) == 2:
            z0, z1 = float(min(bear_fvg)), float(max(bear_fvg))
            near_fvg = (last_price >= z0 - buffer) and (last_price <= z1 + buffer)
            if near_fvg and short_structure_ok and vol_context_ok_short:
                _add("SHORT", "fvg", 10, "Bearish FVG cluster")
        if bull_breaker_retest:
            _add("LONG", "breaker", 20, "Bullish breaker retest")
        if bear_breaker_retest:
            _add("SHORT", "breaker", 20, "Bearish breaker retest")
        if bull_displacement:
            _add("LONG", "displacement", 5, "Bullish displacement")
        if bear_displacement:
            _add("SHORT", "displacement", 5, "Bearish displacement")

        # ADX trend-strength bonus (directional): helps avoid low-energy chop.
        # - If ADX is strong and DI agrees with direction => small bonus.
        # - If ADX is very low => mild penalty (but don't over-filter reversal setups).
        try:
            adx_val = float(adx14) if adx14 is not None else None
            pdi_val = float(plus_di) if plus_di is not None else None
            mdi_val = float(minus_di) if minus_di is not None else None
        except Exception:
            adx_val = pdi_val = mdi_val = None

        long_reversal_cluster = int(bool(was_below_vwap and reclaim_vwap)) + int(bool(rsi_snap and rsi14 < 60)) + int(bool(macd_turn_up)) + int(bool(micro_hl_pre))
        short_reversal_cluster = int(bool(was_above_vwap and reject_vwap)) + int(bool(rsi_downshift and rsi14 > 40)) + int(bool(macd_turn_down)) + int(bool(micro_lh_pre))

        if adx_val is not None and np.isfinite(adx_val):
            adx_regime = str(adx_ctx.get("regime") or "unknown")
            adx_dom = str(adx_ctx.get("dominant_side") or "")
            adx_spread = float(adx_ctx.get("di_spread") or 0.0)
            adx_slope = float(adx_ctx.get("adx_slope") or 0.0)
            adx_dom_bars = int(adx_ctx.get("dominance_bars") or 0)

            long_transition_ok = bool(long_reversal_cluster >= 3 and adx_dom == "SHORT" and adx_slope >= -0.5)
            short_transition_ok = bool(short_reversal_cluster >= 3 and adx_dom == "LONG" and adx_slope >= -0.5)

            if adx_regime in ("strengthening", "healthy_trend", "emerging") and adx_spread >= 5.0:
                if adx_dom == "LONG":
                    bonus = 8 if adx_regime in ("strengthening", "healthy_trend") else 6
                    if adx_slope > 1.0:
                        bonus += 1
                    _add("LONG", "adx_trend", bonus, f"ADX {adx_regime.replace('_', ' ')} (DI+)")
                    if short_transition_ok and adx_regime == "emerging":
                        _add("SHORT", "adx_transition", 3, "ADX transition setup against prior DI+")
                elif adx_dom == "SHORT":
                    bonus = 8 if adx_regime in ("strengthening", "healthy_trend") else 6
                    if adx_slope > 1.0:
                        bonus += 1
                    _add("SHORT", "adx_trend", bonus, f"ADX {adx_regime.replace('_', ' ')} (DI-)")
                    if long_transition_ok and adx_regime == "emerging":
                        _add("LONG", "adx_transition", 3, "ADX transition setup against prior DI-")
            elif adx_regime in ("dead_chop", "coiling"):
                penalty = 5 if adx_regime == "dead_chop" else 3
                long_penalty = max(0, penalty - (2 if long_reversal_cluster >= 3 else 0))
                short_penalty = max(0, penalty - (2 if short_reversal_cluster >= 3 else 0))
                long_points = max(0, long_points - long_penalty)
                short_points = max(0, short_points - short_penalty)
                if long_penalty:
                    contrib["LONG"]["adx_chop_penalty"] = contrib["LONG"].get("adx_chop_penalty", 0) - long_penalty
                if short_penalty:
                    contrib["SHORT"]["adx_chop_penalty"] = contrib["SHORT"].get("adx_chop_penalty", 0) - short_penalty
            elif adx_regime in ("mature_trend", "exhausting") and adx_dom_bars >= 2:
                fade_penalty = 3 if adx_regime == "mature_trend" else 5
                if adx_dom == "LONG":
                    applied = 0 if short_transition_ok else fade_penalty
                    short_bonus = 2 if short_transition_ok else 0
                    long_points = max(0, long_points - applied)
                    if applied:
                        contrib["LONG"]["adx_late_move_penalty"] = contrib["LONG"].get("adx_late_move_penalty", 0) - applied
                    if short_bonus:
                        _add("SHORT", "adx_transition", short_bonus, "ADX late LONG trend may favor SHORT reversal")
                elif adx_dom == "SHORT":
                    applied = 0 if long_transition_ok else fade_penalty
                    long_bonus = 2 if long_transition_ok else 0
                    short_points = max(0, short_points - applied)
                    if applied:
                        contrib["SHORT"]["adx_late_move_penalty"] = contrib["SHORT"].get("adx_late_move_penalty", 0) - applied
                    if long_bonus:
                        _add("LONG", "adx_transition", long_bonus, "ADX late SHORT trend may favor LONG reversal")

        try:
            last_o = float(df['open'].iloc[-1])
            last_h = float(df['high'].iloc[-1])
            last_l = float(df['low'].iloc[-1])
            last_c = float(df['close'].iloc[-1])
            last_rng = max(1e-9, last_h - last_l)
            body_frac = abs(last_c - last_o) / last_rng
            bull_close_loc = (last_c - last_l) / last_rng
            bear_close_loc = (last_h - last_c) / last_rng
            upper_wick_frac = max(0.0, last_h - max(last_o, last_c)) / last_rng
            lower_wick_frac = max(0.0, min(last_o, last_c) - last_l) / last_rng
            bull_candle_quality = float(np.clip(0.50 * body_frac + 0.35 * bull_close_loc + 0.15 * (1.0 - upper_wick_frac), 0.0, 1.0))
            bear_candle_quality = float(np.clip(0.50 * body_frac + 0.35 * bear_close_loc + 0.15 * (1.0 - lower_wick_frac), 0.0, 1.0))
            if last_c < last_o:
                bull_candle_quality *= 0.82
            if last_c > last_o:
                bear_candle_quality *= 0.82
        except Exception:
            bull_candle_quality = 0.5
            bear_candle_quality = 0.5

        scalp_phase3 = _scalp_phase3_profile_adjustment(
            adx_ctx=adx_ctx,
            long_cluster=int(long_reversal_cluster),
            short_cluster=int(short_reversal_cluster),
            long_trend_ok=bool(trend_long_ok),
            short_trend_ok=bool(trend_short_ok),
            rsi14=float(rsi14) if np.isfinite(rsi14) else None,
            last_price=float(last_price) if np.isfinite(last_price) else None,
            ref_vwap=float(ref_vwap) if ref_vwap is not None and np.isfinite(ref_vwap) else None,
            atr_last=float(atr_last) if atr_last is not None and np.isfinite(atr_last) else None,
            bullish_candle_quality=float(bull_candle_quality),
            bearish_candle_quality=float(bear_candle_quality),
        )
        long_phase3_adj = float(scalp_phase3.get("long_adj") or 0.0)
        short_phase3_adj = float(scalp_phase3.get("short_adj") or 0.0)
        if long_phase3_adj > 0:
            _add("LONG", "phase3_context", int(round(long_phase3_adj)), str(scalp_phase3.get("long_note") or "SCALP transition context"))
        elif long_phase3_adj < 0:
            long_pen = int(round(abs(long_phase3_adj)))
            long_points = max(0, long_points - long_pen)
            contrib["LONG"]["phase3_context"] = contrib["LONG"].get("phase3_context", 0) - long_pen
        if short_phase3_adj > 0:
            _add("SHORT", "phase3_context", int(round(short_phase3_adj)), str(scalp_phase3.get("short_note") or "SCALP transition context"))
        elif short_phase3_adj < 0:
            short_pen = int(round(abs(short_phase3_adj)))
            short_points = max(0, short_points - short_pen)
            contrib["SHORT"]["phase3_context"] = contrib["SHORT"].get("phase3_context", 0) - short_pen

        if not trend_long_ok and not (was_below_vwap and reclaim_vwap):
            long_points = max(0, long_points - 15)
        if not trend_short_ok and not (was_above_vwap and reject_vwap):
            short_points = max(0, short_points - 15)

    # HTF overlay scoring
    if htf_b in ("BULL", "BEAR"):
        if htf_b == "BULL":
            long_points += 10; long_reasons.append("HTF bias bullish")
            short_points = max(0, short_points - 10)
        elif htf_b == "BEAR":
            short_points += 10; short_reasons.append("HTF bias bearish")
            long_points = max(0, long_points - 10)

    # Requirements / Gatekeeping (product-safe)
    #
    # Product philosophy:
    #   - Score represents *setup quality*.
    #   - Actionability represents *tradeability* (do we have enough confirmation to plan an entry/stop/targets).
    #
    # We do this with a "confirmation score" (count of independent confirmations) and a
    # "soft-hard" volume requirement:
    #   - Volume is still required for alerting *unless* we have strong Pro confluence
    #     (sweep/OB/breaker/ORB + divergence), so we don't miss real money-makers.
    #
    # Confirmation components are boolean (0/1) and deliberately simple:
    #   confirmation_score = vwap + orb + rsi + micro_structure + volume + divergence + liquidity + fib
    #
    # NOTE: Time-of-day filters do NOT block actionability. They only affect liquidity weighting
    # (via liquidity_mult) and UI display.

    vwap_event = bool((was_below_vwap and reclaim_vwap) or (was_above_vwap and reject_vwap))
    rsi_event = bool(rsi_snap or rsi_downshift)
    macd_event = bool(macd_turn_up or macd_turn_down)
    volume_event = bool(vol_ok)
    volume_event_long = bool(vol_ok_long)
    volume_event_short = bool(vol_ok_short)

    # Micro-structure flags (used for confirmation, not direction)
    micro_hl = bool(df["low"].tail(12).iloc[-1] > df["low"].tail(12).min())
    micro_lh = bool(df["high"].tail(12).iloc[-1] < df["high"].tail(12).max())
    micro_structure_event = bool(micro_hl or micro_lh)

    is_extended_session = session in ("PREMARKET", "AFTERHOURS")

    # Pro structural trigger (if enabled)
    pro_trigger = False
    divergence_event = False
    if pro_mode:
        divergence_event = bool(isinstance(rsi_div, dict) and rsi_div.get("type") in ("bull", "bear"))
        pro_trigger = bool(
            bull_sweep or bear_sweep
            or bull_ob_retest or bear_ob_retest
            or bull_breaker_retest or bear_breaker_retest
            or orb_bull or orb_bear
            or divergence_event
        )
    extras["pro_trigger"] = bool(pro_trigger)

    # Strong Pro confluence: 2+ independent Pro triggers (plus divergence counts as a trigger)
    # This is the override that can allow alerts even without the simplistic volume flag.
    pro_triggers_count = 0
    if pro_mode:
        pro_triggers_count += 1 if (bull_sweep or bear_sweep) else 0
        pro_triggers_count += 1 if (bull_ob_retest or bear_ob_retest) else 0
        pro_triggers_count += 1 if (bull_breaker_retest or bear_breaker_retest) else 0
        pro_triggers_count += 1 if (orb_bull or orb_bear) else 0
        pro_triggers_count += 1 if divergence_event else 0
    strong_pro_confluence = bool(pro_mode and pro_triggers_count >= 2)

    # Confirmation score (directional)
    vwap_event_long = bool(was_below_vwap and reclaim_vwap)
    vwap_event_short = bool(was_above_vwap and reject_vwap)
    orb_event_long = bool(orb_bull)
    orb_event_short = bool(orb_bear)
    orb_event = bool(orb_event_long or orb_event_short)
    liquidity_event_long = bool(bull_sweep or bull_ob_retest or bull_breaker_retest)
    liquidity_event_short = bool(bear_sweep or bear_ob_retest or bear_breaker_retest)
    liquidity_event = bool(liquidity_event_long or liquidity_event_short)
    fib_event_long = bool(fib_near_long)
    fib_event_short = bool(fib_near_short)
    fib_event = bool(fib_event_long or fib_event_short)

    divergence_type = None
    if isinstance(rsi_div, dict):
        divergence_type = str(rsi_div.get("type") or "").lower().strip()

    pro_trigger_long = bool(pro_mode and (bull_sweep or bull_ob_retest or bull_breaker_retest or orb_bull or divergence_type == "bull"))
    pro_trigger_short = bool(pro_mode and (bear_sweep or bear_ob_retest or bear_breaker_retest or orb_bear or divergence_type == "bear"))

    long_confirmation_components = {
        "vwap": int(bool(was_below_vwap and reclaim_vwap)),
        "orb": int(bool(orb_bull)),
        "rsi": int(bool(rsi_snap)),
        "micro_structure": int(bool(micro_hl)),
        "volume": int(volume_event_long),
        "divergence": int(bool(divergence_type == "bull")),
        "liquidity": int(bool(bull_sweep or bull_ob_retest or bull_breaker_retest)),
        "fib": int(bool(fib_near_long)),
    }
    short_confirmation_components = {
        "vwap": int(bool(was_above_vwap and reject_vwap)),
        "orb": int(bool(orb_bear)),
        "rsi": int(bool(rsi_downshift)),
        "micro_structure": int(bool(micro_lh)),
        "volume": int(volume_event_short),
        "divergence": int(bool(divergence_type == "bear")),
        "liquidity": int(bool(bear_sweep or bear_ob_retest or bear_breaker_retest)),
        "fib": int(bool(fib_near_short)),
    }
    try:
        long_confirmation_components["pressure"] = int(float((pressure_states or {}).get("long_pressure_score") or 0.0) >= 2.0)
        short_confirmation_components["pressure"] = int(float((pressure_states or {}).get("short_pressure_score") or 0.0) >= 2.0)
    except Exception:
        pass
    long_confirmation_score = int(sum(long_confirmation_components.values()))
    short_confirmation_score = int(sum(short_confirmation_components.values()))

    # Preserve the legacy generic fields for downstream workflow stability, but make them reflect
    # the stronger directional side rather than a mixed long+short basket of confirmations.
    provisional_confirmation_bias = "LONG" if long_points >= short_points else "SHORT"
    if long_confirmation_score > short_confirmation_score:
        provisional_confirmation_bias = "LONG"
    elif short_confirmation_score > long_confirmation_score:
        provisional_confirmation_bias = "SHORT"

    confirmation_components = long_confirmation_components if provisional_confirmation_bias == "LONG" else short_confirmation_components
    confirmation_score = int(max(long_confirmation_score, short_confirmation_score))
    extras["confirmation_components"] = confirmation_components
    extras["confirmation_score"] = confirmation_score
    extras["confirmation_score_long"] = int(long_confirmation_score)
    extras["confirmation_score_short"] = int(short_confirmation_score)
    extras["provisional_confirmation_bias"] = provisional_confirmation_bias
    extras["confirmation_bias"] = provisional_confirmation_bias
    extras["confirmation_components_long"] = long_confirmation_components
    extras["confirmation_components_short"] = short_confirmation_components
    extras["strong_pro_confluence"] = bool(strong_pro_confluence)

    # Preserve gate diagnostics (used in UI/why strings)
    extras["gates"] = {
        "vwap_event": vwap_event,
        "rsi_event": rsi_event,
        "macd_event": macd_event,
        "volume_event": volume_event,
        "volume_event_long": volume_event_long,
        "volume_event_short": volume_event_short,
        "extended_session": bool(is_extended_session),
        "confirmation_score": confirmation_score,
        "confirmation_score_long": int(long_confirmation_score),
        "confirmation_score_short": int(short_confirmation_score),
        "confirmation_bias": provisional_confirmation_bias,
        "strong_pro_confluence": bool(strong_pro_confluence),
    }

    # Confirm threshold: require multiple independent confirmations before we emit entry/TP or alert.
    # Pro mode gets a slightly lower threshold because we have more independent features.
    confirm_threshold = 4 if not pro_mode else 3
    extras["confirm_threshold"] = int(confirm_threshold)

    # Session preference policy for volatile names:
    # session toggles should matter for alert formation, not just liquidity weighting.
    # We keep workflow intact by using a softer decision-layer approach:
    #   - raise the effective confirmation bar when outside the preferred window
    #   - cap most off-window CONFIRMED setups to PRE unless quality is exceptional
    #   - apply a modest score penalty to the chosen side later in the flow
    session_window_penalty_map = {
        "OPENING": 5,
        "MIDDAY": 4,
        "POWER": 5,
        "PREMARKET": 7,
        "AFTERHOURS": 7,
    }
    session_confirm_bump_map = {
        "OPENING": 1,
        "MIDDAY": 1,
        "POWER": 1,
        "PREMARKET": 2,
        "AFTERHOURS": 2,
    }
    session_window_penalty = int(session_window_penalty_map.get(str(session), 4)) if not bool(allowed) else 0
    session_confirm_bump = int(session_confirm_bump_map.get(str(session), 1)) if not bool(allowed) else 0
    effective_confirm_threshold = int(confirm_threshold + session_confirm_bump)
    extras["session_window_penalty"] = int(session_window_penalty)
    extras["session_confirm_bump"] = int(session_confirm_bump)
    extras["effective_confirm_threshold"] = int(effective_confirm_threshold)
    extras["session_penalty_applied"] = 0
    extras["exceptional_off_window_quality"] = False

    # PRE vs CONFIRMED stages
    # ----------------------
    # Goal: fire *earlier* (pre-trigger) alerts when a high-quality setup is forming,
    # without removing the confirmed (fully gated) alert. We do this by allowing a
    # PRE stage when price is approaching the planned trigger (usually VWAP) with
    # supportive momentum/structure, but before the reclaim/rejection event prints.
    #
    # Stages are stored in extras["stage"]:
    #   - "PRE"        : forming setup, provides an entry/stop/TP plan
    #   - "CONFIRMED"  : classic gated setup (confirm_threshold met + hard gates)
    stage: str | None = None
    stage_note: str = ""

    # Trigger-proximity used for PRE alerts
    # -------------------------------
    # PRE alerts should be *trigger proximity* driven (distance to the trigger line, normalized by ATR),
    # not only score thresholds or "actionable transition".
    #
    # Today the most common trigger line is VWAP (session or cumulative). If VWAP is unavailable (NaN)
    # we still allow PRE when Pro structural trigger exists, but proximity math is skipped.
    prox_atr = None
    prox_abs = None
    try:
        prox_abs = max(0.35 * float(atr_last or 0.0), 0.0008 * float(last_price or 0.0))
    except Exception:
        prox_abs = None

    trigger_near = False
    dist = None
    if isinstance(ref_vwap, (float, int)) and isinstance(last_price, (float, int)) and isinstance(prox_abs, (float, int)) and prox_abs > 0:
        dist = abs(float(last_price) - float(ref_vwap))
        trigger_near = bool(dist <= float(prox_abs))
        try:
            if atr_last and float(atr_last) > 0:
                prox_atr = float(dist) / float(atr_last)
        except Exception:
            prox_atr = None

    extras["trigger_proximity_atr"] = prox_atr
    extras["trigger_proximity_abs"] = float(prox_abs) if isinstance(prox_abs, (float, int)) else None
    extras["trigger_near"] = bool(trigger_near)

    # Momentum/structure "pre" hints
    macd_pre_long = bool(_is_rising(df["macd_hist"], 3))
    macd_pre_short = bool(_is_falling(df["macd_hist"], 3))
    struct_pre_long = bool(micro_hl)
    struct_pre_short = bool(micro_lh)
    try:
        rsi5_last = float(df["rsi5"].iloc[-1])
    except Exception:
        rsi5_last = float("nan")
    # Define preliminary RSI pressure before the trader-edge pre-reclaim lane references it.
    # The richer adaptive-RSI block below can still refine/overwrite these aliases.
    try:
        rsi_pre_long = bool(_is_rising(df["rsi5"], 3) or bool((pressure_states or {}).get("rsi_pressure_long")))
        rsi_pre_short = bool(_is_falling(df["rsi5"], 3) or bool((pressure_states or {}).get("rsi_pressure_short")))
    except Exception:
        rsi_pre_long = False
        rsi_pre_short = False

    # Small early-reversal PRE pathway:
    # allow an approaching / attempted reclaim to surface as PRE when momentum is already
    # improving and local structure has stopped cleanly trending against the trade, without
    # requiring the full reclaim-hold event yet.
    pre_trigger_near = bool(trigger_near)
    try:
        if not pre_trigger_near and isinstance(dist, (float, int)) and isinstance(prox_abs, (float, int)) and np.isfinite(dist) and np.isfinite(prox_abs):
            pre_trigger_near = bool(float(dist) <= max(1.35 * float(prox_abs), 0.55 * float(atr_last or 0.0)))
    except Exception:
        pass
    try:
        reclaim_attempt_long = bool(
            isinstance(ref_vwap, (float, int))
            and float(last_price) < float(ref_vwap)
            and (float(df["high"].tail(3).max()) >= float(ref_vwap) - 0.10 * float(buffer))
        )
        reclaim_attempt_short = bool(
            isinstance(ref_vwap, (float, int))
            and float(last_price) > float(ref_vwap)
            and (float(df["low"].tail(3).min()) <= float(ref_vwap) + 0.10 * float(buffer))
        )
    except Exception:
        reclaim_attempt_long = False
        reclaim_attempt_short = False
    early_pre_long = bool(
        isinstance(ref_vwap, (float, int))
        and isinstance(last_price, (float, int))
        and float(last_price) < float(ref_vwap)
        and pre_trigger_near
        and reclaim_attempt_long
        and ((macd_build_long_pts >= 4) or macd_pre_long or macd_event or bool((pressure_states or {}).get("macd_pressure_long")))
        and (struct_pre_long or liquidity_event_long or orb_event_long or bool((pressure_states or {}).get("volume_absorption_long")) or bool((pressure_states or {}).get("di_transfer_long")))
        and (long_confirmation_score >= max(2, effective_confirm_threshold - 2))
    )
    early_pre_short = bool(
        isinstance(ref_vwap, (float, int))
        and isinstance(last_price, (float, int))
        and float(last_price) > float(ref_vwap)
        and pre_trigger_near
        and reclaim_attempt_short
        and ((macd_build_short_pts >= 4) or macd_pre_short or macd_event or bool((pressure_states or {}).get("macd_pressure_short")))
        and (struct_pre_short or liquidity_event_short or orb_event_short or bool((pressure_states or {}).get("volume_absorption_short")) or bool((pressure_states or {}).get("di_transfer_short")))
        and (short_confirmation_score >= max(2, effective_confirm_threshold - 2))
    )

    try:
        _ext_from_ref = abs(float(last_price) - float(ref_vwap)) / max(1e-9, float(atr_last or 0.0)) if isinstance(ref_vwap, (float, int)) and np.isfinite(float(atr_last or 0.0)) and float(atr_last or 0.0) > 0 else 0.0
    except Exception:
        _ext_from_ref = 0.0

    # Trader-edge pre-reclaim/pre-reject lane: let clean attempts surface before full VWAP consensus.
    try:
        _vwap_dist_atr = abs(float(last_price) - float(ref_vwap)) / max(1e-9, float(atr_last or 0.0)) if isinstance(ref_vwap, (float, int)) else 99.0
        _recent_lows_te = pd.to_numeric(df["low"].tail(3), errors="coerce").dropna()
        _recent_highs_te = pd.to_numeric(df["high"].tail(3), errors="coerce").dropna()
        _hl_forming_te = bool(len(_recent_lows_te) >= 3 and float(_recent_lows_te.iloc[-1]) >= float(_recent_lows_te.iloc[-2]) - 0.08 * float(atr_last) and float(_recent_lows_te.iloc[-2]) >= float(_recent_lows_te.iloc[-3]) - 0.10 * float(atr_last))
        _lh_forming_te = bool(len(_recent_highs_te) >= 3 and float(_recent_highs_te.iloc[-1]) <= float(_recent_highs_te.iloc[-2]) + 0.08 * float(atr_last) and float(_recent_highs_te.iloc[-2]) <= float(_recent_highs_te.iloc[-3]) + 0.10 * float(atr_last))
        trader_edge_pre_long = bool(
            isinstance(ref_vwap, (float, int)) and float(last_price) <= float(ref_vwap) + 0.04 * float(atr_last)
            and _vwap_dist_atr <= 0.42 and (reclaim_attempt_long or trigger_near or pre_trigger_near)
            and (_hl_forming_te or struct_pre_long or liquidity_event_long or orb_event_long)
            and (rsi_pre_long or macd_pre_long or macd_build_long_pts >= 3 or bool((pressure_states or {}).get("rsi_pressure_long")) or bool((pressure_states or {}).get("macd_pressure_long")))
            and not (vol_ok_short and not vol_ok_long and short_confirmation_score > long_confirmation_score + 1)
            and _ext_from_ref <= 0.92
        )
        trader_edge_pre_short = bool(
            isinstance(ref_vwap, (float, int)) and float(last_price) >= float(ref_vwap) - 0.04 * float(atr_last)
            and _vwap_dist_atr <= 0.42 and (reclaim_attempt_short or trigger_near or pre_trigger_near)
            and (_lh_forming_te or struct_pre_short or liquidity_event_short or orb_event_short)
            and (rsi_pre_short or macd_pre_short or macd_build_short_pts >= 3 or bool((pressure_states or {}).get("rsi_pressure_short")) or bool((pressure_states or {}).get("macd_pressure_short")))
            and not (vol_ok_long and not vol_ok_short and long_confirmation_score > short_confirmation_score + 1)
            and _ext_from_ref <= 0.92
        )
    except Exception:
        trader_edge_pre_long = False
        trader_edge_pre_short = False

    try:
        _rc_atr = max(1e-9, float(atr_last or 0.0))
        _rc_lows = pd.to_numeric(df["low"].tail(int(min(8, len(df)))), errors="coerce").dropna()
        _rc_highs = pd.to_numeric(df["high"].tail(int(min(8, len(df)))), errors="coerce").dropna()
        _rc_closes = pd.to_numeric(df["close"].tail(int(min(8, len(df)))), errors="coerce").dropna()
        _l3, _h3, _c3 = _rc_lows.tail(3), _rc_highs.tail(3), _rc_closes.tail(3)
        _rc_hl = bool(len(_l3) >= 3 and len(_c3) >= 3 and float(_l3.iloc[-1]) >= float(_l3.iloc[0]) - 0.10 * _rc_atr and float(_c3.iloc[-1]) >= float(_c3.median()))
        _rc_lh = bool(len(_h3) >= 3 and len(_c3) >= 3 and float(_h3.iloc[-1]) <= float(_h3.iloc[0]) + 0.10 * _rc_atr and float(_c3.iloc[-1]) <= float(_c3.median()))
        _rc_near_ref = bool(isinstance(ref_vwap, (float, int)) and abs(float(last_price) - float(ref_vwap)) <= 1.05 * _rc_atr)
        _rc_press_long = bool(bool((pressure_states or {}).get("fast_trigger_pressure_long")) or bool((pressure_states or {}).get("rsi_pressure_long")) or bool((pressure_states or {}).get("macd_pressure_long")) or bool((pressure_states or {}).get("di_transfer_long")) or bool((pressure_states or {}).get("volume_absorption_long")))
        _rc_press_short = bool(bool((pressure_states or {}).get("fast_trigger_pressure_short")) or bool((pressure_states or {}).get("rsi_pressure_short")) or bool((pressure_states or {}).get("macd_pressure_short")) or bool((pressure_states or {}).get("di_transfer_short")) or bool((pressure_states or {}).get("volume_absorption_short")))
        scalp_reclaim_cont_long = bool(_rc_near_ref and _rc_hl and (reclaim_attempt_long or reclaim_vwap or struct_pre_long or micro_hl_pre) and _rc_press_long and ((macd_build_long_pts >= 3) or bool((pressure_states or {}).get("macd_pressure_long")) or macd_pre_long) and not (vol_ok_short and not vol_ok_long and short_confirmation_score > long_confirmation_score + 1) and _ext_from_ref <= 1.18)
        scalp_reclaim_cont_short = bool(_rc_near_ref and _rc_lh and (reclaim_attempt_short or reject_vwap or struct_pre_short or micro_lh_pre) and _rc_press_short and ((macd_build_short_pts >= 3) or bool((pressure_states or {}).get("macd_pressure_short")) or macd_pre_short) and not (vol_ok_long and not vol_ok_short and long_confirmation_score > short_confirmation_score + 1) and _ext_from_ref <= 1.18)
    except Exception:
        scalp_reclaim_cont_long = False
        scalp_reclaim_cont_short = False

    # SCALP Reversal Trigger Layer:
    # This is the structure-first trigger that makes SCALP act like an inflection system
    # instead of waiting for full VWAP/MACD/RSI confirmation. It is deliberately
    # additive: it can promote a PRE setup and improve entry timing, but it does not
    # change downstream payload contracts or bypass stops/TP geometry.
    try:
        scalp_rev_long = _detect_scalp_reversal_trigger(df, direction="LONG", atr_last=float(atr_last or 0.0), ref_level=ref_vwap)
        scalp_rev_short = _detect_scalp_reversal_trigger(df, direction="SHORT", atr_last=float(atr_last or 0.0), ref_level=ref_vwap)
        scalp_reversal_trigger_long = bool(scalp_rev_long.get("trigger") and not (vol_ok_short and not vol_ok_long and short_confirmation_score > long_confirmation_score + 2))
        scalp_reversal_trigger_short = bool(scalp_rev_short.get("trigger") and not (vol_ok_long and not vol_ok_short and long_confirmation_score > short_confirmation_score + 2))
        # If both sides somehow trigger in chop, keep only the stronger trigger.
        if scalp_reversal_trigger_long and scalp_reversal_trigger_short:
            if float(scalp_rev_long.get("score") or 0.0) >= float(scalp_rev_short.get("score") or 0.0) + 0.08:
                scalp_reversal_trigger_short = False
            elif float(scalp_rev_short.get("score") or 0.0) >= float(scalp_rev_long.get("score") or 0.0) + 0.08:
                scalp_reversal_trigger_long = False
            else:
                scalp_reversal_trigger_long = False
                scalp_reversal_trigger_short = False
    except Exception:
        scalp_rev_long = {"trigger": False}
        scalp_rev_short = {"trigger": False}
        scalp_reversal_trigger_long = False
        scalp_reversal_trigger_short = False

    extras["scalp_reversal_trigger_long"] = bool(scalp_reversal_trigger_long)
    extras["scalp_reversal_trigger_short"] = bool(scalp_reversal_trigger_short)
    extras["scalp_reversal_trigger_type_long"] = scalp_rev_long.get("trigger_type") if isinstance(scalp_rev_long, dict) else None
    extras["scalp_reversal_trigger_type_short"] = scalp_rev_short.get("trigger_type") if isinstance(scalp_rev_short, dict) else None
    extras["scalp_reversal_trigger_score_long"] = float(scalp_rev_long.get("score") or 0.0) if isinstance(scalp_rev_long, dict) else 0.0
    extras["scalp_reversal_trigger_score_short"] = float(scalp_rev_short.get("score") or 0.0) if isinstance(scalp_rev_short, dict) else 0.0
    extras["scalp_reversal_trigger_points_long"] = float(scalp_rev_long.get("trigger_points") or 0.0) if isinstance(scalp_rev_long, dict) else 0.0
    extras["scalp_reversal_trigger_points_short"] = float(scalp_rev_short.get("trigger_points") or 0.0) if isinstance(scalp_rev_short, dict) else 0.0
    extras["scalp_reversal_weighted_points_long"] = float(scalp_rev_long.get("weighted_points") or 0.0) if isinstance(scalp_rev_long, dict) else 0.0
    extras["scalp_reversal_weighted_points_short"] = float(scalp_rev_short.get("weighted_points") or 0.0) if isinstance(scalp_rev_short, dict) else 0.0
    extras["scalp_reversal_trigger_entry_anchor_long"] = scalp_rev_long.get("entry_anchor") if isinstance(scalp_rev_long, dict) else None
    extras["scalp_reversal_trigger_entry_anchor_short"] = scalp_rev_short.get("entry_anchor") if isinstance(scalp_rev_short, dict) else None
    extras["scalp_reversal_trigger_structure_ref_long"] = scalp_rev_long.get("structure_ref") if isinstance(scalp_rev_long, dict) else None
    extras["scalp_reversal_trigger_structure_ref_short"] = scalp_rev_short.get("structure_ref") if isinstance(scalp_rev_short, dict) else None

    early_pre_long = bool(early_pre_long or trader_edge_pre_long or scalp_reclaim_cont_long or scalp_reversal_trigger_long)
    early_pre_short = bool(early_pre_short or trader_edge_pre_short or scalp_reclaim_cont_short or scalp_reversal_trigger_short)
    if scalp_reversal_trigger_long:
        long_points += 4
        long_confirmation_score = max(int(long_confirmation_score), 2)
        long_reasons.append(f"SCALP reversal trigger {extras.get('scalp_reversal_trigger_type_long') or ''}".strip())
    if scalp_reversal_trigger_short:
        short_points += 4
        short_confirmation_score = max(int(short_confirmation_score), 2)
        short_reasons.append(f"SCALP reversal trigger {extras.get('scalp_reversal_trigger_type_short') or ''}".strip())

    # Controlled early-ignition override for volatile $1-$5 names.
    # Goal: surface the first real move earlier without replacing the broader confirmation framework.
    try:
        _ext_from_ref = abs(float(last_price) - float(ref_vwap)) / max(1e-9, float(atr_last or 0.0)) if isinstance(ref_vwap, (float, int)) and np.isfinite(float(atr_last or 0.0)) and float(atr_last or 0.0) > 0 else 0.0
    except Exception:
        _ext_from_ref = 0.0
    adx_regime_local = str(adx_ctx.get("regime") or "unknown")
    adx_dom_local = str(adx_ctx.get("dominant_side") or "").upper()
    adx_slope_local = float(adx_ctx.get("adx_slope") or 0.0)
    adx_spread_local = float(adx_ctx.get("di_spread") or 0.0)
    # These clusters are usually computed in the ADX scoring block, but certain
    # lightweight/non-pro paths can reach early ignition without that block having
    # assigned them. Keep the ignition lane safe and conservative in that case.
    long_reversal_cluster = int(locals().get("long_reversal_cluster", 0) or 0)
    short_reversal_cluster = int(locals().get("short_reversal_cluster", 0) or 0)

    ignition_long_struct = bool(reclaim_vwap or reclaim_attempt_long or orb_bull or bull_sweep or bull_breaker_retest or bull_ob_retest or micro_hl_pre)
    ignition_short_struct = bool(reject_vwap or reclaim_attempt_short or orb_bear or bear_sweep or bear_breaker_retest or bear_ob_retest or micro_lh_pre)
    ignition_long_adx = bool(
        (adx_regime_local in ("emerging", "strengthening") and adx_slope_local >= 0.20 and adx_spread_local >= 3.5 and adx_dom_local in ("LONG", ""))
        or (adx_dom_local == "SHORT" and adx_regime_local in ("mature_trend", "exhausting") and long_reversal_cluster >= 3 and adx_slope_local >= -0.35)
    )
    ignition_short_adx = bool(
        (adx_regime_local in ("emerging", "strengthening") and adx_slope_local >= 0.20 and adx_spread_local >= 3.5 and adx_dom_local in ("SHORT", ""))
        or (adx_dom_local == "LONG" and adx_regime_local in ("mature_trend", "exhausting") and short_reversal_cluster >= 3 and adx_slope_local >= -0.35)
    )

    # Tiered ignition confidence: distinguish ultra-clean first-leg ignition from merely decent ignition.
    # This lets SCALP surface the best explosive reversals earlier without loosening the engine broadly.
    try:
        recent_ign = df.tail(int(min(4, len(df))))
        ign_close = pd.to_numeric(recent_ign.get("close"), errors="coerce") if len(recent_ign) else pd.Series(dtype=float)
        ign_open = pd.to_numeric(recent_ign.get("open"), errors="coerce") if len(recent_ign) else pd.Series(dtype=float)
        ign_vol = pd.to_numeric(recent_ign.get("volume"), errors="coerce") if len(recent_ign) and "volume" in recent_ign else pd.Series(dtype=float)
        momentum_cont_long = bool(len(ign_close) >= 3 and float(ign_close.iloc[-1]) > float(ign_close.iloc[-2]) > float(ign_close.iloc[-3]))
        momentum_cont_short = bool(len(ign_close) >= 3 and float(ign_close.iloc[-1]) < float(ign_close.iloc[-2]) < float(ign_close.iloc[-3]))
        if len(ign_open) >= 3 and len(ign_close) >= 3:
            momentum_cont_long = bool(momentum_cont_long and sum(float(ign_close.iloc[-i]) > float(ign_open.iloc[-i]) for i in (1, 2)) >= 1)
            momentum_cont_short = bool(momentum_cont_short and sum(float(ign_close.iloc[-i]) < float(ign_open.iloc[-i]) for i in (1, 2)) >= 1)
        if len(ign_vol) >= 3 and ign_vol.notna().sum() >= 3:
            v1 = float(ign_vol.iloc[-1]); v2 = float(ign_vol.iloc[-2]); v3 = float(ign_vol.iloc[-3])
            momentum_cont_long = bool(momentum_cont_long and (v1 >= 0.90 * v2 or v1 >= 1.05 * v3))
            momentum_cont_short = bool(momentum_cont_short and (v1 >= 0.90 * v2 or v1 >= 1.05 * v3))
    except Exception:
        momentum_cont_long = False
        momentum_cont_short = False

    ignition_long_strength = 0
    ignition_short_strength = 0
    ignition_long_strength += 1 if ignition_long_struct else 0
    ignition_long_strength += 1 if bull_displacement else 0
    ignition_long_strength += 1 if vol_ok_long else 0
    ignition_long_strength += 1 if bull_ignition_candle_quality >= 0.64 else 0
    ignition_long_strength += 1 if ((macd_build_long_pts >= 3) or macd_pre_long or macd_event or bool((pressure_states or {}).get("macd_pressure_long"))) else 0
    ignition_long_strength += 1 if (ignition_long_adx or bool((pressure_states or {}).get("di_transfer_long"))) else 0
    ignition_long_strength += 1 if (bool((pressure_states or {}).get("volume_absorption_long")) or bool((pressure_states or {}).get("directional_expansion_long"))) else 0
    ignition_long_strength += 1 if (long_confirmation_score >= max(1, effective_confirm_threshold - 2)) else 0
    ignition_long_strength += 1 if momentum_cont_long else 0

    ignition_short_strength += 1 if ignition_short_struct else 0
    ignition_short_strength += 1 if bear_displacement else 0
    ignition_short_strength += 1 if vol_ok_short else 0
    ignition_short_strength += 1 if bear_ignition_candle_quality >= 0.64 else 0
    ignition_short_strength += 1 if ((macd_build_short_pts >= 3) or macd_pre_short or macd_event or bool((pressure_states or {}).get("macd_pressure_short"))) else 0
    ignition_short_strength += 1 if (ignition_short_adx or bool((pressure_states or {}).get("di_transfer_short"))) else 0
    ignition_short_strength += 1 if (bool((pressure_states or {}).get("volume_absorption_short")) or bool((pressure_states or {}).get("directional_expansion_short"))) else 0
    ignition_short_strength += 1 if (short_confirmation_score >= max(1, effective_confirm_threshold - 2)) else 0
    ignition_short_strength += 1 if momentum_cont_short else 0

    ignition_tier_long = 0
    if ignition_long_strength >= 7 and bull_ignition_candle_quality >= 0.72 and _ext_from_ref <= 1.00 and momentum_cont_long:
        ignition_tier_long = 3
    elif ignition_long_strength >= 5 and bull_ignition_candle_quality >= 0.64 and _ext_from_ref <= 1.12:
        ignition_tier_long = 2
    elif ignition_long_strength >= 4 and bull_ignition_candle_quality >= 0.60 and _ext_from_ref <= 0.95 and momentum_cont_long:
        ignition_tier_long = 1

    ignition_tier_short = 0
    if ignition_short_strength >= 7 and bear_ignition_candle_quality >= 0.72 and _ext_from_ref <= 1.00 and momentum_cont_short:
        ignition_tier_short = 3
    elif ignition_short_strength >= 5 and bear_ignition_candle_quality >= 0.64 and _ext_from_ref <= 1.12:
        ignition_tier_short = 2
    elif ignition_short_strength >= 4 and bear_ignition_candle_quality >= 0.60 and _ext_from_ref <= 0.95 and momentum_cont_short:
        ignition_tier_short = 1

    ignition_override_long = bool(ignition_tier_long >= 2)
    ignition_override_short = bool(ignition_tier_short >= 2)
    if ignition_override_long and ignition_override_short:
        if bull_ignition_candle_quality >= bear_ignition_candle_quality + 0.05:
            ignition_override_short = False
        elif bear_ignition_candle_quality >= bull_ignition_candle_quality + 0.05:
            ignition_override_long = False
        else:
            ignition_override_long = False
            ignition_override_short = False
    extras["ignition_override_long"] = bool(ignition_override_long)
    extras["ignition_override_short"] = bool(ignition_override_short)
    extras["ignition_ext_from_ref_atr"] = float(_ext_from_ref)
    extras["ignition_tier_long"] = int(ignition_tier_long)
    extras["ignition_tier_short"] = int(ignition_tier_short)
    extras["ignition_strength_long"] = int(ignition_long_strength)
    extras["ignition_strength_short"] = int(ignition_short_strength)
    extras["momentum_continuity_long"] = bool(momentum_cont_long)
    extras["momentum_continuity_short"] = bool(momentum_cont_short)

    if ignition_override_long:
        _add("LONG", "ignition_override", 4 if ignition_tier_long >= 3 else 3, f"Early ignition override T{ignition_tier_long}")
    elif ignition_tier_long == 1:
        _add("LONG", "ignition_watch", 1, "Developing ignition")
    if ignition_override_short:
        _add("SHORT", "ignition_override", 4 if ignition_tier_short >= 3 else 3, f"Early ignition override T{ignition_tier_short}")
    elif ignition_tier_short == 1:
        _add("SHORT", "ignition_watch", 1, "Developing ignition")

    # Adaptive RSI context for volatile names:
    # keep classic early-zone behavior when RSI-5 is still modest, but allow stronger setups
    # to remain valid in the 60-72 (long) / 28-40 (short) band and only treat true exhaustion as hostile.
    bullish_rsi_ctx = bool((struct_pre_long or liquidity_event_long or orb_event_long or reclaim_attempt_long or vwap_event_long) and ((macd_build_long_pts >= 3) or macd_pre_long or macd_event))
    bearish_rsi_ctx = bool((struct_pre_short or liquidity_event_short or orb_event_short or reclaim_attempt_short or vwap_event_short) and ((macd_build_short_pts >= 3) or macd_pre_short or macd_event))
    strong_bullish_rsi_ctx = bool(pro_trigger_long or vwap_event_long or early_pre_long or (pre_trigger_near and reclaim_attempt_long and macd_event and (struct_pre_long or liquidity_event_long or orb_event_long)))
    strong_bearish_rsi_ctx = bool(pro_trigger_short or vwap_event_short or early_pre_short or (pre_trigger_near and reclaim_attempt_short and macd_event and (struct_pre_short or liquidity_event_short or orb_event_short)))
    rsi_pre_long = bool(
        _is_rising(df["rsi5"], 3)
        and np.isfinite(rsi5_last)
        and (
            rsi5_last < 60
            or (rsi5_last < 72 and bullish_rsi_ctx)
            or (rsi5_last < 75 and strong_bullish_rsi_ctx)
        )
    )
    rsi_pre_short = bool(
        _is_falling(df["rsi5"], 3)
        and np.isfinite(rsi5_last)
        and (
            rsi5_last > 40
            or (rsi5_last > 28 and bearish_rsi_ctx)
            or (rsi5_last > 25 and strong_bearish_rsi_ctx)
        )
    )
    adaptive_rsi_confirm_long = bool(
        np.isfinite(rsi5_last)
        and (
            (rsi5_last < 72 and bullish_rsi_ctx and (struct_pre_long or reclaim_attempt_long or pre_trigger_near))
            or (rsi5_last < 75 and strong_bullish_rsi_ctx and ((macd_build_long_pts >= 3) or macd_event))
        )
    )
    adaptive_rsi_confirm_short = bool(
        np.isfinite(rsi5_last)
        and (
            (rsi5_last > 28 and bearish_rsi_ctx and (struct_pre_short or reclaim_attempt_short or pre_trigger_near))
            or (rsi5_last > 25 and strong_bearish_rsi_ctx and ((macd_build_short_pts >= 3) or macd_event))
        )
    )
    extras["adaptive_rsi_pre_long"] = bool(rsi_pre_long and not rsi_event and np.isfinite(rsi5_last) and rsi5_last >= 60)
    extras["adaptive_rsi_pre_short"] = bool(rsi_pre_short and not rsi_event and np.isfinite(rsi5_last) and rsi5_last <= 40)
    extras["adaptive_rsi_confirm_long"] = bool(adaptive_rsi_confirm_long and not rsi_event)
    extras["adaptive_rsi_confirm_short"] = bool(adaptive_rsi_confirm_short and not rsi_event)
    extras["rsi5_last"] = float(rsi5_last) if np.isfinite(rsi5_last) else None
    extras["pre_trigger_near"] = bool(pre_trigger_near)
    extras["reclaim_attempt_long"] = bool(reclaim_attempt_long)
    extras["reclaim_attempt_short"] = bool(reclaim_attempt_short)
    extras["early_pre_long"] = bool(early_pre_long)
    extras["early_pre_short"] = bool(early_pre_short)
    extras["trader_edge_pre_long"] = bool(trader_edge_pre_long)
    extras["trader_edge_pre_short"] = bool(trader_edge_pre_short)
    extras["scalp_reclaim_continuation_long"] = bool(locals().get("scalp_reclaim_cont_long", False))
    extras["scalp_reclaim_continuation_short"] = bool(locals().get("scalp_reclaim_cont_short", False))
    if trader_edge_pre_long:
        long_points += 2
        long_reasons.append("Trader-edge pre-reclaim")
    if trader_edge_pre_short:
        short_points += 2
        short_reasons.append("Trader-edge pre-reject")
    if bool(locals().get("scalp_reclaim_cont_long", False)):
        long_points += 3
        long_reasons.append("Reclaim-continuation bridge")
    if bool(locals().get("scalp_reclaim_cont_short", False)):
        short_points += 3
        short_reasons.append("Reclaim-continuation bridge")

    # Acceptance progression (SCALP): a small capped energy-build bonus that helps PRE
    # setups surface a bit earlier when momentum is improving and price is starting
    # to interact with reclaim territory. This stays subordinate to the core logic.
    scalp_accept_progress_long_bonus = 0
    scalp_accept_progress_short_bonus = 0
    try:
        hist_tail = pd.to_numeric(df["macd_hist"].tail(int(min(4, len(df)))), errors="coerce").dropna()
        hist_last = float(hist_tail.iloc[-1]) if len(hist_tail) else float("nan")
        recent_abs = float(hist_tail.abs().tail(int(min(12, len(hist_tail)))).median()) if len(hist_tail) else float("nan")
        near_zero_band = max(0.0001, 0.60 * recent_abs) if np.isfinite(recent_abs) and recent_abs > 0 else 0.02
        hist_near_zero = bool(np.isfinite(hist_last) and abs(hist_last) <= near_zero_band)
        hist_green_build = bool(len(hist_tail) >= 3 and hist_tail.iloc[-1] > hist_tail.iloc[-2] > hist_tail.iloc[-3] and hist_tail.iloc[-1] > 0)
        hist_red_build = bool(len(hist_tail) >= 3 and hist_tail.iloc[-1] < hist_tail.iloc[-2] < hist_tail.iloc[-3] and hist_tail.iloc[-1] < 0)
        if early_pre_long:
            scalp_accept_progress_long_bonus += 2
            if hist_near_zero or hist_last > 0:
                scalp_accept_progress_long_bonus += 1
            if hist_green_build:
                scalp_accept_progress_long_bonus += 1
            if rsi_pre_long:
                scalp_accept_progress_long_bonus += 1
        if early_pre_short:
            scalp_accept_progress_short_bonus += 2
            if hist_near_zero or hist_last < 0:
                scalp_accept_progress_short_bonus += 1
            if hist_red_build:
                scalp_accept_progress_short_bonus += 1
            if rsi_pre_short:
                scalp_accept_progress_short_bonus += 1
    except Exception:
        pass
    scalp_accept_progress_long_bonus = int(min(4, max(0, scalp_accept_progress_long_bonus)))
    scalp_accept_progress_short_bonus = int(min(4, max(0, scalp_accept_progress_short_bonus)))
    if scalp_accept_progress_long_bonus:
        long_points += scalp_accept_progress_long_bonus
        long_reasons.append(f"Acceptance build +{scalp_accept_progress_long_bonus}")
    if scalp_accept_progress_short_bonus:
        short_points += scalp_accept_progress_short_bonus
        short_reasons.append(f"Acceptance build +{scalp_accept_progress_short_bonus}")
    extras["scalp_accept_progress_long_bonus"] = int(scalp_accept_progress_long_bonus)
    extras["scalp_accept_progress_short_bonus"] = int(scalp_accept_progress_short_bonus)

    tape_long = {"eligible": False, "readiness": 0.0, "tightening": 0.0, "structural_hold": 0.0, "pressure": 0.0, "release_proximity": 0.0}
    tape_short = {"eligible": False, "readiness": 0.0, "tightening": 0.0, "structural_hold": 0.0, "pressure": 0.0, "release_proximity": 0.0}
    tape_long_bonus = 0
    tape_short_bonus = 0
    tape_pre_long_assist = False
    tape_pre_short_assist = False
    if tape_mode_enabled:
        tape_long = _compute_tape_readiness(
            df,
            direction="LONG",
            atr_last=float(atr_last) if atr_last is not None else None,
            release_level=float(ref_vwap) if isinstance(ref_vwap, (float, int)) else None,
            structural_level=float(ref_vwap) if isinstance(ref_vwap, (float, int)) else None,
            trigger_near=bool(pre_trigger_near),
            baseline_ok=bool(pre_trigger_near and reclaim_attempt_long and (struct_pre_long or liquidity_event_long or orb_event_long) and ((macd_build_long_pts >= 3) or macd_pre_long or rsi_pre_long)),
        )
        tape_short = _compute_tape_readiness(
            df,
            direction="SHORT",
            atr_last=float(atr_last) if atr_last is not None else None,
            release_level=float(ref_vwap) if isinstance(ref_vwap, (float, int)) else None,
            structural_level=float(ref_vwap) if isinstance(ref_vwap, (float, int)) else None,
            trigger_near=bool(pre_trigger_near),
            baseline_ok=bool(pre_trigger_near and reclaim_attempt_short and (struct_pre_short or liquidity_event_short or orb_event_short) and ((macd_build_short_pts >= 3) or macd_pre_short or rsi_pre_short)),
        )
        tape_reversal_long = _compute_scalp_reversal_stabilization(
            df,
            direction="LONG",
            ref_level=float(ref_vwap) if isinstance(ref_vwap, (float, int)) else None,
            atr_last=float(atr_last) if atr_last is not None else None,
        )
        tape_reversal_short = _compute_scalp_reversal_stabilization(
            df,
            direction="SHORT",
            ref_level=float(ref_vwap) if isinstance(ref_vwap, (float, int)) else None,
            atr_last=float(atr_last) if atr_last is not None else None,
        )
        effective_readiness_long = float(tape_long.get("readiness") or 0.0) + 0.90 * float(tape_reversal_long.get("bonus") or 0.0)
        effective_readiness_short = float(tape_short.get("readiness") or 0.0) + 0.90 * float(tape_reversal_short.get("bonus") or 0.0)
        tape_long_bonus = _tape_bonus_from_readiness(
            effective_readiness_long,
            cap=3,
            thresholds=(4.8, 5.8, 6.8, 7.8),
        )
        tape_short_bonus = _tape_bonus_from_readiness(
            effective_readiness_short,
            cap=3,
            thresholds=(4.8, 5.8, 6.8, 7.8),
        )
        if tape_long_bonus:
            long_points += tape_long_bonus
            long_reasons.append(f"Tape readiness +{tape_long_bonus}")
        if tape_short_bonus:
            short_points += tape_short_bonus
            short_reasons.append(f"Tape readiness +{tape_short_bonus}")
        tape_pre_long_assist = bool(
            pre_trigger_near
            and reclaim_attempt_long
            and (struct_pre_long or liquidity_event_long or orb_event_long)
            and (long_confirmation_score >= max(1, effective_confirm_threshold - 2))
            and (not vwap_event_short)
            and (
                effective_readiness_long >= 5.75
                or (
                    effective_readiness_long >= 5.15
                    and bool(tape_reversal_long.get("stabilizing") or False)
                    and bool(tape_reversal_long.get("reclaim_lean") or False)
                    and ((macd_build_long_pts >= 2) or macd_pre_long or rsi_pre_long)
                )
            )
        )
        tape_pre_short_assist = bool(
            pre_trigger_near
            and reclaim_attempt_short
            and (struct_pre_short or liquidity_event_short or orb_event_short)
            and (short_confirmation_score >= max(1, effective_confirm_threshold - 2))
            and (not vwap_event_long)
            and (
                effective_readiness_short >= 5.75
                or (
                    effective_readiness_short >= 5.15
                    and bool(tape_reversal_short.get("stabilizing") or False)
                    and bool(tape_reversal_short.get("reclaim_lean") or False)
                    and ((macd_build_short_pts >= 2) or macd_pre_short or rsi_pre_short)
                )
            )
        )
    else:
        tape_reversal_long = {"bonus": 0.0, "stabilizing": False, "reclaim_lean": False}
        tape_reversal_short = {"bonus": 0.0, "stabilizing": False, "reclaim_lean": False}
        effective_readiness_long = float(tape_long.get("readiness") or 0.0)
        effective_readiness_short = float(tape_short.get("readiness") or 0.0)
    extras["tape_mode_enabled"] = bool(tape_mode_enabled)
    extras["tape_readiness_long"] = float(tape_long.get("readiness") or 0.0)
    extras["tape_readiness_short"] = float(tape_short.get("readiness") or 0.0)
    extras["tape_tightening_long"] = float(tape_long.get("tightening") or 0.0)
    extras["tape_tightening_short"] = float(tape_short.get("tightening") or 0.0)
    extras["tape_hold_long"] = float(tape_long.get("structural_hold") or 0.0)
    extras["tape_hold_short"] = float(tape_short.get("structural_hold") or 0.0)
    extras["tape_pressure_long"] = float(tape_long.get("pressure") or 0.0)
    extras["tape_pressure_short"] = float(tape_short.get("pressure") or 0.0)
    extras["tape_release_proximity_long"] = float(tape_long.get("release_proximity") or 0.0)
    extras["tape_release_proximity_short"] = float(tape_short.get("release_proximity") or 0.0)
    extras["tape_bonus_applied_long"] = int(tape_long_bonus)
    extras["tape_bonus_applied_short"] = int(tape_short_bonus)
    extras["tape_pre_long_assist"] = bool(tape_pre_long_assist)
    extras["tape_pre_short_assist"] = bool(tape_pre_short_assist)
    extras["tape_effective_readiness_long"] = float(effective_readiness_long)
    extras["tape_effective_readiness_short"] = float(effective_readiness_short)
    extras["tape_reversal_stabilization_long"] = bool(tape_reversal_long.get("stabilizing") or False)
    extras["tape_reversal_stabilization_short"] = bool(tape_reversal_short.get("stabilizing") or False)
    extras["tape_reclaim_lean_long"] = bool(tape_reversal_long.get("reclaim_lean") or False)
    extras["tape_reclaim_lean_short"] = bool(tape_reversal_short.get("reclaim_lean") or False)
    extras["tape_reversal_bonus_long"] = float(tape_reversal_long.get("bonus") or 0.0)
    extras["tape_reversal_bonus_short"] = float(tape_reversal_short.get("bonus") or 0.0)

    # Primary trigger must exist (otherwise we have nothing to anchor a plan).
    # NOTE: this is used by both PRE and CONFIRMED routing.
    primary_trigger = bool(vwap_event or rsi_event or macd_event or pro_trigger or early_pre_long or early_pre_short or ignition_override_long or ignition_override_short or tape_pre_long_assist or tape_pre_short_assist or compression_breakout_long or compression_breakout_short or bool(locals().get("scalp_reversal_trigger_long", False)) or bool(locals().get("scalp_reversal_trigger_short", False)))
    extras["primary_trigger"] = primary_trigger

    # PRE condition: near trigger line on the "wrong" side, with momentum/structure pointing toward a flip.
    pre_long_ok = bool(
        isinstance(ref_vwap, (float, int))
        and isinstance(last_price, (float, int))
        and float(last_price) < float(ref_vwap)
        and (trigger_near or early_pre_long or ignition_override_long or tape_pre_long_assist or compression_breakout_long or bool(locals().get("scalp_reversal_trigger_long", False)))
        and (rsi_event or rsi_pre_long or macd_event or macd_pre_long or pro_trigger_long or ignition_override_long or compression_breakout_long or (macd_build_long_pts >= 4) or bool(locals().get("scalp_reversal_trigger_long", False)))
        and (struct_pre_long or liquidity_event_long or orb_event_long or reclaim_attempt_long or ignition_override_long or compression_breakout_long or bool(locals().get("scalp_reversal_trigger_long", False)))
        and (long_confirmation_score >= max(2, effective_confirm_threshold - 1) or early_pre_long or ignition_override_long or tape_pre_long_assist or compression_breakout_long or bool(locals().get("scalp_reversal_trigger_long", False)))
    )
    pre_short_ok = bool(
        isinstance(ref_vwap, (float, int))
        and isinstance(last_price, (float, int))
        and float(last_price) > float(ref_vwap)
        and (trigger_near or early_pre_short or ignition_override_short or tape_pre_short_assist or compression_breakout_short or bool(locals().get("scalp_reversal_trigger_short", False)))
        and (rsi_event or rsi_pre_short or macd_event or macd_pre_short or pro_trigger_short or ignition_override_short or compression_breakout_short or (macd_build_short_pts >= 4) or bool(locals().get("scalp_reversal_trigger_short", False)))
        and (struct_pre_short or liquidity_event_short or orb_event_short or reclaim_attempt_short or ignition_override_short or compression_breakout_short or bool(locals().get("scalp_reversal_trigger_short", False)))
        and (short_confirmation_score >= max(2, effective_confirm_threshold - 1) or early_pre_short or ignition_override_short or tape_pre_short_assist or compression_breakout_short or bool(locals().get("scalp_reversal_trigger_short", False)))
    )

    # If we're near the trigger line and the setup quality is already strong, emit PRE even if we are
    # one confirmation short (so you don't get the alert *after* the move already started).
    # This is intentionally conservative: requires proximity + at least 2 confirmations + a real trigger anchor.
    try:
        setup_quality_points = float(max(long_points_cal, short_points_cal))
    except Exception:
        setup_quality_points = float(max(long_points, short_points))
    pre_proximity_quality = bool(
        trigger_near
        and primary_trigger
        and max(long_confirmation_score, short_confirmation_score) >= 2
        and setup_quality_points >= float(cfg.get("min_actionable_score", 60)) * 0.85
    )
    extras["pre_proximity_quality"] = bool(pre_proximity_quality)

    # "Soft-hard" volume requirement:
    # If volume is required and missing, defer the final decision until after
    # entry-zone context is known. Weak no-volume setups still die; strong
    # reversal-context setups can survive with a score penalty instead.
    volume_gate_active = bool(int(cfg.get("require_volume", 0)) == 1 and (not volume_event) and (not strong_pro_confluence))
    volume_missing_penalty = 0
    low_volume_override = False
    extras["volume_gate_active"] = bool(volume_gate_active)

    if not primary_trigger:
        return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_points, short_points)), "No primary trigger (VWAP/RSI/MACD/Pro)", None, None, None, None, last_price, last_ts, session, extras)

    # Stage selection:
    #   - CONFIRMED requires full confirmation_score + hard gates.
    #   - PRE can be emitted one notch earlier (approaching VWAP) so traders can be ready.
    active_confirmation_bias = provisional_confirmation_bias
    active_confirmation_score = int(long_confirmation_score if active_confirmation_bias == "LONG" else short_confirmation_score)
    extras["active_confirmation_bias"] = active_confirmation_bias
    extras["active_confirmation_score"] = int(active_confirmation_score)

    if active_confirmation_score < effective_confirm_threshold:
        if pre_long_ok or pre_short_ok or pre_proximity_quality:
            stage = "PRE"
            if active_confirmation_bias == "LONG" and ignition_override_long:
                stage_note = f"PRE: early ignition override ({active_confirmation_score}/{effective_confirm_threshold})"
            elif active_confirmation_bias == "SHORT" and ignition_override_short:
                stage_note = f"PRE: early ignition override ({active_confirmation_score}/{effective_confirm_threshold})"
            else:
                stage_note = f"PRE: trigger proximity (confirmations {active_confirmation_score}/{effective_confirm_threshold})"
        else:
            return SignalResult(
                symbol, "NEUTRAL", _cap_score(max(long_points, short_points)),
                f"Not enough confirmations ({active_confirmation_score}/{effective_confirm_threshold})",
                None, None, None, None,
                last_price, last_ts, session, extras,
            )
    else:
        stage = "CONFIRMED"
        stage_note = f"CONFIRMED ({active_confirmation_score}/{effective_confirm_threshold})"

    exceptional_off_window_quality = False
    if not bool(allowed):
        try:
            exceptional_off_window_quality = bool(
                active_confirmation_score >= max(confirm_threshold + session_confirm_bump, effective_confirm_threshold)
                and setup_quality_points >= float(cfg.get("min_actionable_score", 60)) + max(2, session_window_penalty - 2)
                and (
                    strong_pro_confluence
                    or (bool(volume_event) and bool(primary_trigger) and bool(trigger_near))
                    or (str(session) in ("PREMARKET", "AFTERHOURS") and bool(volume_event) and int(max(tape_long_bonus, tape_short_bonus)) >= 2)
                    or (str(session) == "MIDDAY" and bool(volume_event) and active_confirmation_score >= effective_confirm_threshold + 1)
                )
            )
        except Exception:
            exceptional_off_window_quality = False
        extras["exceptional_off_window_quality"] = bool(exceptional_off_window_quality)
        if stage == "CONFIRMED" and not exceptional_off_window_quality:
            stage = "PRE"
            stage_note = f"PRE: outside preferred {session} window"
        elif stage == "CONFIRMED" and exceptional_off_window_quality:
            stage_note = stage_note + f" | Off-window override ({session})"
        elif stage == "PRE":
            stage_note = (stage_note + " | " if stage_note else "") + f"Outside preferred {session} window"
    else:
        extras["exceptional_off_window_quality"] = False

    # Optional: keep classic hard requirements during RTH when Pro confluence is absent.
    # (These protect the "Cleaner signals" preset from becoming too loose.)
    hard_vwap = (int(cfg.get("require_vwap_event", 0)) == 1) and (not is_extended_session)
    hard_rsi  = (int(cfg.get("require_rsi_event", 0)) == 1) and (not is_extended_session)
    hard_macd = (int(cfg.get("require_macd_turn", 0)) == 1) and (not is_extended_session)

    # Hard gates apply to CONFIRMED only (PRE is allowed to form *before* these print).
    if stage == "CONFIRMED":
        dominant_pre_bias = "LONG" if long_points >= short_points else "SHORT"
        adaptive_rsi_context_ok = bool((dominant_pre_bias == "LONG" and adaptive_rsi_confirm_long) or (dominant_pre_bias == "SHORT" and adaptive_rsi_confirm_short))
        if hard_vwap and (not vwap_event) and (not pro_trigger):
            # If the setup is *almost* there, degrade to PRE instead of dropping it.
            if pre_long_ok or pre_short_ok:
                stage = "PRE"; stage_note = "PRE: VWAP event not printed yet"
            else:
                return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_points, short_points)), "No VWAP reclaim/rejection event", None, None, None, None, last_price, last_ts, session, extras)
        if hard_rsi and (not rsi_event) and (not pro_trigger):
            if adaptive_rsi_context_ok:
                extras["adaptive_rsi_gate_override"] = True
                stage_note = stage_note + " | Adaptive RSI context"
            elif pre_long_ok or pre_short_ok:
                stage = "PRE"; stage_note = "PRE: strict RSI snap/downshift not printed yet"
            else:
                return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_points, short_points)), "No strict RSI-5 snap/downshift event", None, None, None, None, last_price, last_ts, session, extras)
        if hard_macd and (not macd_event) and (not pro_trigger):
            if pre_long_ok or pre_short_ok:
                stage = "PRE"; stage_note = "PRE: MACD turn not printed yet"
            else:
                return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_points, short_points)), "No MACD histogram turn event", None, None, None, None, last_price, last_ts, session, extras)

    # For extended sessions (PM/AH), mark missing classic triggers for transparency.
    if is_extended_session:
        if int(cfg.get("require_vwap_event", 0)) == 1 and (not vwap_event) and (not pro_trigger):
            extras["soft_gate_missing_vwap"] = True
        if int(cfg.get("require_rsi_event", 0)) == 1 and (not rsi_event) and (not pro_trigger) and (not adaptive_rsi_confirm_long) and (not adaptive_rsi_confirm_short):
            extras["soft_gate_missing_rsi"] = True
        if int(cfg.get("require_macd_turn", 0)) == 1 and (not macd_event) and (not pro_trigger):
            extras["soft_gate_missing_macd"] = True

    # ATR-normalized score calibration (per ticker)
    # If target_atr_pct is None => auto-tune per ticker using median ATR% over a recent window.
    # Otherwise => use the manual target ATR% as a global anchor.
    scale = 1.0
    ref_atr_pct = None
    if atr_pct:
        if target_atr_pct is None:
            atr_series = df["atr14"].tail(120)
            close_series = df["close"].tail(120).replace(0, np.nan)
            atr_pct_series = (atr_series / close_series).replace([np.inf, -np.inf], np.nan).dropna()
            if len(atr_pct_series) >= 20:
                ref_atr_pct = float(np.nanmedian(atr_pct_series.values))
        else:
            ref_atr_pct = float(target_atr_pct)

        if ref_atr_pct and ref_atr_pct > 0:
            scale = ref_atr_pct / atr_pct
            # Keep calibration gentle; we want comparability, not distortion.
            scale = float(np.clip(scale, 0.75, 1.25))

    extras["atr_score_scale"] = scale
    extras["atr_ref_pct"] = ref_atr_pct

    long_points_cal = int(round(long_points * scale))
    short_points_cal = int(round(short_points * scale))
    extras["long_points_raw"] = long_points
    extras["short_points_raw"] = short_points
    extras["long_points_cal"] = long_points_cal
    extras["short_points_cal"] = short_points_cal
    extras["contrib_points"] = contrib

    min_score = int(cfg["min_actionable_score"])

    # Entry/stop + targets
    tighten_factor = 1.0
    if pro_mode:
        # Tighten stops a bit when we have structural confluence.
        # NOTE: We intentionally do NOT mutate the setup_score here; scoring is handled above.
        confluence = bool(
            (isinstance(rsi_div, dict) and rsi_div.get("type") in ("bull", "bear"))
            or bull_sweep or bear_sweep
            or orb_bull or orb_bear
            or bull_ob_retest or bear_ob_retest
            or bull_breaker_retest or bear_breaker_retest
            or (bull_fvg is not None) or (bear_fvg is not None)
        )
        if confluence:
            tighten_factor = 0.85
        extras["stop_tighten_factor"] = float(tighten_factor)

    def _fib_take_profits_long(entry_px: float) -> Tuple[Optional[float], Optional[float]]:
        if rng <= 0:
            return None, None
        exts = _fib_extensions(hi, lo)
        # Partial at recent high if above entry, else at ext 1.272
        tp1 = hi if entry_px < hi else next((lvl for _, lvl in exts if lvl > entry_px), None)
        tp2 = next((lvl for _, lvl in exts if lvl and tp1 and lvl > tp1), None)
        return (float(tp1) if tp1 else None, float(tp2) if tp2 else None)

    def _fib_take_profits_short(entry_px: float) -> Tuple[Optional[float], Optional[float]]:
        if rng <= 0:
            return None, None
        # Mirror extensions below lo
        ratios = [1.0, 1.272, 1.618]
        exts_dn = [ (f"Ext -{r:g}", lo - (r - 1.0) * rng) for r in ratios ]
        tp1 = lo if entry_px > lo else next((lvl for _, lvl in exts_dn if lvl < entry_px), None)
        tp2 = next((lvl for _, lvl in exts_dn if lvl and tp1 and lvl < tp1), None)
        return (float(tp1) if tp1 else None, float(tp2) if tp2 else None)

    def _long_entry_stop(entry_px: float):
        stop_px = float(min(recent_swing_low, entry_px - max(atr_last, 0.0) * 0.8))
        if pro_mode and tighten_factor < 1.0:
            stop_px = float(entry_px - (entry_px - stop_px) * tighten_factor)
        if bull_breaker_retest and brk_bull[0] is not None:
            stop_px = float(min(stop_px, brk_bull[0] - buffer))
        if fib_near_long and fib_level is not None:
            stop_px = float(min(stop_px, fib_level - buffer))
        return entry_px, stop_px

    def _short_entry_stop(entry_px: float):
        stop_px = float(max(recent_swing_high, entry_px + max(atr_last, 0.0) * 0.8))
        if pro_mode and tighten_factor < 1.0:
            stop_px = float(entry_px + (stop_px - entry_px) * tighten_factor)
        if bear_breaker_retest and brk_bear[1] is not None:
            stop_px = float(max(stop_px, brk_bear[1] + buffer))
        if fib_near_short and fib_level is not None:
            stop_px = float(max(stop_px, fib_level + buffer))
        return entry_px, stop_px
    # Final decision + trade levels
    long_score = int(round(float(long_points_cal))) if 'long_points_cal' in locals() else int(round(float(long_points)))
    short_score = int(round(float(short_points_cal))) if 'short_points_cal' in locals() else int(round(float(short_points)))

    # Never allow scores outside 0..100.
    long_score = _cap_score(long_score)
    short_score = _cap_score(short_score)

    # SCALP reversal-trigger scoring bridge:
    # A true structure-first reversal trigger should not die merely because the old
    # confirmation stack has not fully printed yet. Apply a quality-gated floor only
    # for the emitted trigger side. This keeps weak candle noise from auto-exec while
    # allowing high-quality hammer/engulf/fakeout/absorption triggers to clear a
    # 90-ish auto-exec threshold when the rest of the setup is already close.
    try:
        _rev_floor_applied = 0
        if bool(extras.get("scalp_reversal_trigger_long")):
            _q = float(extras.get("scalp_reversal_trigger_score_long") or 0.0)
            _floor = 0
            if _q >= 0.70:
                _floor = max(90, int(min_score))
            elif _q >= 0.58:
                _floor = max(88, int(min_score) - 2)
            if _floor and long_score >= (_floor - 8):
                long_score = _cap_score(max(long_score, _floor))
                _rev_floor_applied = int(_floor)
        if bool(extras.get("scalp_reversal_trigger_short")):
            _q = float(extras.get("scalp_reversal_trigger_score_short") or 0.0)
            _floor = 0
            if _q >= 0.70:
                _floor = max(90, int(min_score))
            elif _q >= 0.58:
                _floor = max(88, int(min_score) - 2)
            if _floor and short_score >= (_floor - 8):
                short_score = _cap_score(max(short_score, _floor))
                _rev_floor_applied = int(_floor)
        extras["scalp_reversal_trigger_score_floor_applied"] = int(_rev_floor_applied)
    except Exception:
        extras["scalp_reversal_trigger_score_floor_applied"] = 0

    # NOTE: entry-zone context is evaluated later, after we have a concrete executable entry_limit.
    # So do not apply any zone adjustment or min-score return here.
    extras["entry_zone_score_adj"] = 0

    # Stage + direction
    extras["stage"] = stage
    extras["stage_note"] = stage_note

    # For PRE alerts, prefer the directional pre-condition when it is unambiguous.
    if stage == "PRE" and (ignition_override_long or pre_long_ok) and not (ignition_override_short or pre_short_ok):
        bias = "LONG"
    elif stage == "PRE" and (ignition_override_short or pre_short_ok) and not (ignition_override_long or pre_long_ok):
        bias = "SHORT"
    else:
        bias = "LONG" if long_score >= short_score else "SHORT"

    # Final confirmation alignment: keep stage/confirmation diagnostics anchored to the
    # actual emitted side after all later score adjustments settle the winning bias.
    final_confirmation_score = int(long_confirmation_score if bias == "LONG" else short_confirmation_score)
    final_confirmation_components = long_confirmation_components if bias == "LONG" else short_confirmation_components
    active_confirmation_bias = bias
    active_confirmation_score = final_confirmation_score
    extras["active_confirmation_bias"] = active_confirmation_bias
    extras["active_confirmation_score"] = int(active_confirmation_score)
    extras["active_confirmation_components"] = final_confirmation_components
    extras["confirmation_bias"] = bias
    extras["confirmation_score"] = int(final_confirmation_score)
    extras["confirmation_components"] = final_confirmation_components

    if stage == "CONFIRMED" and active_confirmation_score < effective_confirm_threshold:
        if ((bias == "LONG" and pre_long_ok) or (bias == "SHORT" and pre_short_ok)):
            stage = "PRE"
            stage_note = f"PRE: final bias confirmation alignment ({active_confirmation_score}/{effective_confirm_threshold})"
        else:
            stage = None
            return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_score, short_score)), f"Final bias lacks aligned confirmations ({active_confirmation_score}/{effective_confirm_threshold})", None, None, None, None, last_price, last_ts, session, extras)

    extras["stage"] = stage
    extras["stage_note"] = stage_note
    setup_score = _cap_score(max(long_score, short_score))

    # Assemble reason text from the winning side
    if bias == "LONG":
        reasons = long_reasons[:] if 'long_reasons' in locals() else []
    else:
        reasons = short_reasons[:] if 'short_reasons' in locals() else []
    try:
        if isinstance(extras.get("entry_zone_context"), dict):
            ez = extras.get("entry_zone_context") or {}
            if ez.get("favorable") and ez.get("favorable_type"):
                reasons.append(f"Entry near {ez.get('favorable_type')}")
            if ez.get("hostile") and ez.get("hostile_type"):
                reasons.append(f"Entry near {ez.get('hostile_type')}")
    except Exception:
        pass

    core_reason = "; ".join(reasons) if reasons else "Actionable setup"
    reason = (stage_note + " — " if stage_note else "") + core_reason

    # Entry model context
    ref_vwap = None
    try:
        ref_vwap = float(vwap_use.iloc[-1])
    except Exception:
        ref_vwap = None

    mid_price = None
    try:
        mid_price = float((df["high"].iloc[-1] + df["low"].iloc[-1]) / 2.0)
    except Exception:
        mid_price = None

    # Adaptive SCALP acceptance line:
    # blend reclaim/rejection anchors with nearby structure, but favor the level
    # the market has defended most recently so entry logic follows the live shelf.
    scalp_accept_line = ref_vwap
    scalp_accept_src = "VWAP"
    scalp_accept_diag = {}
    try:
        ema20_ref = float(df["ema20"].iloc[-1]) if np.isfinite(df["ema20"].iloc[-1]) else None
    except Exception:
        ema20_ref = None
    try:
        if bias == "LONG":
            hold_count = int((close.astype(float).tail(int(min(3, len(close)))) >= float(ref_vwap) - 0.10 * float(buffer)).sum()) if isinstance(ref_vwap, (float, int)) else 0
            pivot_anchor = float(recent_swing_low)
            anchors = []
            weights = []
            diag = {}
            if isinstance(ref_vwap, (float, int)):
                base_w = 0.58 + 0.10 * hold_count
                rx = _anchor_recent_interaction_score(df, direction="LONG", anchor=float(ref_vwap), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
                diag["VWAP"] = rx
                anchors.append(float(ref_vwap)); weights.append(base_w + 0.30 * float(rx.get("score") or 0.0))
            if isinstance(pivot_anchor, (float, int)):
                pivot_near = max(0.0, 1.0 - (abs(float(last_price) - float(pivot_anchor)) / max(1e-9, 1.25 * float(atr_last or 1.0))))
                rx = _anchor_recent_interaction_score(df, direction="LONG", anchor=float(pivot_anchor), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
                diag["PIVOT"] = rx
                weights.append(0.22 + 0.18 * float(micro_hl) + 0.10 * pivot_near + 0.34 * float(rx.get("score") or 0.0))
                anchors.append(float(pivot_anchor))
            if isinstance(ema20_ref, (float, int)):
                ema_near = max(0.0, 1.0 - (abs(float(last_price) - float(ema20_ref)) / max(1e-9, 1.50 * float(atr_last or 1.0))))
                rx = _anchor_recent_interaction_score(df, direction="LONG", anchor=float(ema20_ref), atr_last=float(atr_last) if atr_last is not None else None, lookback=6)
                diag["EMA20"] = rx
                anchors.append(float(ema20_ref)); weights.append(0.12 + 0.08 * float(trend_long_ok) + 0.05 * ema_near + 0.20 * float(rx.get("score") or 0.0))
            if anchors and sum(weights) > 0:
                scalp_accept_line = float(np.average(np.asarray(anchors, dtype=float), weights=np.asarray(weights, dtype=float)))
                scalp_accept_line = float(min(float(last_price), max(float(min(anchors)), scalp_accept_line)))
                scalp_accept_src = "BLEND" if len(anchors) > 1 else "VWAP"
            scalp_accept_diag = diag
        else:
            hold_count = int((close.astype(float).tail(int(min(3, len(close)))) <= float(ref_vwap) + 0.10 * float(buffer)).sum()) if isinstance(ref_vwap, (float, int)) else 0
            pivot_anchor = float(recent_swing_high)
            anchors = []
            weights = []
            diag = {}
            if isinstance(ref_vwap, (float, int)):
                base_w = 0.58 + 0.10 * hold_count
                rx = _anchor_recent_interaction_score(df, direction="SHORT", anchor=float(ref_vwap), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
                diag["VWAP"] = rx
                anchors.append(float(ref_vwap)); weights.append(base_w + 0.30 * float(rx.get("score") or 0.0))
            if isinstance(pivot_anchor, (float, int)):
                pivot_near = max(0.0, 1.0 - (abs(float(last_price) - float(pivot_anchor)) / max(1e-9, 1.25 * float(atr_last or 1.0))))
                rx = _anchor_recent_interaction_score(df, direction="SHORT", anchor=float(pivot_anchor), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
                diag["PIVOT"] = rx
                anchors.append(float(pivot_anchor)); weights.append(0.22 + 0.18 * float(micro_lh) + 0.10 * pivot_near + 0.34 * float(rx.get("score") or 0.0))
            if isinstance(ema20_ref, (float, int)):
                ema_near = max(0.0, 1.0 - (abs(float(last_price) - float(ema20_ref)) / max(1e-9, 1.50 * float(atr_last or 1.0))))
                rx = _anchor_recent_interaction_score(df, direction="SHORT", anchor=float(ema20_ref), atr_last=float(atr_last) if atr_last is not None else None, lookback=6)
                diag["EMA20"] = rx
                anchors.append(float(ema20_ref)); weights.append(0.12 + 0.08 * float(trend_short_ok) + 0.05 * ema_near + 0.20 * float(rx.get("score") or 0.0))
            if anchors and sum(weights) > 0:
                scalp_accept_line = float(np.average(np.asarray(anchors, dtype=float), weights=np.asarray(weights, dtype=float)))
                scalp_accept_line = float(max(float(last_price), min(float(max(anchors)), scalp_accept_line)))
                scalp_accept_src = "BLEND" if len(anchors) > 1 else "VWAP"
            scalp_accept_diag = diag
    except Exception:
        scalp_accept_line = ref_vwap
        scalp_accept_src = "VWAP"
        scalp_accept_diag = {}

    extras["accept_line"] = float(scalp_accept_line) if isinstance(scalp_accept_line, (float, int)) else None
    extras["accept_src"] = scalp_accept_src
    extras["accept_line_raw"] = float(ref_vwap) if isinstance(ref_vwap, (float, int)) else None
    extras["accept_line_recent_diag"] = scalp_accept_diag

    entry_px = _entry_from_model(
        bias,
        entry_model=entry_model,
        last_price=float(last_price),
        ref_vwap=(float(scalp_accept_line) if isinstance(scalp_accept_line, (float, int)) else ref_vwap),
        mid_price=mid_price,
        atr_last=float(atr_last) if atr_last is not None else 0.0,
        slippage_mode=slippage_mode,
        fixed_slippage_cents=fixed_slippage_cents,
        atr_fraction_slippage=atr_fraction_slippage,
    )

    # Entry model upgrade: expose both a limit entry and a chase-line.
    entry_limit, chase_line = _entry_limit_and_chase(
        bias,
        entry_px=float(entry_px),
        last_px=float(last_price),
        atr_last=float(atr_last) if atr_last is not None else 0.0,
        slippage_mode=slippage_mode,
        fixed_slippage_cents=fixed_slippage_cents,
        atr_fraction_slippage=atr_fraction_slippage,
    )

    # Priority #2 — SCALP timing improvement:
    # On fast snap reversals, the old flow could wait for reclaim/confirmation and end up
    # entering after a meaningful portion of the first impulse had already occurred.
    # Here we allow a modest "early reversal" entry improvement when price has clearly
    # stabilized and started reclaiming, but before the move becomes fully comfortable.
    scalp_early_entry_applied = False
    scalp_early_entry_reason = None
    scalp_early_entry_anchor = None
    try:
        atr_ref = float(atr_last or 0.0)
        if isinstance(entry_limit, (float, int)) and isinstance(last_price, (float, int)) and atr_ref > 0:
            recent = df.tail(int(min(6, len(df))))
            lows = pd.to_numeric(recent.get("low"), errors="coerce") if len(recent) else pd.Series(dtype=float)
            highs = pd.to_numeric(recent.get("high"), errors="coerce") if len(recent) else pd.Series(dtype=float)
            closes = pd.to_numeric(recent.get("close"), errors="coerce") if len(recent) else pd.Series(dtype=float)
            opens = pd.to_numeric(recent.get("open"), errors="coerce") if len(recent) else pd.Series(dtype=float)
            vols = pd.to_numeric(recent.get("volume"), errors="coerce") if len(recent) and "volume" in recent else pd.Series(dtype=float)
            if len(closes) >= 4 and len(lows) >= 4 and len(highs) >= 4 and len(opens) >= 4:
                body_last = abs(float(closes.iloc[-1]) - float(opens.iloc[-1]))
                body_prev = abs(float(closes.iloc[-2]) - float(opens.iloc[-2]))
                rng_last = max(1e-9, float(highs.iloc[-1]) - float(lows.iloc[-1]))
                rng_prev = max(1e-9, float(highs.iloc[-2]) - float(lows.iloc[-2]))
                upper_wick_last = float(highs.iloc[-1]) - max(float(closes.iloc[-1]), float(opens.iloc[-1]))
                lower_wick_last = min(float(closes.iloc[-1]), float(opens.iloc[-1])) - float(lows.iloc[-1])
                upper_wick_prev = float(highs.iloc[-2]) - max(float(closes.iloc[-2]), float(opens.iloc[-2]))
                lower_wick_prev = min(float(closes.iloc[-2]), float(opens.iloc[-2])) - float(lows.iloc[-2])
                close_pos_last = (float(closes.iloc[-1]) - float(lows.iloc[-1])) / rng_last
                close_pos_last_short = (float(highs.iloc[-1]) - float(closes.iloc[-1])) / rng_last
                local_low = float(lows.tail(4).min())
                local_high = float(highs.tail(4).max())
                low_hold = min(float(lows.iloc[-1]), float(lows.iloc[-2])) >= local_low - 0.06 * atr_ref
                high_hold = max(float(highs.iloc[-1]), float(highs.iloc[-2])) <= local_high + 0.06 * atr_ref
                reclaim_progress_long = float(closes.iloc[-1]) >= local_low + 0.32 * atr_ref
                reclaim_progress_short = float(closes.iloc[-1]) <= local_high - 0.32 * atr_ref
                vol_support_long = True
                vol_support_short = True
                if len(vols) >= 4 and vols.notna().sum() >= 4:
                    v_last = float(vols.iloc[-1])
                    v_prev = float(vols.iloc[-2])
                    v_base = float(vols.iloc[:-2].replace(0, pd.NA).dropna().mean()) if len(vols) > 2 and not vols.iloc[:-2].replace(0, pd.NA).dropna().empty else 0.0
                    if v_base > 0:
                        vol_support_long = (v_last >= 0.70 * v_base) or (v_last >= 0.90 * v_prev)
                        vol_support_short = vol_support_long

                # Reversal-trigger entry: when SCALP sees the actual rejection/engulf/sweep candle,
                # prefer an executable close/next-hold entry instead of waiting for the old VWAP
                # limit to become stale and then blocking it as CHASE.
                if str(bias).upper() == "LONG" and bool(extras.get("scalp_reversal_trigger_long")):
                    try:
                        trig_anchor = float(extras.get("scalp_reversal_trigger_entry_anchor_long") or closes.iloc[-1])
                        struct_ref = float(extras.get("scalp_reversal_trigger_structure_ref_long") or lows.tail(4).min())
                        # Keep the entry close enough to the trigger candle to avoid a false chase,
                        # but do not set an unrealistically low limit that never fills.
                        candidate = min(max(float(entry_limit), float(last_price) - 0.05 * atr_ref), float(last_price) + 0.03 * atr_ref)
                        candidate = max(candidate, struct_ref + 0.18 * atr_ref)
                        if candidate <= float(last_price) + 0.08 * atr_ref:
                            entry_limit = float(candidate)
                            chase_line = max(float(chase_line), float(entry_limit) + 0.10 * atr_ref) if isinstance(chase_line, (float, int)) else float(entry_limit + 0.10 * atr_ref)
                            scalp_early_entry_applied = True
                            scalp_early_entry_reason = f"REVERSAL_TRIGGER_{extras.get('scalp_reversal_trigger_type_long') or 'LONG'}"
                            scalp_early_entry_anchor = float(trig_anchor)
                    except Exception:
                        pass
                elif str(bias).upper() == "SHORT" and bool(extras.get("scalp_reversal_trigger_short")):
                    try:
                        trig_anchor = float(extras.get("scalp_reversal_trigger_entry_anchor_short") or closes.iloc[-1])
                        struct_ref = float(extras.get("scalp_reversal_trigger_structure_ref_short") or highs.tail(4).max())
                        candidate = max(min(float(entry_limit), float(last_price) + 0.05 * atr_ref), float(last_price) - 0.03 * atr_ref)
                        candidate = min(candidate, struct_ref - 0.18 * atr_ref)
                        if candidate >= float(last_price) - 0.08 * atr_ref:
                            entry_limit = float(candidate)
                            chase_line = min(float(chase_line), float(entry_limit) - 0.10 * atr_ref) if isinstance(chase_line, (float, int)) else float(entry_limit - 0.10 * atr_ref)
                            scalp_early_entry_applied = True
                            scalp_early_entry_reason = f"REVERSAL_TRIGGER_{extras.get('scalp_reversal_trigger_type_short') or 'SHORT'}"
                            scalp_early_entry_anchor = float(trig_anchor)
                    except Exception:
                        pass

                fast_rev_long = bool(
                    str(bias).upper() == "LONG"
                    and (bool(early_pre_long) or bool(tape_pre_long_assist) or (str(stage).upper() == "CONFIRMED" and bool(rsi_pre_long) and bool(macd_pre_long) and bool(struct_pre_long)))
                    and float(closes.iloc[-1]) > float(closes.iloc[-2])
                    and low_hold
                    and reclaim_progress_long
                    and body_last >= max(0.08 * atr_ref, 0.85 * max(1e-9, body_prev))
                    and body_last >= 0.42 * rng_last
                    and close_pos_last >= 0.58
                    and upper_wick_last <= 0.52 * rng_last
                    and lower_wick_last + 0.03 * atr_ref >= 0.85 * max(0.0, lower_wick_prev)
                    and vol_support_long
                    and float(last_price) <= float(entry_limit) + 0.32 * atr_ref
                )
                fast_rev_short = bool(
                    str(bias).upper() == "SHORT"
                    and (bool(early_pre_short) or bool(tape_pre_short_assist) or (str(stage).upper() == "CONFIRMED" and bool(rsi_pre_short) and bool(macd_pre_short) and bool(struct_pre_short)))
                    and float(closes.iloc[-1]) < float(closes.iloc[-2])
                    and high_hold
                    and reclaim_progress_short
                    and body_last >= max(0.08 * atr_ref, 0.85 * max(1e-9, body_prev))
                    and body_last >= 0.42 * rng_last
                    and close_pos_last_short >= 0.58
                    and lower_wick_last <= 0.52 * rng_last
                    and upper_wick_last + 0.03 * atr_ref >= 0.85 * max(0.0, upper_wick_prev)
                    and vol_support_short
                    and float(last_price) >= float(entry_limit) - 0.32 * atr_ref
                )
                if fast_rev_long:
                    accept_ref = float(scalp_accept_line) if isinstance(scalp_accept_line, (float, int)) else float(entry_limit)
                    reclaim_anchor = max(local_low + 0.28 * atr_ref, accept_ref + 0.01 * atr_ref)
                    candidate = min(float(entry_limit), max(reclaim_anchor, float(last_price) - 0.10 * atr_ref))
                    if candidate < float(entry_limit):
                        entry_limit = float(candidate)
                        chase_line = max(float(chase_line), float(entry_limit) + 0.12 * atr_ref) if isinstance(chase_line, (float, int)) else float(entry_limit + 0.12 * atr_ref)
                        scalp_early_entry_applied = True
                        scalp_early_entry_reason = "FAST_REVERSAL_LONG"
                        scalp_early_entry_anchor = float(reclaim_anchor)
                elif fast_rev_short:
                    accept_ref = float(scalp_accept_line) if isinstance(scalp_accept_line, (float, int)) else float(entry_limit)
                    reclaim_anchor = min(local_high - 0.28 * atr_ref, accept_ref - 0.01 * atr_ref)
                    candidate = max(float(entry_limit), min(reclaim_anchor, float(last_price) + 0.10 * atr_ref))
                    if candidate > float(entry_limit):
                        entry_limit = float(candidate)
                        chase_line = min(float(chase_line), float(entry_limit) - 0.12 * atr_ref) if isinstance(chase_line, (float, int)) else float(entry_limit - 0.12 * atr_ref)
                        scalp_early_entry_applied = True
                        scalp_early_entry_reason = "FAST_REVERSAL_SHORT"
                        scalp_early_entry_anchor = float(reclaim_anchor)

                # Ignition micro-pullback entry: after a clean ignition bar, prefer a tiny one-bar pullback
                # instead of buying the full extension. Tier 3 can lean in shallower; Tier 2 asks for a touch more giveback.
                ign_tier_long = int(extras.get("ignition_tier_long") or 0)
                ign_tier_short = int(extras.get("ignition_tier_short") or 0)
                if str(bias).upper() == "LONG" and ign_tier_long >= 2:
                    accept_ref = float(scalp_accept_line) if isinstance(scalp_accept_line, (float, int)) else float(entry_limit)
                    micro_pad = (0.12 if ign_tier_long >= 3 else 0.18) * atr_ref
                    micro_anchor = max(accept_ref + 0.01 * atr_ref, float(closes.iloc[-1]) - micro_pad)
                    candidate = min(float(entry_limit), micro_anchor)
                    if candidate < float(entry_limit) and candidate <= float(last_price) + 0.06 * atr_ref:
                        entry_limit = float(candidate)
                        chase_line = max(float(chase_line), float(entry_limit) + (0.10 if ign_tier_long >= 3 else 0.12) * atr_ref) if isinstance(chase_line, (float, int)) else float(entry_limit + 0.12 * atr_ref)
                        scalp_early_entry_applied = True
                        scalp_early_entry_reason = f"IGNITION_TIER{ign_tier_long}_LONG"
                        scalp_early_entry_anchor = float(micro_anchor)
                elif str(bias).upper() == "SHORT" and ign_tier_short >= 2:
                    accept_ref = float(scalp_accept_line) if isinstance(scalp_accept_line, (float, int)) else float(entry_limit)
                    micro_pad = (0.12 if ign_tier_short >= 3 else 0.18) * atr_ref
                    micro_anchor = min(accept_ref - 0.01 * atr_ref, float(closes.iloc[-1]) + micro_pad)
                    candidate = max(float(entry_limit), micro_anchor)
                    if candidate > float(entry_limit) and candidate >= float(last_price) - 0.06 * atr_ref:
                        entry_limit = float(candidate)
                        chase_line = min(float(chase_line), float(entry_limit) - (0.10 if ign_tier_short >= 3 else 0.12) * atr_ref) if isinstance(chase_line, (float, int)) else float(entry_limit - 0.12 * atr_ref)
                        scalp_early_entry_applied = True
                        scalp_early_entry_reason = f"IGNITION_TIER{ign_tier_short}_SHORT"
                        scalp_early_entry_anchor = float(micro_anchor)
    except Exception:
        scalp_early_entry_applied = False
        scalp_early_entry_reason = None
        scalp_early_entry_anchor = None

    extras["scalp_early_entry_applied"] = bool(scalp_early_entry_applied)
    extras["scalp_early_entry_reason"] = scalp_early_entry_reason
    extras["scalp_early_entry_anchor"] = float(scalp_early_entry_anchor) if isinstance(scalp_early_entry_anchor, (float, int)) else None

    # Entry model upgrade: adapt when the planned limit is already stale.
    # If price has already moved beyond the limit by a meaningful fraction of ATR,
    # we flip the plan to a chase-based execution so we don't alert *after* the move.
    #
    # - LONG: if last is above the limit by > stale_buffer => use chase line as the new entry.
    # - SHORT: if last is below the limit by > stale_buffer => use chase line as the new entry.
    #
    # This keeps entry/stop/TP coherent (all are computed off entry_limit) while preserving
    # the informational chase line for the trader.
    stale_buffer = None
    try:
        stale_buffer = max(0.25 * float(atr_last or 0.0), 0.0006 * float(last_price or 0.0))
    except Exception:
        stale_buffer = None

    exec_mode = "LIMIT"
    entry_stale = False
    if isinstance(stale_buffer, (float, int)) and stale_buffer and stale_buffer > 0:
        try:
            if bias == "LONG" and float(last_price) > float(entry_limit) + float(stale_buffer):
                exec_mode = "CHASE"; entry_stale = True
                entry_limit = float(chase_line)
            elif bias == "SHORT" and float(last_price) < float(entry_limit) - float(stale_buffer):
                exec_mode = "CHASE"; entry_stale = True
                entry_limit = float(chase_line)
        except Exception:
            pass

    extras["execution_mode"] = exec_mode
    extras["entry_stale"] = bool(entry_stale)
    extras["entry_stale_buffer"] = float(stale_buffer) if isinstance(stale_buffer, (float, int)) else None
    extras["entry_limit"] = float(entry_limit)
    extras["entry_chase_line"] = float(chase_line)

    # Trader-edge late-entry guard: CHASE is allowed only for elite ignition/continuation.
    try:
        _reclaim_cont_chase_ok = bool(
            (str(bias).upper() == "LONG" and bool(extras.get("scalp_reclaim_continuation_long")) and (vol_ok_long or bool((pressure_states or {}).get("volume_absorption_long")) or bool((pressure_states or {}).get("directional_expansion_long"))) and float(locals().get("_ext_from_ref", 99.0) or 99.0) <= 1.25)
            or (str(bias).upper() == "SHORT" and bool(extras.get("scalp_reclaim_continuation_short")) and (vol_ok_short or bool((pressure_states or {}).get("volume_absorption_short")) or bool((pressure_states or {}).get("directional_expansion_short"))) and float(locals().get("_ext_from_ref", 99.0) or 99.0) <= 1.25)
        )
        _elite_chase_ok = bool(
            (str(bias).upper() == "LONG" and int(extras.get("ignition_tier_long") or 0) >= 3 and (vol_ok_long or momentum_cont_long))
            or (str(bias).upper() == "SHORT" and int(extras.get("ignition_tier_short") or 0) >= 3 and (vol_ok_short or momentum_cont_short))
            or _reclaim_cont_chase_ok
            or (str(bias).upper() == "LONG" and bool(extras.get("scalp_reversal_trigger_long")) and float(locals().get("_ext_from_ref", 99.0) or 99.0) <= 1.30)
            or (str(bias).upper() == "SHORT" and bool(extras.get("scalp_reversal_trigger_short")) and float(locals().get("_ext_from_ref", 99.0) or 99.0) <= 1.30)
        )
    except Exception:
        _reclaim_cont_chase_ok = False
        _elite_chase_ok = False
    extras["reclaim_cont_chase_ok"] = bool(_reclaim_cont_chase_ok)
    extras["reversal_trigger_chase_ok"] = bool((str(bias).upper() == "LONG" and bool(extras.get("scalp_reversal_trigger_long"))) or (str(bias).upper() == "SHORT" and bool(extras.get("scalp_reversal_trigger_short"))))
    extras["elite_chase_ok"] = bool(_elite_chase_ok)
    if exec_mode == "CHASE" and not _elite_chase_ok:
        return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_score, short_score)), "Late CHASE blocked: edge already moved", None, None, None, None, last_price, last_ts, session, extras)

    # Entry-zone context: small local demand/supply tilt around the proposed executable entry.
    scalp_zone_ctx = _evaluate_entry_zone_context(
        df, entry_price=float(entry_limit), direction=str(bias), atr_last=float(atr_last) if atr_last is not None else None, lookback=10
    )
    scalp_zone_adj = 0.0
    fav_q = float(scalp_zone_ctx.get("favorable_quality") or 0.0)
    host_q = float(scalp_zone_ctx.get("hostile_quality") or 0.0)
    fav_inside = bool(scalp_zone_ctx.get("favorable_inside"))
    host_inside = bool(scalp_zone_ctx.get("hostile_inside"))
    if bool(scalp_zone_ctx.get("favorable")):
        favorable_boost = 3.0 + 2.0 * fav_q
        if fav_inside:
            favorable_boost += 1.25 + 1.25 * fav_q
        scalp_zone_adj += favorable_boost
    if bool(scalp_zone_ctx.get("hostile")):
        hostile_pen = 4.0 + 3.0 * host_q
        if host_inside:
            hostile_pen += 1.5 + 1.5 * host_q
        if stage == "PRE":
            hostile_pen += 1.5 + 1.5 * host_q
        if exec_mode == "CHASE":
            hostile_pen += 0.75
        scalp_zone_adj -= hostile_pen
    extras["entry_zone_context"] = scalp_zone_ctx
    extras["entry_zone_score_adj"] = int(round(float(scalp_zone_adj))) if isinstance(scalp_zone_adj, (int, float)) else 0

    # Apply the zone tilt to the currently selected direction BEFORE final threshold gating.
    try:
        if isinstance(scalp_zone_adj, (int, float)) and scalp_zone_adj != 0:
            if str(bias).upper() == "LONG":
                long_score = _cap_score(long_score + int(round(float(scalp_zone_adj))))
            elif str(bias).upper() == "SHORT":
                short_score = _cap_score(short_score + int(round(float(scalp_zone_adj))))
    except Exception:
        pass
    setup_score = _cap_score(max(long_score, short_score))

    if session_window_penalty > 0:
        if str(bias).upper() == "LONG":
            long_score = _cap_score(long_score - int(session_window_penalty))
        else:
            short_score = _cap_score(short_score - int(session_window_penalty))
        setup_score = _cap_score(max(long_score, short_score))
        reason = (reason + "; " if reason else "") + f"Outside preferred {session} window (-{int(session_window_penalty)})"
        extras["session_penalty_applied"] = int(session_window_penalty)
    else:
        extras["session_penalty_applied"] = 0

    scalp_extension_profile = _compute_multibar_extension_profile(
        df,
        direction=str(bias),
        atr_last=float(atr_last) if atr_last is not None else None,
        accept_line=float(scalp_accept_line) if isinstance(scalp_accept_line, (float, int)) else None,
    )
    extras["scalp_extension_profile"] = scalp_extension_profile
    scalp_ext_penalty = int(round(4.0 * float(scalp_extension_profile.get("penalty") or 0.0)))
    if scalp_ext_penalty > 0:
        if str(bias).upper() == "LONG":
            long_score = _cap_score(long_score - scalp_ext_penalty)
        else:
            short_score = _cap_score(short_score - scalp_ext_penalty)
        setup_score = _cap_score(max(long_score, short_score))
        reason = (reason + "; " if reason else "") + f"Extension guard (-{int(scalp_ext_penalty)})"

    scalp_weak_tape_diag = {"score": 1.0, "ok": True, "stall": False, "rejection": False}
    weak_tape_env = bool((not strong_pro_confluence) and ((adx_last is None) or (float(adx_last) < 18.0) or (not volume_event)))
    if stage == "PRE" and weak_tape_env:
        scalp_weak_tape_diag = _assess_scalp_weak_tape_turn(
            df,
            direction=str(bias),
            trigger_line=float(scalp_accept_line) if isinstance(scalp_accept_line, (float, int)) else float(ref_vwap) if isinstance(ref_vwap, (float, int)) else None,
            atr_last=float(atr_last) if atr_last is not None else None,
        )
        extras["scalp_weak_tape_diag"] = scalp_weak_tape_diag
        if bool(scalp_weak_tape_diag.get("rejection")):
            return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_score, short_score)), "PRE blocked: repeated trigger rejection in weak tape", None, None, None, None, last_price, last_ts, session, extras)
        if not bool(scalp_weak_tape_diag.get("ok")):
            return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_score, short_score)), "PRE blocked: weak tape lacks stall/turn quality", None, None, None, None, last_price, last_ts, session, extras)
    else:
        extras["scalp_weak_tape_diag"] = scalp_weak_tape_diag

    try:
        if isinstance(scalp_zone_ctx, dict):
            if scalp_zone_ctx.get("favorable") and scalp_zone_ctx.get("favorable_type"):
                reason = (reason + "; " if reason else "") + f"Entry near {scalp_zone_ctx.get('favorable_type')}"
            if scalp_zone_ctx.get("hostile") and scalp_zone_ctx.get("hostile_type"):
                reason = (reason + "; " if reason else "") + f"Entry near {scalp_zone_ctx.get('hostile_type')}"
    except Exception:
        pass

    # Deferred low-volume handling: allow only strong reversal-context setups to survive.
    if volume_gate_active:
        reversal_flags = [liquidity_event, micro_structure_event, rsi_event, macd_event, orb_event]
        reversal_structure_count = int(sum(bool(x) for x in reversal_flags))
        level_family_ok = bool(liquidity_event or orb_event or vwap_event or trigger_near)
        momentum_family_ok = bool(rsi_event or macd_event or (bias == "LONG" and (rsi_pre_long or macd_pre_long)) or (bias == "SHORT" and (rsi_pre_short or macd_pre_short)))
        structure_family_ok = bool(micro_structure_event or (bias == "LONG" and (struct_pre_long or trader_edge_pre_long)) or (bias == "SHORT" and (struct_pre_short or trader_edge_pre_short)))
        reversal_structure_ok = bool(reversal_structure_count >= 3 or (level_family_ok and momentum_family_ok and structure_family_ok))
        fav_q_for_override = float(scalp_zone_ctx.get("favorable_quality") or 0.0) if isinstance(scalp_zone_ctx, dict) else 0.0
        favorable_zone_ok = bool(isinstance(scalp_zone_ctx, dict) and scalp_zone_ctx.get("favorable") and fav_q_for_override >= 0.40)
        if favorable_zone_ok and reversal_structure_ok:
            low_volume_override = True
            base_penalty = float(5.0 * float(liquidity_mult)) if isinstance(liquidity_mult, (int, float)) else 5.0
            structure_quality = float(np.clip(reversal_structure_count / 4.0, 0.0, 1.0))
            proximity_quality = 1.0 if bool(trigger_near) else 0.0
            override_quality = float(np.clip((0.50 * fav_q_for_override) + (0.30 * structure_quality) + (0.20 * proximity_quality), 0.0, 1.0))
            penalty_scale = float(np.clip(1.15 - 0.45 * override_quality, 0.70, 1.15))
            volume_missing_penalty = int(np.clip(round(base_penalty * penalty_scale), 2, 8))
            if str(bias).upper() == "LONG":
                long_score = _cap_score(long_score - int(volume_missing_penalty))
            elif str(bias).upper() == "SHORT":
                short_score = _cap_score(short_score - int(volume_missing_penalty))
            setup_score = _cap_score(max(long_score, short_score))
            extras["volume_override_quality"] = float(round(override_quality, 4))
            extras["volume_override_structure_count"] = int(reversal_structure_count)
            reason = (reason + "; " if reason else "") + f"Low-volume override (-{int(volume_missing_penalty)})"
        else:
            extras["low_volume_override"] = False
            extras["volume_missing_penalty"] = 0
            return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_score, short_score)), "No volume confirmation", None, None, None, None, last_price, last_ts, session, extras)

    extras["low_volume_override"] = bool(low_volume_override)
    extras["volume_missing_penalty"] = int(volume_missing_penalty)

    if long_score < min_score and short_score < min_score:
        extras["decision"] = {"bias": bias, "long": long_score, "short": short_score, "min": min_score}
        neutral_reason = reason if reason else "Score below threshold"
        return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_score, short_score)), neutral_reason, None, None, None, None, last_price, last_ts, session, extras)

    # PRE tier risk tightening: smaller risk ⇒ closer TP ⇒ more hits.
    interval_mins_i = int(interval_mins) if isinstance(interval_mins, (int, float)) else 1
    pre_stop_tighten = 0.70 if stage == "PRE" else 1.0
    extras["pre_stop_tighten"] = float(pre_stop_tighten)

    if bias == "LONG":
        entry_px, stop_px = _long_entry_stop(float(entry_limit))
        if stage == "PRE":
            stop_px = float(entry_px - (entry_px - stop_px) * pre_stop_tighten)
        risk = max(1e-9, entry_px - stop_px)
        # Targeting overhaul (structure-first): TP0/TP1/TP2
        lvl_map = _candidate_levels_from_context(
            levels=levels if isinstance(levels, dict) else {},
            recent_swing_high=float(recent_swing_high),
            recent_swing_low=float(recent_swing_low),
            hi=float(hi),
            lo=float(lo),
        )
        tp0 = _pick_tp0("LONG", entry_px=entry_px, last_px=float(last_price), atr_last=float(atr_last), levels=lvl_map)
        tp1 = (tp0 + 0.9 * risk) if tp0 is not None else (entry_px + risk)
        tp2 = (tp0 + 1.8 * risk) if tp0 is not None else (entry_px + 2 * risk)
        # Optional TP3: expected excursion (rolling MFE) for similar historical signatures
        sig_key = {
            "rsi_event": bool(rsi_snap and rsi14 < 60),
            "macd_event": bool(macd_turn_up),
            "vol_event": bool(vol_ok),
            "struct_event": bool(micro_hl),
            "vol_mult": float(cfg.get("vol_multiplier", 1.25)),
        }
        tp3, tp3_diag = _tp3_from_expected_excursion(
            df, direction="LONG", signature=sig_key, entry_px=float(entry_px), interval_mins=int(interval_mins_i)
        )
        extras["tp3"] = float(tp3) if tp3 is not None else None
        extras["tp3_diag"] = tp3_diag

        # If fib extension helper is available, prefer it for pro mode.
        if pro_mode and "_fib_take_profits_long" in locals():
            f1, f2 = _fib_take_profits_long(entry_px)
            # Use fib as TP2 (runner) when it is further than our structure target.
            if f1 is not None and (tp0 is None or float(f1) > float(tp0)):
                tp1 = float(f1)
            if f2 is not None and float(f2) > float(tp1):
                tp2 = float(f2)
            extras["fib_tp1"] = float(tp1) if tp1 is not None else None
            extras["fib_tp2"] = float(tp2) if tp2 is not None else None
    else:
        entry_px, stop_px = _short_entry_stop(float(entry_limit))
        if stage == "PRE":
            stop_px = float(entry_px + (stop_px - entry_px) * pre_stop_tighten)
        risk = max(1e-9, stop_px - entry_px)
        lvl_map = _candidate_levels_from_context(
            levels=levels if isinstance(levels, dict) else {},
            recent_swing_high=float(recent_swing_high),
            recent_swing_low=float(recent_swing_low),
            hi=float(hi),
            lo=float(lo),
        )
        tp0 = _pick_tp0("SHORT", entry_px=entry_px, last_px=float(last_price), atr_last=float(atr_last), levels=lvl_map)
        tp1 = (tp0 - 0.9 * risk) if tp0 is not None else (entry_px - risk)
        tp2 = (tp0 - 1.8 * risk) if tp0 is not None else (entry_px - 2 * risk)
        sig_key = {
            "rsi_event": bool(rsi_downshift and rsi14 > 40),
            "macd_event": bool(macd_turn_down),
            "vol_event": bool(vol_ok),
            "struct_event": bool(micro_lh),
            "vol_mult": float(cfg.get("vol_multiplier", 1.25)),
        }
        tp3, tp3_diag = _tp3_from_expected_excursion(
            df, direction="SHORT", signature=sig_key, entry_px=float(entry_px), interval_mins=int(interval_mins_i)
        )
        extras["tp3"] = float(tp3) if tp3 is not None else None
        extras["tp3_diag"] = tp3_diag

        if pro_mode and "_fib_take_profits_short" in locals():
            f1, f2 = _fib_take_profits_short(entry_px)
            if f1 is not None and (tp0 is None or float(f1) < float(tp0)):
                tp1 = float(f1)
            if f2 is not None and float(f2) < float(tp1):
                tp2 = float(f2)
            extras["fib_tp1"] = float(tp1) if tp1 is not None else None
            extras["fib_tp2"] = float(tp2) if tp2 is not None else None
            extras["fib_tp1"] = float(tp1) if tp1 is not None else None
            extras["fib_tp2"] = float(tp2) if tp2 is not None else None

    # Trader-edge executable room gate: keep stops/targets, block late entries with collapsed R:R.
    try:
        if tp0 is not None and entry_px is not None and stop_px is not None:
            if str(bias).upper() == "LONG":
                _reward0 = float(tp0) - float(entry_px); _risk0 = float(entry_px) - float(stop_px)
            else:
                _reward0 = float(entry_px) - float(tp0); _risk0 = float(stop_px) - float(entry_px)
            _rr0 = float(_reward0 / max(1e-9, _risk0)); _reward_pct0 = float(_reward0 / max(1e-9, float(entry_px)))
            extras["tp0_rr"] = float(round(_rr0, 4)); extras["tp0_reward_pct"] = float(round(_reward_pct0, 5))
            _elite_early = bool(scalp_early_entry_applied or trader_edge_pre_long or trader_edge_pre_short or int(extras.get("ignition_tier_long") or 0) >= 3 or int(extras.get("ignition_tier_short") or 0) >= 3)
            _min_rr = 0.78 if _elite_early else 0.90
            _min_reward_pct = 0.0038 if _elite_early else 0.0048
            if _reward0 <= 0 or _rr0 < _min_rr or _reward_pct0 < _min_reward_pct:
                return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_score, short_score)), f"Late SCALP blocked: TP0 room too thin (RR={_rr0:.2f}, reward={_reward_pct0:.2%})", None, None, None, None, last_price, last_ts, session, extras)
    except Exception:
        pass

    # Expected time-to-TP0 UI helper
    extras["tp0"] = float(tp0) if "tp0" in locals() and tp0 is not None else None
    extras["eta_tp0_min"] = _eta_minutes_to_tp0(
        last_px=float(last_price),
        tp0=tp0 if "tp0" in locals() else None,
        atr_last=float(atr_last) if atr_last else 0.0,
        interval_mins=interval_mins_i,
        liquidity_mult=float(liquidity_mult) if "liquidity_mult" in locals() else 1.0,
    )

    extras["decision"] = {"bias": bias, "long": long_score, "short": short_score, "min": min_score}
    return SignalResult(
        symbol,
        bias,
        setup_score,
        reason,
        float(entry_px),
        float(stop_px),
        float(tp1) if tp1 is not None else None,
        float(tp2) if tp2 is not None else None,
        last_price,
        last_ts,
        session,
        extras,
    )

def _slip_amount(*, slippage_mode: str, fixed_slippage_cents: float, atr_last: float, atr_fraction_slippage: float) -> float:
    """Return slippage amount in price units (not percent)."""
    try:
        mode = (slippage_mode or "Off").strip()
    except Exception:
        mode = "Off"

    if mode == "Off":
        return 0.0

    if mode == "Fixed cents":
        try:
            return max(0.0, float(fixed_slippage_cents)) / 100.0
        except Exception:
            return 0.0

    if mode == "ATR fraction":
        try:
            return max(0.0, float(atr_last)) * max(0.0, float(atr_fraction_slippage))
        except Exception:
            return 0.0

    return 0.0
def _entry_from_model(
    direction: str,
    *,
    entry_model: str,
    last_price: float,
    ref_vwap: float | None,
    mid_price: float | None,
    atr_last: float,
    slippage_mode: str,
    fixed_slippage_cents: float,
    atr_fraction_slippage: float,
) -> float:
    """Compute an execution-realistic entry based on the selected entry model."""
    slip = _slip_amount(
        slippage_mode=slippage_mode,
        fixed_slippage_cents=fixed_slippage_cents,
        atr_last=atr_last,
        atr_fraction_slippage=atr_fraction_slippage,
    )

    model = (entry_model or "Last price").strip()

    # 1) VWAP-based: place a limit slightly beyond VWAP in the adverse direction (more realistic fills).
    if model == "VWAP reclaim limit" and isinstance(ref_vwap, (float, int)):
        return (float(ref_vwap) + slip) if direction == "LONG" else (float(ref_vwap) - slip)

    # 2) Midpoint of the last completed bar
    if model == "Midpoint (last closed bar)" and isinstance(mid_price, (float, int)):
        return (float(mid_price) + slip) if direction == "LONG" else (float(mid_price) - slip)

    # 3) Default: last price with slippage in the adverse direction
    return (float(last_price) + slip) if direction == "LONG" else (float(last_price) - slip)

# ===========================
# RIDE / Continuation signals
# ===========================

def _last_swing_level(series: pd.Series, *, kind: str, lookback: int = 60) -> float | None:
    """Return the most recent swing high/low level in the lookback window (excluding the last bar)."""
    if series is None or len(series) < 10:
        return None
    s = series.astype(float).tail(int(min(len(series), max(12, lookback))))
    flags = rolling_swing_highs(s, left=3, right=3) if kind == "high" else rolling_swing_lows(s, left=3, right=3)

    # exclude last bar (cannot be a confirmed pivot yet)
    flags = flags.iloc[:-1]
    s2 = s.iloc[:-1]

    idx = None
    for i in range(len(flags) - 1, -1, -1):
        if bool(flags.iloc[i]):
            idx = flags.index[i]
            break
    if idx is None:
        return None
    try:
        return float(s2.loc[idx])
    except Exception:
        return None


def compute_ride_signal(
    symbol: str,
    ohlcv: pd.DataFrame,
    rsi5: pd.Series,
    rsi14: pd.Series,
    macd_hist: pd.Series,
    *,
    pro_mode: bool = False,
    allow_opening: bool = True,
    allow_midday: bool = False,
    allow_power: bool = True,
    allow_premarket: bool = False,
    allow_afterhours: bool = False,
    use_last_closed_only: bool = False,
    bar_closed_guard: bool = True,
    interval: str = "1min",
    vwap_logic: str = "session",
    session_vwap_include_premarket: bool = False,
    entry_model: str = "Last price",
    slippage_mode: str = "Off",
    fixed_slippage_cents: float = 0.02,
    atr_fraction_slippage: float = 0.15,
    fib_lookback_bars: int = 200,
    killzone_preset: str = "none",
    target_atr_pct: float = 0.004,
    htf_bias: dict | None = None,
    orb_minutes: int = 15,
    liquidity_weighting: float = 0.55,
    tape_mode_enabled: bool = False,
    **_ignored: object,
) -> SignalResult:
    """Continuation / Drive signal family.

    Returns bias:
      - RIDE_LONG / RIDE_SHORT when trend + impulse/acceptance exists (actionable proximity)
      - CHOP when trend is insufficient or setup is not actionable yet
    """
    try:
        df = ohlcv.sort_index().copy()
    except Exception:
        df = ohlcv.copy()

    # interval mins
    try:
        interval_mins = int(str(interval).replace("min", "").strip())
    except Exception:
        interval_mins = 1

    # bar-closed guard (avoid partial last bar)
    df = _asof_slice(df, interval_mins, use_last_closed_only, bar_closed_guard)

    if df is None or len(df) < 60:
        return SignalResult(symbol, "CHOP", 0, "Not enough data for continuation scan.", None, None, None, None, None, None, "OFF", {"mode": "RIDE"})

    # attach indicators (aligned)
    df["rsi5"] = pd.to_numeric(rsi5.reindex(df.index).ffill(), errors="coerce")
    df["rsi14"] = pd.to_numeric(rsi14.reindex(df.index).ffill(), errors="coerce")
    df["macd_hist"] = pd.to_numeric(macd_hist.reindex(df.index).ffill(), errors="coerce")

    session = classify_session(
        df.index[-1],
        allow_opening=allow_opening,
        allow_midday=allow_midday,
        allow_power=allow_power,
        allow_premarket=allow_premarket,
        allow_afterhours=allow_afterhours,
    )
    liquidity_phase = classify_liquidity_phase(df.index[-1])
    raw_session = str(liquidity_phase).upper()

    # RIDE trader-edge liquidity weighting: session-aware like SCALP.
    try:
        _lw = float(np.clip(float(liquidity_weighting), 0.0, 1.0))
    except Exception:
        _lw = 0.55
    _liq_base = 1.0
    if raw_session in ("OPENING", "POWER"):
        _liq_base = 1.15
    elif raw_session == "MIDDAY":
        _liq_base = 0.85
    elif raw_session in ("PREMARKET", "AFTERHOURS"):
        _liq_base = 0.75
    liquidity_mult = float(np.clip(1.0 + _lw * (_liq_base - 1.0), 0.75, 1.18))

    last_ts = pd.to_datetime(df.index[-1])
    last_bar_price = float(df["close"].iloc[-1])
    try:
        live_last_price = float(_ignored.get("live_last_price")) if _ignored.get("live_last_price") is not None else float("nan")
    except Exception:
        live_last_price = float("nan")
    last_price = float(live_last_price) if np.isfinite(live_last_price) and float(live_last_price) > 0 else float(last_bar_price)

    # VWAP reference
    vwap_sess = calc_session_vwap(df, include_premarket=session_vwap_include_premarket, include_afterhours=allow_afterhours)
    vwap_cum = calc_vwap(df)
    ref_vwap_series = vwap_sess if str(vwap_logic).lower() == "session" else vwap_cum
    ref_vwap = float(ref_vwap_series.iloc[-1]) if len(ref_vwap_series) else None

    # ATR + trend stats
    atr_s = calc_atr(df, period=14).reindex(df.index).ffill()
    atr_last = float(atr_s.iloc[-1]) if len(atr_s) else None
    if atr_last is None or not np.isfinite(atr_last) or atr_last <= 0:
        atr_last = max(1e-6, float(df["high"].iloc[-10:].max() - df["low"].iloc[-10:].min()) / 10.0)

    close = df["close"].astype(float)
    ema20 = calc_ema(close, 20)
    ema50 = calc_ema(close, 50)
    adx, di_plus, di_minus = calc_adx(df, period=14)

    adx_ff = adx.reindex(df.index).ffill() if len(adx) else pd.Series(dtype=float)
    di_plus_ff = di_plus.reindex(df.index).ffill() if len(di_plus) else pd.Series(dtype=float)
    di_minus_ff = di_minus.reindex(df.index).ffill() if len(di_minus) else pd.Series(dtype=float)
    adx_ctx = calc_adx_context(adx_ff, di_plus_ff, di_minus_ff)
    try:
        df["adx14"] = adx_ff
        df["plus_di14"] = di_plus_ff
        df["minus_di14"] = di_minus_ff
    except Exception:
        pass
    ride_pressure_states = _indicator_pressure_states(df, adx_ctx)
    adx_last = float(adx_ctx.get("adx")) if adx_ctx.get("adx") is not None else float("nan")
    di_p = float(adx_ctx.get("plus_di")) if adx_ctx.get("plus_di") is not None else float("nan")
    di_m = float(adx_ctx.get("minus_di")) if adx_ctx.get("minus_di") is not None else float("nan")

    def _ride_adx_modifier(ctx: dict[str, object]) -> tuple[float, str | None]:
        try:
            last = float(ctx.get("adx")) if ctx.get("adx") is not None else float("nan")
        except Exception:
            last = float("nan")
        if not np.isfinite(last):
            return 0.0, None
        regime = str(ctx.get("regime") or "unknown")
        slope = float(ctx.get("adx_slope") or 0.0)
        spread = float(ctx.get("di_spread") or 0.0)
        dom_bars = int(ctx.get("dominance_bars") or 0)

        if regime == "dead_chop":
            mod = -6.0
        elif regime == "coiling":
            mod = -2.0 if slope <= 0.5 else 0.0
        elif regime == "emerging":
            mod = 2.0 if spread >= 5.0 else 0.5
        elif regime == "strengthening":
            mod = 5.0 if spread >= 6.0 else 3.0
        elif regime == "healthy_trend":
            mod = 4.0 if dom_bars >= 2 else 3.0
        elif regime == "mature_trend":
            mod = 1.0 if slope >= 0.0 else -1.0
        elif regime == "exhausting":
            mod = -4.0
        else:
            mod = 0.0

        if slope > 1.25 and regime in ("emerging", "strengthening", "healthy_trend"):
            mod += 1.0
        elif slope < -1.0 and regime in ("mature_trend", "healthy_trend", "exhausting"):
            mod -= 1.0

        mod = float(np.clip(mod, -7.0, 8.0))
        note = None
        if mod > 0:
            note = f"ADX {regime.replace('_', ' ')} tailwind (+{mod:.0f})"
        elif mod < 0:
            note = f"ADX {regime.replace('_', ' ')} headwind ({mod:.0f})"
        return mod, note

    def _ride_adx_gate(ctx: dict[str, object]) -> tuple[bool, bool, str | None]:
        regime = str(ctx.get("regime") or "unknown")
        slope = float(ctx.get("adx_slope") or 0.0)
        spread = float(ctx.get("di_spread") or 0.0)
        dom_bars = int(ctx.get("dominance_bars") or 0)

        constructive_regime = regime in ("emerging", "strengthening", "healthy_trend")
        context_adx_ok = bool(
            constructive_regime
            and (
                (spread >= max(4.0, di_gap_floor - 1.0) and slope >= 0.35)
                or (spread >= max(5.0, di_gap_floor) and dom_bars >= 2)
                or regime in ("strengthening", "healthy_trend")
            )
        )
        context_di_ok = bool(
            constructive_regime
            and (
                spread >= max(4.0, di_gap_floor - 1.0)
                or (spread >= 3.5 and slope >= 0.75 and dom_bars >= 1)
            )
        )
        note = None
        if (context_adx_ok or context_di_ok) and not (pass_adx and pass_di_gap):
            note = f"ADX {regime.replace('_', ' ')} context salvaged early trend gate"
        return context_adx_ok, context_di_ok, note

    adx_modifier, adx_modifier_note = _ride_adx_modifier(adx_ctx)

    adx_floor = 20.0 if interval_mins <= 1 else 18.0
    di_gap_floor = 6.0 if interval_mins <= 1 else 5.0

    pass_adx = bool(np.isfinite(adx_last) and adx_last >= adx_floor)
    pass_di_gap = bool(np.isfinite(di_p) and np.isfinite(di_m) and abs(di_p - di_m) >= di_gap_floor)
    pass_ema_up = bool(float(ema20.iloc[-1]) > float(ema50.iloc[-1]))
    pass_ema_dn = bool(float(ema20.iloc[-1]) < float(ema50.iloc[-1]))

    context_adx_ok, context_di_ok, adx_gate_note = _ride_adx_gate(adx_ctx)
    adx_gate_pass = bool(pass_adx or context_adx_ok)
    di_gate_pass = bool(pass_di_gap or context_di_ok)

    regime = str(adx_ctx.get("regime") or "unknown")
    adx_slope = float(adx_ctx.get("adx_slope") or 0.0)
    di_spread = float(adx_ctx.get("di_spread") or 0.0)
    dom_side = str(adx_ctx.get("dominant_side") or "NONE").upper()
    dom_bars = int(adx_ctx.get("dominance_bars") or 0)
    constructive_regime = regime in ("emerging", "strengthening", "healthy_trend", "mature_trend")
    strong_regime = regime in ("strengthening", "healthy_trend")

    long_di_align = bool(np.isfinite(di_p) and np.isfinite(di_m) and ((di_p > di_m) or (dom_side == "LONG" and di_spread >= max(3.5, di_gap_floor - 1.5))))
    short_di_align = bool(np.isfinite(di_p) and np.isfinite(di_m) and ((di_m > di_p) or (dom_side == "SHORT" and di_spread >= max(3.5, di_gap_floor - 1.5))))

    long_adx_align = bool(
        adx_gate_pass
        and constructive_regime
        and regime != "exhausting"
        and (long_di_align or dom_side == "LONG" or (di_spread >= max(4.0, di_gap_floor - 1.0) and adx_slope >= 0.35))
    )
    short_adx_align = bool(
        adx_gate_pass
        and constructive_regime
        and regime != "exhausting"
        and (short_di_align or dom_side == "SHORT" or (di_spread >= max(4.0, di_gap_floor - 1.0) and adx_slope >= 0.35))
    )

    long_trend_votes = int(long_adx_align) + int(di_gate_pass and long_di_align) + int(pass_ema_up)
    short_trend_votes = int(short_adx_align) + int(di_gate_pass and short_di_align) + int(pass_ema_dn)

    long_trend_ok = bool(
        long_trend_votes >= 2
        or (
            long_trend_votes >= 1
            and strong_regime
            and long_di_align
            and dom_bars >= 2
            and di_spread >= max(5.0, di_gap_floor - 0.5)
        )
    )
    short_trend_ok = bool(
        short_trend_votes >= 2
        or (
            short_trend_votes >= 1
            and strong_regime
            and short_di_align
            and dom_bars >= 2
            and di_spread >= max(5.0, di_gap_floor - 0.5)
        )
    )

    trend_votes = max(long_trend_votes, short_trend_votes)
    trend_ok = bool(long_trend_ok or short_trend_ok)

    if not trend_ok:
        return SignalResult(
            symbol=symbol,
            bias="CHOP",
            setup_score=0,
            reason=f"Too choppy for RIDE (dir trend L={long_trend_votes}/3, S={short_trend_votes}/3).",
            entry=None, stop=None, target_1r=None, target_2r=None,
            last_price=last_price, timestamp=last_ts, session=session,
            extras={"mode": "RIDE", "stage": None, "trend_votes": trend_votes, "long_trend_votes": long_trend_votes, "short_trend_votes": short_trend_votes, "long_trend_ok": long_trend_ok, "short_trend_ok": short_trend_ok, "adx": adx_last, "di_plus": di_p, "di_minus": di_m, "liquidity_phase": liquidity_phase, "adx_gate_note": adx_gate_note},
        )

    # ORB / pivots / displacement
    levels = _session_liquidity_levels(df, interval_mins, orb_minutes)
    orb_high = levels.get("orb_high")
    orb_low = levels.get("orb_low")
    buffer = 0.15 * float(atr_last)

    orb_seq = _orb_three_stage(df, orb_high=orb_high, orb_low=orb_low, buffer=buffer, lookback_bars=60, accept_bars=2)
    swing_hi = _last_swing_level(df["high"], kind="high", lookback=60)
    swing_lo = _last_swing_level(df["low"], kind="low", lookback=60)

    # --- Impulse legitimacy inputs ---
    # We want more than a retail-visible line break. Reward sequence quality,
    # held reclaims, and breaks that emerge from compression/sweep context.
    close_s = df["close"].astype(float)
    high_s = df["high"].astype(float)
    low_s = df["low"].astype(float)
    open_s = df["open"].astype(float)

    last_range = float(high_s.iloc[-1] - low_s.iloc[-1])
    disp_ratio = float(last_range / max(1e-9, float(atr_last)))
    disp_ok = disp_ratio >= 1.2
    prev_close = float(close_s.iloc[-2])
    prev_low_min = float(low_s.iloc[-6:-1].min()) if len(low_s) >= 6 else float(low_s.iloc[:-1].min())
    prev_high_max = float(high_s.iloc[-6:-1].max()) if len(high_s) >= 6 else float(high_s.iloc[:-1].max())
    prior_rng = float(high_s.tail(6).max() - low_s.tail(6).min()) if len(df) >= 6 else float(last_range)
    compression_ok = bool(prior_rng <= 1.35 * float(atr_last))
    body_ratio = float(abs(close_s.iloc[-1] - open_s.iloc[-1]) / max(1e-9, last_range))
    close_pos = float((close_s.iloc[-1] - low_s.iloc[-1]) / max(1e-9, last_range))
    close_q_long = float(np.clip(close_pos, 0.0, 1.0))
    close_q_short = float(np.clip(1.0 - close_pos, 0.0, 1.0))

    # VWAP reclaim/reject legitimacy: confirm from completed bars; keep quote for
    # downstream tradability only. This prevents a transient live print from
    # manufacturing a reclaim/reject that the last closed bar has not confirmed.
    confirm_close = float(close_s.iloc[-1])
    vwap_reclaim_cross = bool(ref_vwap is not None and prev_close <= ref_vwap and confirm_close > ref_vwap and disp_ok)
    vwap_reject_cross = bool(ref_vwap is not None and prev_close >= ref_vwap and confirm_close < ref_vwap and disp_ok)
    if ref_vwap is not None and len(close_s) >= 2:
        vwap_reclaim_hold = bool((close_s.tail(2) > float(ref_vwap) - 0.10 * buffer).all())
        vwap_reject_hold = bool((close_s.tail(2) < float(ref_vwap) + 0.10 * buffer).all())
    else:
        vwap_reclaim_hold = False
        vwap_reject_hold = False
    swept_low_then_reclaim = bool(ref_vwap is not None and low_s.iloc[-1] < (prev_low_min - 0.05 * atr_last) and confirm_close > ref_vwap and close_q_long >= 0.55)
    swept_high_then_reject = bool(ref_vwap is not None and high_s.iloc[-1] > (prev_high_max + 0.05 * atr_last) and confirm_close < ref_vwap and close_q_short >= 0.55)
    vwap_reclaim = bool(vwap_reclaim_cross and (vwap_reclaim_hold or swept_low_then_reclaim))
    vwap_reject = bool(vwap_reject_cross and (vwap_reject_hold or swept_high_then_reject))
    vwap_score_up = 0.0
    vwap_score_dn = 0.0
    if vwap_reclaim_cross:
        vwap_score_up += 0.45
    if vwap_reclaim_hold:
        vwap_score_up += 0.30
    if swept_low_then_reclaim:
        vwap_score_up += 0.25
    if vwap_reject_cross:
        vwap_score_dn += 0.45
    if vwap_reject_hold:
        vwap_score_dn += 0.30
    if swept_high_then_reject:
        vwap_score_dn += 0.25

    # Pivot legitimacy: confirm break/hold from completed bars; current quote can
    # still decide whether the setup remains tradable later in the pipeline.
    pivot_break_up = bool(swing_hi is not None and confirm_close > float(swing_hi) + buffer)
    pivot_break_dn = bool(swing_lo is not None and confirm_close < float(swing_lo) - buffer)
    pivot_hold_up = bool(swing_hi is not None and len(close_s) >= 2 and (close_s.tail(2) > float(swing_hi) + 0.05 * buffer).all())
    pivot_hold_dn = bool(swing_lo is not None and len(close_s) >= 2 and (close_s.tail(2) < float(swing_lo) - 0.05 * buffer).all())
    pivot_score_up = (0.45 if pivot_break_up else 0.0) + (0.30 if pivot_hold_up else 0.0) + (0.25 if (pivot_break_up and compression_ok and close_q_long >= 0.60) else 0.0)
    pivot_score_dn = (0.45 if pivot_break_dn else 0.0) + (0.30 if pivot_hold_dn else 0.0) + (0.25 if (pivot_break_dn and compression_ok and close_q_short >= 0.60) else 0.0)

    # ORB legitimacy: sequence beats break-only, but break confirmation stays bar-based.
    orb_break_up = bool(orb_high is not None and orb_seq.get("bull_break") and confirm_close > float(orb_high) + buffer)
    orb_break_dn = bool(orb_low is not None and orb_seq.get("bear_break") and confirm_close < float(orb_low) - buffer)
    orb_accept_up = bool(orb_high is not None and orb_break_up and len(close_s) >= 2 and (close_s.tail(2) > float(orb_high) + 0.05 * buffer).all())
    orb_accept_dn = bool(orb_low is not None and orb_break_dn and len(close_s) >= 2 and (close_s.tail(2) < float(orb_low) - 0.05 * buffer).all())
    orb_retest_up = bool(orb_seq.get("bull_orb_seq"))
    orb_retest_dn = bool(orb_seq.get("bear_orb_seq"))
    orb_score_up = (0.35 if orb_break_up else 0.0) + (0.30 if orb_accept_up else 0.0) + (0.35 if orb_retest_up else 0.0)
    orb_score_dn = (0.35 if orb_break_dn else 0.0) + (0.30 if orb_accept_dn else 0.0) + (0.35 if orb_retest_dn else 0.0)

    impulse_scores_long = {"ORB": float(np.clip(orb_score_up, 0.0, 1.0)), "PIVOT": float(np.clip(pivot_score_up, 0.0, 1.0)), "VWAP": float(np.clip(vwap_score_up, 0.0, 1.0))}
    impulse_scores_short = {"ORB": float(np.clip(orb_score_dn, 0.0, 1.0)), "PIVOT": float(np.clip(pivot_score_dn, 0.0, 1.0)), "VWAP": float(np.clip(vwap_score_dn, 0.0, 1.0))}
    long_best_type = max(impulse_scores_long, key=impulse_scores_long.get)
    short_best_type = max(impulse_scores_short, key=impulse_scores_short.get)
    long_legitimacy = float(impulse_scores_long.get(long_best_type, 0.0))
    short_legitimacy = float(impulse_scores_short.get(short_best_type, 0.0))

    long_legit_trigger = bool((orb_score_up > 0.45) or (pivot_score_up > 0.45) or vwap_reclaim)
    short_legit_trigger = bool((orb_score_dn > 0.45) or (pivot_score_dn > 0.45) or vwap_reject)
    impulse_long = bool(long_legit_trigger and long_legitimacy >= 0.45)
    impulse_short = bool(short_legit_trigger and short_legitimacy >= 0.45)

    if not impulse_long and not impulse_short:
        return SignalResult(
            symbol=symbol,
            bias="CHOP",
            setup_score=0,
            reason="Trend present but no legitimate impulse/drive signature yet.",
            entry=None, stop=None, target_1r=None, target_2r=None,
            last_price=last_price, timestamp=last_ts, session=session,
            extras={"mode": "RIDE", "stage": None, "trend_votes": trend_votes, "liquidity_phase": liquidity_phase,
                    "long_legitimacy": long_legitimacy, "short_legitimacy": short_legitimacy},
        )

    # Direction-selection refinement:
    # when both sides print local impulse signatures, do not choose mostly on
    # local legitimacy alone. Weight the choice by aligned trend support so a
    # flashy micro-break against healthier broader control is less likely to win.
    long_trend_context = float(
        0.08 * float(long_trend_votes)
        + (0.07 if long_trend_ok else 0.0)
        + (0.06 if long_di_align else 0.0)
        + (0.04 if dom_side == "LONG" else 0.0)
        + (0.04 if pass_ema_up else 0.0)
        + (0.03 if constructive_regime and regime != "exhausting" else 0.0)
        + (0.03 if (adx_slope >= 0.20 and di_spread >= max(4.0, di_gap_floor - 1.0)) else 0.0)
        + (0.02 if dom_bars >= 2 else 0.0)
    )
    short_trend_context = float(
        0.08 * float(short_trend_votes)
        + (0.07 if short_trend_ok else 0.0)
        + (0.06 if short_di_align else 0.0)
        + (0.04 if dom_side == "SHORT" else 0.0)
        + (0.04 if pass_ema_dn else 0.0)
        + (0.03 if constructive_regime and regime != "exhausting" else 0.0)
        + (0.03 if (adx_slope >= 0.20 and di_spread >= max(4.0, di_gap_floor - 1.0)) else 0.0)
        + (0.02 if dom_bars >= 2 else 0.0)
    )

    direction = None
    if impulse_long and not impulse_short:
        direction = "LONG" if long_trend_ok else None
    elif impulse_short and not impulse_long:
        direction = "SHORT" if short_trend_ok else None
    else:
        if long_trend_ok and not short_trend_ok:
            direction = "LONG"
        elif short_trend_ok and not long_trend_ok:
            direction = "SHORT"
        else:
            long_total = float(long_legitimacy) + float(long_trend_context)
            short_total = float(short_legitimacy) + float(short_trend_context)
            # Small directional bias from current DI edge only as a final nudge.
            dir_edge = float((di_p if np.isfinite(di_p) else 0.0) - (di_m if np.isfinite(di_m) else 0.0))
            if dir_edge >= 0:
                long_total += 0.02
            if dir_edge <= 0:
                short_total += 0.02
            direction = "LONG" if long_total >= short_total else "SHORT"

    if direction is None:
        return SignalResult(
            symbol=symbol,
            bias="CHOP",
            setup_score=0,
            reason="Impulse present but direction lacks aligned trend support.",
            entry=None, stop=None, target_1r=None, target_2r=None,
            last_price=last_price, timestamp=last_ts, session=session,
            extras={"mode": "RIDE", "stage": None, "trend_votes": trend_votes, "long_trend_votes": long_trend_votes, "short_trend_votes": short_trend_votes, "long_trend_ok": long_trend_ok, "short_trend_ok": short_trend_ok, "long_legitimacy": long_legitimacy, "short_legitimacy": short_legitimacy, "liquidity_phase": liquidity_phase},
        )

    impulse_legitimacy = long_legitimacy if direction == "LONG" else short_legitimacy
    impulse_type_hint = long_best_type if direction == "LONG" else short_best_type

    # Adaptive RIDE acceptance line:
    # weight structural anchors by source confidence, but also by *recently defended*
    # interaction quality so stale ORB/pivot references fade when VWAP/EMA20 becomes
    # the line the market is actually holding now.
    accept_components: Dict[str, float] = {}
    accept_component_weights: Dict[str, float] = {}
    accept_recent_diag: Dict[str, Dict[str, float | int]] = {}
    try:
        ema20_ref = float(ema20.iloc[-1]) if np.isfinite(ema20.iloc[-1]) else None
    except Exception:
        ema20_ref = None

    if direction == "LONG":
        if ref_vwap is not None and np.isfinite(ref_vwap):
            accept_components["VWAP"] = float(ref_vwap)
            rx = _anchor_recent_interaction_score(df, direction="LONG", anchor=float(ref_vwap), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
            accept_recent_diag["VWAP"] = rx
            accept_component_weights["VWAP"] = max(0.05, 0.30 + 0.55 * float(np.clip(vwap_score_up, 0.0, 1.0)) + 0.20 * float(vwap_reclaim_hold) - 0.12 * float(vwap_reclaim_cross and not vwap_reclaim_hold) + 0.32 * float(rx.get("score") or 0.0))
        if orb_high is not None and np.isfinite(orb_high):
            accept_components["ORB"] = float(orb_high)
            rx = _anchor_recent_interaction_score(df, direction="LONG", anchor=float(orb_high), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
            accept_recent_diag["ORB"] = rx
            accept_component_weights["ORB"] = max(0.05, 0.28 + 0.55 * float(np.clip(orb_score_up, 0.0, 1.0)) + 0.18 * float(orb_accept_up) + 0.14 * float(orb_retest_up) - 0.14 * float(orb_break_up and not orb_accept_up) + 0.28 * float(rx.get("score") or 0.0))
        if swing_hi is not None and np.isfinite(swing_hi):
            accept_components["PIVOT"] = float(swing_hi)
            rx = _anchor_recent_interaction_score(df, direction="LONG", anchor=float(swing_hi), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
            accept_recent_diag["PIVOT"] = rx
            accept_component_weights["PIVOT"] = max(0.05, 0.24 + 0.55 * float(np.clip(pivot_score_up, 0.0, 1.0)) + 0.18 * float(pivot_hold_up) - 0.12 * float(pivot_break_up and not pivot_hold_up) + 0.26 * float(rx.get("score") or 0.0))
        if ema20_ref is not None and np.isfinite(ema20_ref):
            accept_components["EMA20"] = float(ema20_ref)
            rx = _anchor_recent_interaction_score(df, direction="LONG", anchor=float(ema20_ref), atr_last=float(atr_last) if atr_last is not None else None, lookback=6)
            accept_recent_diag["EMA20"] = rx
            accept_component_weights["EMA20"] = 0.10 + 0.08 * float(trend_votes >= 2) + 0.18 * float(rx.get("score") or 0.0)
    else:
        if ref_vwap is not None and np.isfinite(ref_vwap):
            accept_components["VWAP"] = float(ref_vwap)
            rx = _anchor_recent_interaction_score(df, direction="SHORT", anchor=float(ref_vwap), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
            accept_recent_diag["VWAP"] = rx
            accept_component_weights["VWAP"] = max(0.05, 0.30 + 0.55 * float(np.clip(vwap_score_dn, 0.0, 1.0)) + 0.20 * float(vwap_reject_hold) - 0.12 * float(vwap_reject_cross and not vwap_reject_hold) + 0.32 * float(rx.get("score") or 0.0))
        if orb_low is not None and np.isfinite(orb_low):
            accept_components["ORB"] = float(orb_low)
            rx = _anchor_recent_interaction_score(df, direction="SHORT", anchor=float(orb_low), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
            accept_recent_diag["ORB"] = rx
            accept_component_weights["ORB"] = max(0.05, 0.28 + 0.55 * float(np.clip(orb_score_dn, 0.0, 1.0)) + 0.18 * float(orb_accept_dn) + 0.14 * float(orb_retest_dn) - 0.14 * float(orb_break_dn and not orb_accept_dn) + 0.28 * float(rx.get("score") or 0.0))
        if swing_lo is not None and np.isfinite(swing_lo):
            accept_components["PIVOT"] = float(swing_lo)
            rx = _anchor_recent_interaction_score(df, direction="SHORT", anchor=float(swing_lo), atr_last=float(atr_last) if atr_last is not None else None, lookback=8)
            accept_recent_diag["PIVOT"] = rx
            accept_component_weights["PIVOT"] = max(0.05, 0.24 + 0.55 * float(np.clip(pivot_score_dn, 0.0, 1.0)) + 0.18 * float(pivot_hold_dn) - 0.12 * float(pivot_break_dn and not pivot_hold_dn) + 0.26 * float(rx.get("score") or 0.0))
        if ema20_ref is not None and np.isfinite(ema20_ref):
            accept_components["EMA20"] = float(ema20_ref)
            rx = _anchor_recent_interaction_score(df, direction="SHORT", anchor=float(ema20_ref), atr_last=float(atr_last) if atr_last is not None else None, lookback=6)
            accept_recent_diag["EMA20"] = rx
            accept_component_weights["EMA20"] = 0.10 + 0.08 * float(trend_votes >= 2) + 0.18 * float(rx.get("score") or 0.0)

    if accept_components:
        valid_keys = [
            k for k in accept_components
            if np.isfinite(accept_components.get(k, float("nan")))
            and np.isfinite(accept_component_weights.get(k, float("nan")))
            and float(accept_component_weights.get(k, 0.0)) > 0.0
        ]
        if valid_keys:
            anchors = np.asarray([accept_components[k] for k in valid_keys], dtype=float)
            weights = np.asarray([max(0.05, accept_component_weights[k]) for k in valid_keys], dtype=float)
            accept_line = float(np.average(anchors, weights=weights))
            if len(valid_keys) == 1:
                accept_src = str(valid_keys[0])
            else:
                ranked = sorted(
                    ((str(k), float(max(0.05, accept_component_weights.get(k, 0.0)))) for k in valid_keys),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
                total_w = float(sum(w for _, w in ranked)) or 1.0
                top1_k, top1_w = ranked[0]
                top2_k, top2_w = ranked[1]
                top1_frac = top1_w / total_w
                top2_frac = top2_w / total_w
                gap_frac = top1_frac - top2_frac
                if top1_frac >= 0.56 and gap_frac >= 0.12:
                    accept_src = top1_k
                elif top1_frac >= 0.42 and top2_frac >= 0.25 and gap_frac <= 0.12:
                    accept_src = f"{top1_k}+{top2_k}"
                else:
                    accept_src = "BLEND"
        else:
            accept_line = float(ema20_ref if isinstance(ema20_ref, (float, int)) and np.isfinite(ema20_ref) else last_price)
            accept_src = "EMA20"
    else:
        accept_line = float(ema20_ref if isinstance(ema20_ref, (float, int)) and np.isfinite(ema20_ref) else last_price)
        accept_src = "EMA20"

    # --- Acceptance / retest logic ---
    # NOTE: In live tape, ORB/VWAP levels can be *valid* but still too far from price
    # to be a realistic pullback entry for a continuation scalp.
    #
    # Example: price is actionable via break trigger proximity, but the selected
    # accept line sits far away (stale ORB from earlier in the session). In these
    # cases we still want to surface the breakout plan, and we clamp the accept
    # line used for pullback bands into a sane ATR window around last.
    accept_line_raw = float(accept_line)
    accept_dist_atr_raw = abs(float(last_price) - float(accept_line_raw)) / max(1e-9, float(atr_last))
    if regime in ("healthy_trend", "strengthening") and float(impulse_legitimacy) >= 0.72 and dom_bars >= 2:
        accept_max_mult = 1.55
    elif regime in ("mature_trend", "exhausting") or float(impulse_legitimacy) < 0.45:
        accept_max_mult = 1.00
    else:
        accept_max_mult = 1.20
    accept_min_mult = 0.05
    if direction == "LONG":
        # accept line should be below last, but not absurdly far.
        lo = float(last_price - accept_max_mult * atr_last)
        hi = float(last_price - accept_min_mult * atr_last)
        accept_line = float(np.clip(accept_line_raw, lo, hi))
    else:
        # accept line should be above last, but not absurdly far.
        lo = float(last_price + accept_min_mult * atr_last)
        hi = float(last_price + accept_max_mult * atr_last)
        accept_line = float(np.clip(accept_line_raw, lo, hi))
    accept_dist_atr_clamped = abs(float(last_price) - float(accept_line)) / max(1e-9, float(atr_last))
    accept_clamp_delta_atr = abs(float(accept_line_raw) - float(accept_line)) / max(1e-9, float(atr_last))
    accept_line_synthetic = bool(accept_clamp_delta_atr >= 0.35)
    # Accept = closes remain on the correct side of the accept line.
    look = int(min(3, len(df) - 1))
    recent_closes = df["close"].astype(float).iloc[-look:]
    if direction == "LONG":
        accept_ok = bool((recent_closes > float(accept_line) - buffer).all())
    else:
        accept_ok = bool((recent_closes < float(accept_line) + buffer).all())

    # Retest/hold = within the last few bars, price *tests* the accept line band and holds.
    retest_look = int(min(6, len(df) - 1))
    recent_lows = df["low"].astype(float).iloc[-retest_look:]
    recent_highs = df["high"].astype(float).iloc[-retest_look:]
    if direction == "LONG":
        retest_seen = bool((recent_lows <= float(accept_line) + buffer).any())
        hold_ok = bool((recent_closes >= float(accept_line) - buffer).all())
    else:
        retest_seen = bool((recent_highs >= float(accept_line) - buffer).any())
        hold_ok = bool((recent_closes <= float(accept_line) + buffer).all())

    stage = "CONFIRMED" if (accept_ok and retest_seen and hold_ok) else "PRE"

    # Early structural preview: use the current accept line plus a provisional
    # break trigger so displacement / rideability can be shaped by context
    # without waiting for the later full geometry build.
    preview_break_trigger = float('nan')
    phase_preview_info: Dict[str, object] = {
        'route_phase': 'UNSET',
        'detail_phase': 'UNSET',
        'confidence': 0.0,
        'interpretation': 'No clean continuation phase established',
    }
    phase_preview_detail = 'UNSET'
    phase_preview_conf = 0.0
    try:
        if direction == "LONG":
            preview_break_trigger = float(max(float(impulse_level), float(df["high"].iloc[impulse_idx]))) if impulse_idx is not None else float(impulse_level)
        else:
            preview_break_trigger = float(min(float(impulse_level), float(df["low"].iloc[impulse_idx]))) if impulse_idx is not None else float(impulse_level)
    except Exception:
        preview_break_trigger = float(last_price)
    try:
        phase_preview_info = _classify_ride_structure_phase_info(
            direction=str(direction),
            df=df,
            accept_line=float(accept_line),
            break_trigger=float(preview_break_trigger),
            atr_last=float(atr_last) if atr_last is not None else None,
        )
        phase_preview_detail = str(phase_preview_info.get('detail_phase') or 'UNSET')
        phase_preview_conf = float(phase_preview_info.get('confidence') or 0.0)
    except Exception:
        phase_preview_info = {
            'route_phase': 'UNSET',
            'detail_phase': 'UNSET',
            'confidence': 0.0,
            'interpretation': 'No clean continuation phase established',
        }
        phase_preview_detail = 'UNSET'
        phase_preview_conf = 0.0

    # volume pattern: directional impulse participation + hold compression
    # For volatile $1-$5 names we want to distinguish:
    # - loud but sloppy volume
    # - directional, controlled participation that can actually carry
    vol = df["volume"].astype(float)
    med30 = float(vol.tail(60).rolling(30).median().iloc[-1]) if len(vol) >= 30 else float(vol.median())
    vol_impulse = float(vol.iloc[-1])
    vol_hold = float(vol.tail(3).mean()) if len(vol) >= 3 else vol_impulse
    base_relvol_ok = bool(med30 > 0 and (vol_impulse >= 1.5 * med30))
    hold_compression_ok = bool(vol_impulse > 0 and (vol_hold <= 1.12 * vol_impulse))
    try:
        price_med30 = float(close_s.tail(60).rolling(30).median().iloc[-1]) if len(close_s) >= 30 else float(close_s.median())
    except Exception:
        price_med30 = float(last_price)
    dollar_flow = float(vol_impulse * max(0.01, last_price))
    dollar_flow_ref = float(max(1.0, med30 * max(0.01, price_med30)))
    dollar_flow_ok = bool(dollar_flow >= 0.85 * dollar_flow_ref)
    upper_wick_frac = float((high_s.iloc[-1] - max(open_s.iloc[-1], close_s.iloc[-1])) / max(1e-9, last_range))
    lower_wick_frac = float((min(open_s.iloc[-1], close_s.iloc[-1]) - low_s.iloc[-1]) / max(1e-9, last_range))
    bull_participation = bool(close_s.iloc[-1] >= open_s.iloc[-1] and body_ratio >= 0.42 and close_q_long >= 0.60 and upper_wick_frac <= 0.30)
    bear_participation = bool(close_s.iloc[-1] <= open_s.iloc[-1] and body_ratio >= 0.42 and close_q_short >= 0.60 and lower_wick_frac <= 0.30)
    hold_tail = int(min(4, len(df)))
    hold_closes = close_s.tail(hold_tail)
    hold_highs = high_s.tail(hold_tail)
    hold_lows = low_s.tail(hold_tail)
    if direction == "LONG":
        hold_progress_ok = bool((hold_closes >= float(accept_line) - 0.10 * float(atr_last)).all())
        hold_reject_wick = float((hold_highs - hold_closes).clip(lower=0.0).mean() / max(1e-9, float(atr_last)))
        hold_structure_quality = float(np.clip((0.65 * float(hold_progress_ok) + 0.35 * float(hold_reject_wick <= 0.22)), 0.0, 1.0))
    else:
        hold_progress_ok = bool((hold_closes <= float(accept_line) + 0.10 * float(atr_last)).all())
        hold_reject_wick = float((hold_closes - hold_lows).clip(lower=0.0).mean() / max(1e-9, float(atr_last)))
        hold_structure_quality = float(np.clip((0.65 * float(hold_progress_ok) + 0.35 * float(hold_reject_wick <= 0.22)), 0.0, 1.0))
    vol_ok_long = bool(base_relvol_ok and hold_compression_ok and dollar_flow_ok and bull_participation and hold_structure_quality >= 0.60)
    vol_ok_short = bool(base_relvol_ok and hold_compression_ok and dollar_flow_ok and bear_participation and hold_structure_quality >= 0.60)
    vol_ok = bool(vol_ok_long) if direction == "LONG" else bool(vol_ok_short)

    # exhaustion guard
    r5 = float(df["rsi5"].iloc[-1]) if np.isfinite(df["rsi5"].iloc[-1]) else None
    r14 = float(df["rsi14"].iloc[-1]) if np.isfinite(df["rsi14"].iloc[-1]) else None
    exhausted = False
    if direction == "LONG" and r5 is not None and r14 is not None:
        exhausted = bool(r5 > 85 and r14 > 70)
    if direction == "SHORT" and r5 is not None and r14 is not None:
        exhausted = bool(r5 < 15 and r14 < 30)

    # RSI rideability context (not a trigger, a realism guard):
    # - Continuations are best when short-term momentum is strong *but not blown out*.
    # - We use RSI-5 for timing and RSI-14 as a validation/backdrop.
    rsi_q = 0.5
    try:
        if r5 is not None and r14 is not None:
            if direction == "LONG":
                # Ideal: RSI-5 45..78 with RSI-14 >= ~45.
                base = 1.0 if (45.0 <= r5 <= 78.0 and r14 >= 45.0) else 0.6
                # If it's very hot, require pullback/retest; otherwise reduce quality.
                if r5 >= 85.0 and r14 >= 70.0:
                    base = 0.2
                rsi_q = float(np.clip(base, 0.0, 1.0))
            else:
                base = 1.0 if (22.0 <= r5 <= 55.0 and r14 <= 55.0) else 0.6
                if r5 <= 15.0 and r14 <= 30.0:
                    base = 0.2
                rsi_q = float(np.clip(base, 0.0, 1.0))
    except Exception:
        rsi_q = 0.5

    ride_macd_info = _classify_macd_momentum_state(
        df.get('macd_hist'),
        atr_last=float(atr_last) if atr_last is not None else None,
        direction=str(direction),
    )

    # --- Impulse/Hold quality score (0..1) ---
    # We want Score 100 to *mean* something tradeable:
    #   - displacement strength
    #   - close in the direction of travel
    #   - impulse volume expansion + hold compression
    #   - (for CONFIRMED) accept+retest/hold quality
    try:
        close_pos = (float(df["close"].iloc[-1]) - float(df["low"].iloc[-1])) / max(1e-9, last_range)
    except Exception:
        close_pos = 0.5

    # Directional close quality: long wants close near highs; short near lows.
    close_q = float(close_pos) if direction == "LONG" else float(1.0 - close_pos)
    close_q = float(np.clip(close_q, 0.0, 1.0))

    # Multi-bar close stacking quality: reward orderly continuation, not just the latest bar.
    stack_tail = int(min(4, len(close_s)))
    stack_closes = close_s.tail(stack_tail)
    stack_highs = high_s.tail(stack_tail)
    stack_lows = low_s.tail(stack_tail)
    stack_ranges = (stack_highs - stack_lows).replace(0.0, np.nan)
    stack_close_pos = (((stack_closes - stack_lows) / stack_ranges).replace([np.inf, -np.inf], np.nan).fillna(0.5)).astype(float)
    if direction == "LONG":
        continuity_raw = float((stack_closes.diff().dropna() >= -0.02 * float(atr_last)).mean()) if len(stack_closes) >= 2 else 0.5
        stack_pos_q = float(np.clip(stack_close_pos.mean(), 0.0, 1.0))
    else:
        continuity_raw = float((stack_closes.diff().dropna() <= 0.02 * float(atr_last)).mean()) if len(stack_closes) >= 2 else 0.5
        stack_pos_q = float(np.clip((1.0 - stack_close_pos).mean(), 0.0, 1.0))
    close_stack_q = float(np.clip(0.55 * continuity_raw + 0.45 * stack_pos_q, 0.0, 1.0))

    # Phase-aware displacement elasticity:
    # - early / buildup phases can be legitimate before the latest bar becomes
    #   a textbook 1.2x ATR impulse
    # - mature / extended phases should stay stricter
    phase_disp_floor = 1.20
    try:
        if phase_preview_detail in ('PRE_COMPRESSION', 'ACCEPTANCE_ATTEMPT', 'COMPRESSION_BUILDUP', 'RE_COMPRESSION', 'BREAK_AND_HOLD', 'EARLY_ACCEPTANCE'):
            phase_disp_floor = 1.05
        elif phase_preview_detail == 'PERSISTENT_CONTINUATION':
            phase_disp_floor = 1.10
        elif phase_preview_detail in ('MATURE_ACCEPTANCE', 'EXTEND_THEN_PULLBACK', 'STALLING_EXTENSION', 'FAILED_EXTENSION'):
            phase_disp_floor = 1.25
    except Exception:
        phase_disp_floor = 1.20
    disp_q = float(np.clip((disp_ratio - max(0.88, phase_disp_floor - 0.15)) / 1.45, 0.0, 1.0))
    disp_q = float(np.clip(disp_q + 0.04 * np.clip(phase_preview_conf, 0.0, 1.0) * float(phase_preview_detail in ('PRE_COMPRESSION', 'ACCEPTANCE_ATTEMPT', 'COMPRESSION_BUILDUP', 'RE_COMPRESSION', 'BREAK_AND_HOLD', 'EARLY_ACCEPTANCE')), 0.0, 1.0))

    # Grade volume quality rather than using a near-binary ladder.
    relvol_ratio = float(vol_impulse / max(1e-9, 1.5 * med30)) if med30 > 0 else 0.0
    relvol_q = float(np.clip(relvol_ratio, 0.0, 1.0))
    dollar_flow_ratio = float(dollar_flow / max(1e-9, 0.85 * dollar_flow_ref)) if dollar_flow_ref > 0 else 0.0
    dollar_q = float(np.clip(dollar_flow_ratio, 0.0, 1.0))
    if vol_impulse > 0:
        hold_ratio = float(vol_hold / max(1e-9, vol_impulse))
        compression_q = 1.0 if hold_compression_ok else float(np.clip((1.30 - hold_ratio) / 0.30, 0.0, 1.0))
    else:
        compression_q = 0.0
    if direction == "LONG":
        participation_q = 1.0 if bull_participation else (0.55 if (close_q_long >= 0.58 and body_ratio >= 0.32 and upper_wick_frac <= 0.38) else 0.15)
    else:
        participation_q = 1.0 if bear_participation else (0.55 if (close_q_short >= 0.58 and body_ratio >= 0.32 and lower_wick_frac <= 0.38) else 0.15)
    wick_control_q = float(np.clip(1.0 - hold_reject_wick / 0.35, 0.0, 1.0))
    vol_q = float(np.clip(0.28 * relvol_q + 0.22 * dollar_q + 0.20 * compression_q + 0.15 * float(hold_structure_quality) + 0.15 * participation_q, 0.0, 1.0))

    # Persistence quality rewards orderly continuation / shallow holds common in strong volatile names.
    accept_persist_q = 1.0 if accept_ok else (0.65 if retest_seen else 0.35)
    persistence_q = float(np.clip(0.35 * close_stack_q + 0.30 * float(hold_structure_quality) + 0.20 * wick_control_q + 0.15 * accept_persist_q, 0.0, 1.0))

    retest_q = 1.0 if (stage == "CONFIRMED") else (0.5 if accept_ok else 0.0)
    legitimacy_q = float(np.clip(impulse_legitimacy, 0.0, 1.0))

    # Phase-sensitive weighting keeps early build / clean hold states from being
    # over-penalized for lacking a textbook retest, while mature / late phases
    # demand more persistence and less displacement hype.
    disp_w = 0.18
    close_w = 0.15
    close_stack_w = 0.12
    vol_w = 0.15
    retest_w = 0.12
    legitimacy_w = 0.18
    persistence_w = 0.10
    try:
        if phase_preview_detail in ('PRE_COMPRESSION', 'ACCEPTANCE_ATTEMPT', 'COMPRESSION_BUILDUP', 'RE_COMPRESSION', 'BREAK_AND_HOLD', 'EARLY_ACCEPTANCE'):
            disp_w += 0.03
            close_stack_w += 0.02
            persistence_w += 0.03
            retest_w -= 0.05
        elif phase_preview_detail == 'PERSISTENT_CONTINUATION':
            persistence_w += 0.03
            close_stack_w += 0.02
            retest_w -= 0.03
        elif phase_preview_detail in ('MATURE_ACCEPTANCE', 'EXTEND_THEN_PULLBACK', 'STALLING_EXTENSION'):
            persistence_w += 0.04
            retest_w += 0.02
            disp_w -= 0.03
        elif phase_preview_detail == 'FAILED_EXTENSION':
            persistence_w += 0.05
            retest_w += 0.03
            disp_w -= 0.04
    except Exception:
        pass
    weight_sum = float(max(1e-9, disp_w + close_w + close_stack_w + vol_w + retest_w + legitimacy_w + persistence_w))
    impulse_quality = float(np.clip((
        disp_w * disp_q
        + close_w * close_q
        + close_stack_w * close_stack_q
        + vol_w * vol_q
        + retest_w * retest_q
        + legitimacy_w * legitimacy_q
        + persistence_w * persistence_q
    ) / weight_sum, 0.0, 1.0))
    try:
        if phase_preview_detail in ('PRE_COMPRESSION', 'ACCEPTANCE_ATTEMPT', 'COMPRESSION_BUILDUP', 'RE_COMPRESSION', 'BREAK_AND_HOLD', 'EARLY_ACCEPTANCE') and float(phase_preview_conf) >= 0.60:
            impulse_quality = float(np.clip(impulse_quality + 0.03 * np.clip(close_stack_q * persistence_q, 0.0, 1.0), 0.0, 1.0))
        elif phase_preview_detail in ('MATURE_ACCEPTANCE', 'EXTEND_THEN_PULLBACK', 'STALLING_EXTENSION'):
            impulse_quality = float(np.clip(impulse_quality - 0.02 * np.clip(1.0 - persistence_q, 0.0, 1.0), 0.0, 1.0))
    except Exception:
        pass

    # Fold in RSI rideability (timing realism). This does NOT create the signal;
    # it just prevents weak/overextended moves from scoring like perfect rides.
    # Keep it gentle: at most ~15% adjustment.
    impulse_quality = float(np.clip(impulse_quality * (0.85 + 0.15 * float(rsi_q)), 0.0, 1.0))

    # If we're exhausted, don't allow CONFIRMED without a retest/hold.
    if exhausted and stage == "CONFIRMED":
        stage = "PRE"

    # If the impulse/accept sequence is low quality, don't label it "rideable".
    # This keeps 100 scores from appearing on flimsy moves.
    if impulse_quality < 0.35:
        # Too weak to trade as continuation.
        return SignalResult(
            symbol=symbol,
            bias="CHOP",
            setup_score=0.0,
            reason="Not rideable (low impulse quality)",
            entry=None,
            stop=None,
            target_1r=None,
            target_2r=None,
            last_price=last_price,
            timestamp=last_ts,
            session=session,
            extras={"mode": "RIDE", "stage": None, "trend_votes": trend_votes, "impulse_quality": impulse_quality,
                    "disp_ratio": disp_ratio, "liquidity_phase": liquidity_phase, "vol_ok_long": bool(vol_ok_long), "vol_ok_short": bool(vol_ok_short), "dollar_flow_ok": bool(dollar_flow_ok)},
        )

    # If quality is mediocre, allow PRE but not CONFIRMED.
    if stage == "CONFIRMED" and impulse_quality < 0.55:
        stage = "PRE"

    # scoring (quality-weighted)
    pts = 0.0
    pts += 22.0  # base for being in a trend-filtered universe
    pts += 18.0 if pass_adx else 0.0
    pts += 12.0 if pass_di_gap else 0.0
    pts += 15.0 if (direction == "LONG" and pass_ema_up) or (direction == "SHORT" and pass_ema_dn) else 0.0

    # Impulse + acceptance are amplified by quality; weak impulses shouldn't look like 100s.
    pts += (26.0 * impulse_quality)
    pts += (14.0 * impulse_quality) if stage == "CONFIRMED" else (7.0 * impulse_quality)
    pts += (10.0 * liquidity_mult) if vol_ok else (4.0 * liquidity_mult if vol_q >= 0.50 else 0.0)
    pts -= 12.0 if exhausted else 0.0

    htf_effect = 0.0
    htf_label = None
    if isinstance(htf_bias, dict) and "bias" in htf_bias:
        hb = str(htf_bias.get("bias", "")).upper()
        htf_label = hb or None
        if direction == "LONG":
            if hb in ("BULL", "BULLISH"):
                htf_effect = 6.0
            elif hb in ("BEAR", "BEARISH"):
                htf_effect = -5.0
        elif direction == "SHORT":
            if hb in ("BEAR", "BEARISH"):
                htf_effect = 6.0
            elif hb in ("BULL", "BULLISH"):
                htf_effect = -5.0
        pts += htf_effect

    score = _cap_score(pts)

    # Session discipline for volatile continuation setups:
    # treat off-window and lower-carry windows as requiring more proof.
    session_confirm_bump_map = {"OFF": 2, "MIDDAY": 1, "PREMARKET": 2, "AFTERHOURS": 2, "OPENING": 0, "POWER": 0}
    session_penalty_map = {"OFF": 8, "MIDDAY": 3, "PREMARKET": 5, "AFTERHOURS": 5, "OPENING": 0, "POWER": 0}
    raw_session_risk = int(session_penalty_map.get(str(session), session_penalty_map.get(raw_session, 2))) if str(session) == "OFF" else int(session_penalty_map.get(raw_session, 0))
    session_confirm_bump = int(session_confirm_bump_map.get(str(session), session_confirm_bump_map.get(raw_session, 0))) if str(session) == "OFF" else int(session_confirm_bump_map.get(raw_session, 0))

    # --- Entries: pullback band (PB1/PB2) + break trigger ---
    # A single-line pullback is too brittle. Bands are more realistic for continuation execution.
    #
    # Phase 2 upgrade: keep payload names the same, but make the pullback band width adaptive
    # to ATR and impulse quality. Stronger impulses deserve shallower pullback bands; weaker
    # setups require deeper pullbacks before we call them attractive.
    #
    # Additional refinement: use the *final* RIDE score as the conviction signal for banding.
    # High-conviction (score >= 87) setups get tighter/shallower pullback bands; everything
    # else gets slightly deeper bands. We compute a provisional geometry to score the entry zone,
    # then rebuild the final band from the final score so the displayed setup and final score agree.
    #
    # Also: if the setup is actionable because we're *near the break trigger* (not near pullback),
    # we should not keep showing a stale pullback limit far away. In that case we surface a
    # breakout-style entry (stop/trigger + a small chase line).
    q_weak = float(np.clip(1.0 - float(impulse_quality), 0.0, 1.0))

    # Wire the existing slippage controls into RIDE in a controlled way.
    # This should only add small tactical elasticity to the executable entry / proximity checks;
    # it should not rewrite the overall trade thesis.
    slip_amt = _slip_amount(
        slippage_mode=slippage_mode,
        fixed_slippage_cents=fixed_slippage_cents,
        atr_last=float(atr_last or 0.0),
        atr_fraction_slippage=float(atr_fraction_slippage or 0.0),
    )
    entry_pad = float(min(max(0.0, float(slip_amt)), 0.30 * float(atr_last)))

    # IMPORTANT: break_trigger must be stable across refreshes.
    # Anchor it to the strongest legitimate impulse source, not just the first raw break we see.
    if direction == "LONG":
        if impulse_type_hint == "ORB" and isinstance(orb_high, (float, int)):
            impulse_type, impulse_level = "ORB", float(orb_high)
        elif impulse_type_hint == "PIVOT" and isinstance(swing_hi, (float, int)):
            impulse_type, impulse_level = "PIVOT", float(swing_hi)
        else:
            impulse_type, impulse_level = "VWAP", float(ref_vwap) if isinstance(ref_vwap, (float, int)) else float(accept_line)
    else:
        if impulse_type_hint == "ORB" and isinstance(orb_low, (float, int)):
            impulse_type, impulse_level = "ORB", float(orb_low)
        elif impulse_type_hint == "PIVOT" and isinstance(swing_lo, (float, int)):
            impulse_type, impulse_level = "PIVOT", float(swing_lo)
        else:
            impulse_type, impulse_level = "VWAP", float(ref_vwap) if isinstance(ref_vwap, (float, int)) else float(accept_line)

    impulse_idx: Optional[int] = None
    try:
        lvl = float(impulse_level)
        if direction == "LONG":
            crossed = (df["close"].astype(float) > lvl) & (df["close"].astype(float).shift(1) <= lvl)
        else:
            crossed = (df["close"].astype(float) < lvl) & (df["close"].astype(float).shift(1) >= lvl)
        cross_locs = np.flatnonzero(crossed.fillna(False).to_numpy())
        if len(cross_locs):
            impulse_idx = int(cross_locs[-1])
    except Exception:
        impulse_idx = None

    def _build_ride_entry_geometry(conviction_score: float) -> dict[str, object]:
        high_conviction_local = bool(float(conviction_score or 0.0) >= 87.0)

        # Make the pullback band adapt not only to weak/strong impulse quality,
        # but also to how well the move is actually being accepted/held.
        src_weight_raw = float(accept_component_weights.get(accept_src, 0.18)) if isinstance(accept_component_weights, dict) else 0.18
        src_conf_local = float(np.clip((src_weight_raw - 0.05) / 1.05, 0.0, 1.0))
        accept_quality_local = float(np.clip(
            0.42 * float(accept_ok) + 0.23 * float(retest_seen) + 0.35 * float(hold_ok),
            0.0,
            1.0,
        ))
        acceptance_loose_local = float(np.clip(1.0 - accept_quality_local, 0.0, 1.0))
        source_loose_local = float(np.clip(1.0 - src_conf_local, 0.0, 1.0))
        extension_from_accept_local = (
            max(0.0, float(last_price - accept_line)) if direction == "LONG" else max(0.0, float(accept_line - last_price))
        )
        extension_loose_local = float(np.clip(extension_from_accept_local / max(1e-9, 0.85 * float(atr_last)), 0.0, 1.0))

        width_bias_inner = (0.00 if high_conviction_local else 0.03)
        width_bias_outer = (0.00 if high_conviction_local else 0.05)
        width_bias_inner += 0.025 * acceptance_loose_local + 0.015 * source_loose_local + 0.010 * extension_loose_local
        width_bias_outer += 0.055 * acceptance_loose_local + 0.030 * source_loose_local + 0.025 * extension_loose_local
        width_bias_inner -= 0.015 * accept_quality_local + 0.010 * src_conf_local
        width_bias_outer -= 0.030 * accept_quality_local + 0.015 * src_conf_local

        # Dynamic pullback depth: strong, legitimate, well-supported continuation
        # should not wait for a fantasy deep pullback, while weaker/looser trends
        # should demand more depth before being called attractive.
        trend_context_local = float(long_trend_context if direction == "LONG" else short_trend_context)
        trend_strength_local = float(np.clip(trend_context_local / 0.34, 0.0, 1.0))
        legit_local = float(np.clip(impulse_legitimacy, 0.0, 1.0))
        vol_side_q_local = float(np.clip(vol_q, 0.0, 1.0))
        session_quality_local = float(np.clip(1.0 - (float(raw_session_risk) / 8.0), 0.0, 1.0))
        dynamic_shallow_bias_local = float(np.clip(
            0.34 * trend_strength_local
            + 0.24 * legit_local
            + 0.18 * accept_quality_local
            + 0.14 * vol_side_q_local
            + 0.10 * session_quality_local,
            0.0,
            1.0,
        ))
        dynamic_deep_bias_local = float(np.clip(
            0.30 * float(q_weak)
            + 0.24 * acceptance_loose_local
            + 0.18 * source_loose_local
            + 0.16 * extension_loose_local
            + 0.12 * float(1.0 - session_quality_local),
            0.0,
            1.0,
        ))

        # Enhancement B: adaptive pullback depth should recognize when a strong,
        # persistent tape is still best traded as a pullback, but only a shallow one.
        # This is especially important for volatile names that staircase higher with
        # minimal giveback once directional control is established.
        recent_hold_window_local = df.tail(int(min(4, len(df)))).copy()
        try:
            rh_open_local = pd.to_numeric(recent_hold_window_local['open'], errors='coerce').dropna()
            rh_close_local = pd.to_numeric(recent_hold_window_local['close'], errors='coerce').dropna()
            rh_high_local = pd.to_numeric(recent_hold_window_local['high'], errors='coerce').dropna()
            rh_low_local = pd.to_numeric(recent_hold_window_local['low'], errors='coerce').dropna()
            if min(len(rh_open_local), len(rh_close_local), len(rh_high_local), len(rh_low_local)) >= 3 and float(atr_last) > 0:
                body_frac_seq_local = float(np.mean(np.abs(rh_close_local.values - rh_open_local.values) / np.maximum(1e-9, rh_high_local.values - rh_low_local.values)))
                if direction == 'LONG':
                    close_stack_quality_local = float(np.clip((float(rh_close_local.iloc[-1]) - float(rh_close_local.min())) / max(1e-9, 0.55 * float(atr_last)), 0.0, 1.0))
                    rejection_wick_local = float(np.mean((rh_high_local.values - np.maximum(rh_close_local.values, rh_open_local.values)) / np.maximum(1e-9, rh_high_local.values - rh_low_local.values)))
                    shallow_retrace_local = float(np.clip((float(rh_high_local.max()) - float(rh_low_local.min())) / max(1e-9, 0.90 * float(atr_last)), 0.0, 1.0))
                else:
                    close_stack_quality_local = float(np.clip((float(rh_close_local.max()) - float(rh_close_local.iloc[-1])) / max(1e-9, 0.55 * float(atr_last)), 0.0, 1.0))
                    rejection_wick_local = float(np.mean((np.minimum(rh_close_local.values, rh_open_local.values) - rh_low_local.values) / np.maximum(1e-9, rh_high_local.values - rh_low_local.values)))
                    shallow_retrace_local = float(np.clip((float(rh_high_local.max()) - float(rh_low_local.min())) / max(1e-9, 0.90 * float(atr_last)), 0.0, 1.0))
            else:
                body_frac_seq_local = 0.0
                close_stack_quality_local = 0.0
                rejection_wick_local = 1.0
                shallow_retrace_local = 1.0
        except Exception:
            body_frac_seq_local = 0.0
            close_stack_quality_local = 0.0
            rejection_wick_local = 1.0
            shallow_retrace_local = 1.0

        persistent_hold_quality_local = float(np.clip(
            0.34 * trend_strength_local
            + 0.24 * float(np.clip((float(adx_last) - 32.0) / 18.0, 0.0, 1.0))
            + 0.16 * legit_local
            + 0.12 * vol_side_q_local
            + 0.08 * body_frac_seq_local
            + 0.06 * close_stack_quality_local,
            0.0,
            1.0,
        ))

        # Compute the raw break trigger first. Pullback-band geometry is built later
        # after the dynamic width biases and phase-aware adjustments are finalized.
        if direction == "LONG":
            break_trigger_local = float(max(float(impulse_level), float(df["high"].iloc[impulse_idx]))) if impulse_idx is not None else float(impulse_level)
        else:
            break_trigger_local = float(min(float(impulse_level), float(df["low"].iloc[impulse_idx]))) if impulse_idx is not None else float(impulse_level)

        # Compression-breakout detection is shared with SCALP, but RIDE must initialize
        # the local values before any downstream breakout-bias logic touches them.
        compression_breakout_ctx_local = {"score": 0.0, "ready": False, "compression": 0.0, "breakout": 0.0, "volume": 0.0, "momentum": 0.0, "close_quality": 0.0}
        compression_breakout_ready_local = False
        compression_breakout_score_local = 0.0
        try:
            compression_breakout_ctx_local = _assess_compression_breakout(
                str(direction),
                df,
                float(atr_last) if atr_last is not None else None,
                lookback=6,
                break_trigger=float(break_trigger_local),
            ) or compression_breakout_ctx_local
            compression_breakout_ready_local = bool(compression_breakout_ctx_local.get("ready") or False)
            compression_breakout_score_local = float(compression_breakout_ctx_local.get("score") or 0.0)
        except Exception:
            compression_breakout_ctx_local = {"score": 0.0, "ready": False, "compression": 0.0, "breakout": 0.0, "volume": 0.0, "momentum": 0.0, "close_quality": 0.0}
            compression_breakout_ready_local = False
            compression_breakout_score_local = 0.0

        structure_phase_info_local = _classify_ride_structure_phase_info(
            direction=str(direction),
            df=df,
            accept_line=float(accept_line),
            break_trigger=float(break_trigger_local),
            atr_last=float(atr_last) if atr_last is not None else None,
        )
        structure_phase_local = str(structure_phase_info_local.get('route_phase') or 'UNSET')
        structure_phase_detail_local = str(structure_phase_info_local.get('detail_phase') or structure_phase_local)
        structure_phase_confidence_local = float(structure_phase_info_local.get('confidence') or 0.0)
        structure_phase_interpretation_local = str(structure_phase_info_local.get('interpretation') or 'No clean continuation phase established')
        late_trend_shallow_bias_local = float(np.clip(
            0.42 * float(structure_phase_detail_local in ('EXTEND_THEN_PULLBACK', 'STALLING_EXTENSION', 'MATURE_ACCEPTANCE'))
            + 0.28 * persistent_hold_quality_local
            + 0.18 * float(np.clip(1.0 - rejection_wick_local, 0.0, 1.0))
            + 0.12 * float(np.clip(1.0 - shallow_retrace_local, 0.0, 1.0)),
            0.0,
            1.0,
        ))
        adaptive_pullback_refine_local = float(np.clip(
            0.34 * float(structure_phase_detail_local in ('MATURE_ACCEPTANCE', 'EXTEND_THEN_PULLBACK', 'STALLING_EXTENSION'))
            + 0.18 * float(np.clip(structure_phase_confidence_local, 0.0, 1.0))
            + 0.20 * float(np.clip(trend_strength_local, 0.0, 1.0))
            + 0.14 * float(np.clip(persistence_q, 0.0, 1.0))
            + 0.14 * float(np.clip(close_stack_q, 0.0, 1.0)),
            0.0,
            1.0,
        ))
        phase_pullback_tighten_local = 0.0
        phase_tighten_conf_local = float(np.clip(0.65 + 0.35 * float(np.clip(structure_phase_confidence_local, 0.0, 1.0)), 0.65, 1.0))
        if structure_phase_detail_local in ('EARLY_ACCEPTANCE', 'ACCEPTANCE_ATTEMPT'):
            phase_pullback_tighten_local = 1.00 * phase_tighten_conf_local
        elif structure_phase_detail_local == 'BREAK_AND_HOLD':
            phase_pullback_tighten_local = 0.85 * phase_tighten_conf_local
        elif structure_phase_detail_local == 'PERSISTENT_CONTINUATION':
            phase_pullback_tighten_local = 0.95 * phase_tighten_conf_local
        elif structure_phase_detail_local in ('COMPRESSION_BUILDUP', 'PRE_COMPRESSION', 'RE_COMPRESSION', 'RECOVERY_BUILD'):
            phase_pullback_tighten_local = 0.42 * phase_tighten_conf_local
        elif structure_phase_detail_local == 'MATURE_ACCEPTANCE':
            phase_pullback_tighten_local = 0.72 * phase_tighten_conf_local
        elif structure_phase_detail_local in ('EXTEND_THEN_PULLBACK', 'STALLING_EXTENSION'):
            phase_pullback_tighten_local = 0.08 * phase_tighten_conf_local
        elif structure_phase_detail_local in ('UNSTRUCTURED', 'EARLY_BUILD'):
            phase_pullback_tighten_local = 0.0
        else:
            phase_pullback_tighten_local = 0.0

        width_bias_inner -= 0.030 * dynamic_shallow_bias_local
        width_bias_outer -= 0.080 * dynamic_shallow_bias_local
        width_bias_inner -= 0.025 * late_trend_shallow_bias_local
        width_bias_outer -= 0.095 * late_trend_shallow_bias_local
        width_bias_inner -= 0.020 * adaptive_pullback_refine_local
        width_bias_outer -= 0.070 * adaptive_pullback_refine_local
        width_bias_inner -= 0.018 * phase_pullback_tighten_local
        width_bias_outer -= 0.085 * phase_pullback_tighten_local
        width_bias_inner += 0.020 * dynamic_deep_bias_local
        width_bias_outer += 0.085 * dynamic_deep_bias_local
        width_bias_inner = float(np.clip(width_bias_inner, -0.05, 0.10))
        width_bias_outer = float(np.clip(width_bias_outer, -0.10, 0.18))

        pb_inner_mult_local = float(np.clip(0.18 + 0.14 * q_weak + width_bias_inner, 0.12, 0.35))
        pb_outer_mult_local = float(np.clip(0.42 + 0.30 * q_weak + width_bias_outer, 0.28, 0.86))
        pb_inner_local = float(pb_inner_mult_local * float(atr_last))
        pb_outer_local = float(pb_outer_mult_local * float(atr_last))
        phase_pullback_anchor_local = float(accept_line)
        last_px_local = float(last_price)
        try:
            if direction == "LONG":
                if structure_phase_detail_local in ("EARLY_ACCEPTANCE", "ACCEPTANCE_ATTEMPT"):
                    phase_pullback_anchor_local = float(max(float(accept_line), last_px_local - 0.10 * float(atr_last)))
                elif structure_phase_detail_local == "PERSISTENT_CONTINUATION":
                    phase_pullback_anchor_local = float(max(float(accept_line), last_px_local - 0.08 * float(atr_last)))
                elif structure_phase_detail_local in ("COMPRESSION_BUILDUP", "PRE_COMPRESSION", "RE_COMPRESSION", "RECOVERY_BUILD"):
                    phase_pullback_anchor_local = float(max(float(accept_line), last_px_local - 0.04 * float(atr_last)))
                elif structure_phase_detail_local == "MATURE_ACCEPTANCE":
                    mature_anchor_blend_local = float(0.60 * last_px_local + 0.40 * float(accept_line))
                    phase_pullback_anchor_local = float(max(float(accept_line) + 0.05 * float(atr_last), mature_anchor_blend_local, last_px_local - 0.12 * float(atr_last)))
            else:
                if structure_phase_detail_local in ("EARLY_ACCEPTANCE", "ACCEPTANCE_ATTEMPT"):
                    phase_pullback_anchor_local = float(min(float(accept_line), last_px_local + 0.10 * float(atr_last)))
                elif structure_phase_detail_local == "PERSISTENT_CONTINUATION":
                    phase_pullback_anchor_local = float(min(float(accept_line), last_px_local + 0.08 * float(atr_last)))
                elif structure_phase_detail_local in ("COMPRESSION_BUILDUP", "PRE_COMPRESSION", "RE_COMPRESSION", "RECOVERY_BUILD"):
                    phase_pullback_anchor_local = float(min(float(accept_line), last_px_local + 0.04 * float(atr_last)))
                elif structure_phase_detail_local == "MATURE_ACCEPTANCE":
                    mature_anchor_blend_local = float(0.60 * last_px_local + 0.40 * float(accept_line))
                    phase_pullback_anchor_local = float(min(float(accept_line) - 0.05 * float(atr_last), mature_anchor_blend_local, last_px_local + 0.12 * float(atr_last)))
        except Exception:
            phase_pullback_anchor_local = float(accept_line)
        if direction == "LONG":
            pb1_local = float(phase_pullback_anchor_local) + pb_inner_local
            pb2_local = float(phase_pullback_anchor_local) - pb_outer_local
            pullback_entry_local = float(np.clip(float(phase_pullback_anchor_local), pb2_local, pb1_local))
            stop_mult_local = 0.55 if stage == "PRE" else 0.80
            stop_local = float(pullback_entry_local - stop_mult_local * atr_last)
        else:
            pb1_local = float(phase_pullback_anchor_local) - pb_inner_local
            pb2_local = float(phase_pullback_anchor_local) + pb_outer_local
            pullback_entry_local = float(np.clip(float(phase_pullback_anchor_local), pb1_local, pb2_local))
            stop_mult_local = 0.55 if stage == "PRE" else 0.80
            stop_local = float(pullback_entry_local + stop_mult_local * atr_last)
        prefers_breakout_phase_local = bool(structure_phase_detail_local in ("BREAK_AND_HOLD", "EARLY_ACCEPTANCE", "ACCEPTANCE_ATTEMPT", "PERSISTENT_CONTINUATION", "COMPRESSION_BUILDUP", "PRE_COMPRESSION", "RE_COMPRESSION", "RECOVERY_BUILD"))
        prefers_pullback_phase_local = bool(structure_phase_detail_local in ("EXTEND_THEN_PULLBACK", "STALLING_EXTENSION", "FAILED_EXTENSION", "MATURE_ACCEPTANCE"))
        multibar_extension_profile_local = _compute_multibar_extension_profile(
            df,
            direction=str(direction),
            atr_last=float(atr_last) if atr_last is not None else None,
            accept_line=float(accept_line),
        )

        prox_atr_local = 0.45
        try:
            hist_tail_prox = pd.to_numeric(df.get("macd_hist", pd.Series(index=df.index, dtype=float)).tail(int(min(4, len(df)))), errors="coerce").dropna()
        except Exception:
            hist_tail_prox = pd.Series(dtype=float)
        strong_pressure_local = False
        try:
            if len(hist_tail_prox) >= 3:
                if direction == "LONG":
                    strong_pressure_local = bool(hist_tail_prox.iloc[-1] > hist_tail_prox.iloc[-2] > hist_tail_prox.iloc[-3])
                else:
                    strong_pressure_local = bool(hist_tail_prox.iloc[-1] < hist_tail_prox.iloc[-2] < hist_tail_prox.iloc[-3])
        except Exception:
            strong_pressure_local = False
        macd_breakout_bonus_local = 0.0
        macd_state_local = str((ride_macd_info or {}).get('aligned_state') or 'NEUTRAL_NOISE').upper().strip()
        if prefers_breakout_phase_local:
            if macd_state_local in ('IGNITING', 'ACCELERATING'):
                macd_breakout_bonus_local = 0.08 if macd_state_local == 'IGNITING' else 0.12
                prox_atr_local = max(prox_atr_local, 0.50 + macd_breakout_bonus_local)
            elif macd_state_local == 'TAPERING':
                prox_atr_local = min(prox_atr_local, 0.42)
            elif macd_state_local in ('ROLLING_OVER', 'COUNTER_TREND'):
                prox_atr_local = min(prox_atr_local, 0.36)
        if bool(tape_mode_enabled) and strong_pressure_local and float(impulse_quality or 0.0) >= 0.48 and float(adx_modifier or 0.0) >= 1.0:
            prox_atr_local = max(prox_atr_local, 0.50)
        if prefers_breakout_phase_local and float(impulse_quality or 0.0) >= 0.45 and not bool(multibar_extension_profile_local.get("path_stretched") or False):
            prox_atr_local = max(prox_atr_local, 0.50)
        elif prefers_pullback_phase_local:
            prox_atr_local = max(prox_atr_local, 0.48)

        # Entry timing acceleration: on the cleanest continuation states, allow
        # RIDE to lean in slightly earlier instead of waiting for a picture-perfect
        # retrace that fast volatile names often never give.
        acceleration_ready_local = bool(
            prefers_breakout_phase_local
            and trend_strength_local >= 0.58
            and legit_local >= 0.60
            and float(impulse_quality or 0.0) >= 0.58
            and vol_side_q_local >= 0.50
            and accept_quality_local >= 0.56
            and float(raw_session_risk) <= 3.0
            and (not bool(multibar_extension_profile_local.get("path_stretched") or False))
            and (not bool(multibar_extension_profile_local.get("stalling") or False))
            and (not bool(multibar_extension_profile_local.get("fading") or multibar_extension_profile_local.get("momentum_fade") or False))
            and extension_loose_local <= 0.72
        )
        acceleration_boost_local = 0.0
        if acceleration_ready_local:
            acceleration_boost_local = float(np.clip(
                0.45 * trend_strength_local + 0.25 * legit_local + 0.20 * vol_side_q_local + 0.10 * accept_quality_local,
                0.0,
                1.0,
            ))
            prox_atr_local = max(prox_atr_local, 0.53 + 0.05 * acceleration_boost_local)

        prebreak_compression_ready_local = False
        prebreak_compression_score_local = 0.0
        prebreak_bias_bonus_local = 0.0
        persistent_trend_ready_local = False
        persistent_trend_score_local = 0.0
        persistent_bias_bonus_local = 0.0
        ignition_continuation_ready_local = False
        ignition_continuation_score_local = 0.0
        ignition_bias_bonus_local = 0.0

        prox_dist_local = prox_atr_local * float(atr_last) + entry_pad
        breakout_stale_mult_local = 0.50 if bool(multibar_extension_profile_local.get("path_stretched") or False) else 0.60
        if acceleration_ready_local:
            breakout_stale_mult_local = float(min(0.76, breakout_stale_mult_local + 0.08 + 0.08 * acceleration_boost_local))
        if direction == "LONG":
            dist_pb_band_local = 0.0 if (last_price >= pb2_local and last_price <= pb1_local) else min(abs(last_price - pb2_local), abs(last_price - pb1_local))
            stale_breakout_local = bool(last_price > break_trigger_local + breakout_stale_mult_local * atr_last + entry_pad and dist_pb_band_local > prox_dist_local)
        else:
            dist_pb_band_local = 0.0 if (last_price <= pb2_local and last_price >= pb1_local) else min(abs(last_price - pb2_local), abs(last_price - pb1_local))
            stale_breakout_local = bool(last_price < break_trigger_local - breakout_stale_mult_local * atr_last - entry_pad and dist_pb_band_local > prox_dist_local)

        dist_br_local = abs(last_price - break_trigger_local)
        breakout_proximity_ratio_local = float(dist_br_local / max(1e-9, prox_dist_local))
        breakout_proximity_score_local = 0.0
        breakout_proximity_penalty_local = 0.0
        breakout_proximity_bucket_local = "far"
        if breakout_proximity_ratio_local <= 0.55:
            breakout_proximity_score_local = 1.00
            breakout_proximity_bucket_local = "very_close"
        elif breakout_proximity_ratio_local <= 0.90:
            breakout_proximity_score_local = 0.82
            breakout_proximity_bucket_local = "close"
        elif breakout_proximity_ratio_local <= 1.12:
            breakout_proximity_score_local = 0.58
            breakout_proximity_bucket_local = "workable"
        elif breakout_proximity_ratio_local <= 1.32:
            breakout_proximity_score_local = 0.30
            breakout_proximity_bucket_local = "stretched"
        elif breakout_proximity_ratio_local <= 1.55:
            breakout_proximity_score_local = 0.08
            breakout_proximity_penalty_local = 0.45
            breakout_proximity_bucket_local = "late"
        else:
            breakout_proximity_score_local = 0.0
            breakout_proximity_penalty_local = 0.95
            breakout_proximity_bucket_local = "chase"

        near_pullback_local = bool(dist_pb_band_local <= prox_dist_local)
        near_break_local = bool(breakout_proximity_ratio_local <= 1.0)
        if not near_break_local and prefers_breakout_phase_local and float(impulse_quality or 0.0) >= 0.45 and breakout_proximity_ratio_local <= 1.10 and not stale_breakout_local and not bool(multibar_extension_profile_local.get("path_stretched") or False):
            near_break_local = True
        if not near_break_local and acceleration_ready_local and breakout_proximity_ratio_local <= (1.18 + 0.12 * acceleration_boost_local) and not stale_breakout_local:
            near_break_local = True
        if not near_pullback_local and prefers_pullback_phase_local and dist_pb_band_local <= 1.10 * prox_dist_local:
            near_pullback_local = True

        # Trader-edge defense lane: enter on accept-line defense before full acceptance.
        defense_entry_ready_local = False
        defense_entry_score_local = 0.0
        try:
            _def_recent = df.tail(int(min(4, len(df)))).copy()
            _def_close = pd.to_numeric(_def_recent["close"], errors="coerce").dropna()
            _def_low = pd.to_numeric(_def_recent["low"], errors="coerce").dropna()
            _def_high = pd.to_numeric(_def_recent["high"], errors="coerce").dropna()
            _def_open = pd.to_numeric(_def_recent["open"], errors="coerce").dropna()
            if min(len(_def_close), len(_def_low), len(_def_high), len(_def_open)) >= 3 and float(atr_last) > 0:
                if direction == "LONG":
                    _def_level_hold = bool(float(_def_low.tail(3).min()) >= float(accept_line) - 0.22 * float(atr_last))
                    _def_structure = bool(float(_def_low.iloc[-1]) >= float(_def_low.iloc[-2]) - 0.08 * float(atr_last) and float(_def_close.iloc[-1]) >= float(_def_close.iloc[-2]) - 0.04 * float(atr_last))
                    _def_close_quality = bool(float(_def_close.iloc[-1]) >= float(_def_open.iloc[-1]) or float(_def_close.iloc[-1]) >= float(_def_close.iloc[-2]))
                else:
                    _def_level_hold = bool(float(_def_high.tail(3).max()) <= float(accept_line) + 0.22 * float(atr_last))
                    _def_structure = bool(float(_def_high.iloc[-1]) <= float(_def_high.iloc[-2]) + 0.08 * float(atr_last) and float(_def_close.iloc[-1]) <= float(_def_close.iloc[-2]) + 0.04 * float(atr_last))
                    _def_close_quality = bool(float(_def_close.iloc[-1]) <= float(_def_open.iloc[-1]) or float(_def_close.iloc[-1]) <= float(_def_close.iloc[-2]))
                _def_pullback_close = bool(dist_pb_band_local <= 1.38 * prox_dist_local)
                _def_momentum = bool((float(adx_slope if 'adx_slope' in locals() else 0.0) >= -0.05) or float(vol_side_q_local) >= 0.48 or float(close_stack_q) >= 0.52)
                defense_entry_score_local = float(np.clip(0.25 * float(_def_level_hold) + 0.25 * float(_def_structure) + 0.18 * float(_def_close_quality) + 0.17 * float(_def_pullback_close) + 0.15 * float(_def_momentum), 0.0, 1.0))
                defense_entry_ready_local = bool(_def_level_hold and _def_structure and _def_close_quality and _def_pullback_close and _def_momentum and legit_local >= 0.54 and trend_strength_local >= 0.50 and not bool(multibar_extension_profile_local.get('fading') or multibar_extension_profile_local.get('momentum_fade') or False))
                if defense_entry_ready_local:
                    near_pullback_local = True
                    stale_breakout_local = False
        except Exception:
            defense_entry_ready_local = False
            defense_entry_score_local = 0.0

        # Pre-break compression entry: multi-speed on 5m bars.
        # Fast (4 bars) catches the first trader-readable ignition, standard (6)
        # catches the clean coil, and slow (10) catches broader pressure builds.
        # The best window is selected, but all windows still obey phase, pressure,
        # extension, and proximity guards so we do not turn RIDE into a chase engine.
        try:
            _prebreak_windows_local = {}
            for _pb_label_local, _pb_n_local in (("fast", 4), ("standard", 6), ("slow", 10)):
                recent_pb_local = df.tail(int(min(_pb_n_local, len(df)))).copy()
                pb_close_local = pd.to_numeric(recent_pb_local['close'], errors='coerce').dropna()
                pb_open_local = pd.to_numeric(recent_pb_local['open'], errors='coerce').dropna()
                pb_high_local = pd.to_numeric(recent_pb_local['high'], errors='coerce').dropna()
                pb_low_local = pd.to_numeric(recent_pb_local['low'], errors='coerce').dropna()
                pb_vol_local = pd.to_numeric(recent_pb_local.get('volume', pd.Series(index=recent_pb_local.index, dtype=float)), errors='coerce').dropna()
                if min(len(pb_close_local), len(pb_open_local), len(pb_high_local), len(pb_low_local)) < 4 or float(atr_last) <= 0:
                    continue
                pb_range_atr_local = float((float(pb_high_local.max()) - float(pb_low_local.min())) / max(1e-9, float(atr_last)))
                pb_body_frac_local = float(np.mean(np.abs(pb_close_local.values - pb_open_local.values) / np.maximum(1e-9, pb_high_local.values - pb_low_local.values)))
                # Fast ignition can be a little less compressed; slow coils should be tighter and more orderly.
                if _pb_label_local == "fast":
                    pb_tight_local = bool(pb_range_atr_local <= 1.18 and pb_body_frac_local <= 0.72)
                elif _pb_label_local == "slow":
                    pb_tight_local = bool(pb_range_atr_local <= 1.45 and pb_body_frac_local <= 0.66)
                else:
                    pb_tight_local = bool(pb_range_atr_local <= 1.05 and pb_body_frac_local <= 0.68)
                if len(pb_vol_local) >= 4:
                    early_vol_local = float(np.mean(pb_vol_local.values[:max(2, len(pb_vol_local)//3)]))
                    late_vol_local = float(np.mean(pb_vol_local.values[-max(2, len(pb_vol_local)//3):]))
                    # Accept either dry-up coil or early expansion after dry-up.
                    pb_vol_tight_local = bool(
                        (late_vol_local <= 0.98 * max(1e-9, early_vol_local) and late_vol_local >= 0.55 * max(1e-9, float(med30 or 0.0)))
                        or (bool(strong_pressure_local) and late_vol_local >= 0.85 * max(1e-9, early_vol_local))
                    )
                else:
                    pb_vol_tight_local = bool(vol_side_q_local >= 0.45)
                if direction == 'LONG':
                    defended_pb_local = bool(float(pb_low_local.tail(int(min(3, len(pb_low_local)))).min()) >= float(accept_line) - 0.18 * float(atr_last))
                    trigger_ready_pb_local = bool(float(last_price) <= float(break_trigger_local) + 0.05 * float(atr_last) and float(last_price) >= float(break_trigger_local) - (0.50 if _pb_label_local != "fast" else 0.42) * float(atr_last))
                    close_stack_pb_local = bool(float(pb_close_local.iloc[-1]) >= float(pb_close_local.iloc[-2]) - 0.03 * float(atr_last) and float(pb_close_local.iloc[-1]) >= float(pb_close_local.iloc[-3]) - 0.02 * float(atr_last))
                    wick_ratio_pb_local = float(np.mean((pb_high_local.values - np.maximum(pb_close_local.values, pb_open_local.values)) / np.maximum(1e-9, pb_high_local.values - pb_low_local.values)))
                    wick_ok_pb_local = bool(wick_ratio_pb_local <= (0.34 if _pb_label_local == "fast" else 0.30))
                    anticipatory_side_ok_local = bool(float(last_price) < float(break_trigger_local) + 0.02 * float(atr_last))
                    anticipatory_dist_ok_local = bool(0.025 * float(atr_last) <= (float(break_trigger_local) - float(last_price)) <= (0.52 if _pb_label_local != "fast" else 0.44) * float(atr_last))
                    anticipatory_structure_ok_local = bool(
                        float(pb_low_local.iloc[-1]) >= float(pb_low_local.iloc[-2]) - 0.05 * float(atr_last)
                        and float(pb_low_local.iloc[-2]) >= float(pb_low_local.iloc[-3]) - 0.06 * float(atr_last)
                        and float(pb_close_local.iloc[-1]) >= float(pb_close_local.iloc[-2]) - 0.03 * float(atr_last)
                    )
                else:
                    defended_pb_local = bool(float(pb_high_local.tail(int(min(3, len(pb_high_local)))).max()) <= float(accept_line) + 0.18 * float(atr_last))
                    trigger_ready_pb_local = bool(float(last_price) >= float(break_trigger_local) - 0.05 * float(atr_last) and float(last_price) <= float(break_trigger_local) + (0.50 if _pb_label_local != "fast" else 0.42) * float(atr_last))
                    close_stack_pb_local = bool(float(pb_close_local.iloc[-1]) <= float(pb_close_local.iloc[-2]) + 0.03 * float(atr_last) and float(pb_close_local.iloc[-1]) <= float(pb_close_local.iloc[-3]) + 0.02 * float(atr_last))
                    wick_ratio_pb_local = float(np.mean((np.minimum(pb_close_local.values, pb_open_local.values) - pb_low_local.values) / np.maximum(1e-9, pb_high_local.values - pb_low_local.values)))
                    wick_ok_pb_local = bool(wick_ratio_pb_local <= (0.34 if _pb_label_local == "fast" else 0.30))
                    anticipatory_side_ok_local = bool(float(last_price) > float(break_trigger_local) - 0.02 * float(atr_last))
                    anticipatory_dist_ok_local = bool(0.025 * float(atr_last) <= (float(last_price) - float(break_trigger_local)) <= (0.52 if _pb_label_local != "fast" else 0.44) * float(atr_last))
                    anticipatory_structure_ok_local = bool(
                        float(pb_high_local.iloc[-1]) <= float(pb_high_local.iloc[-2]) + 0.05 * float(atr_last)
                        and float(pb_high_local.iloc[-2]) <= float(pb_high_local.iloc[-3]) + 0.06 * float(atr_last)
                        and float(pb_close_local.iloc[-1]) <= float(pb_close_local.iloc[-2]) + 0.03 * float(atr_last)
                    )

                anticipatory_pressure_ok_local = bool(
                    strong_pressure_local
                    or float(close_stack_q) >= 0.56
                    or float(adx_slope if 'adx_slope' in locals() else 0.0) >= 0.02
                )
                prebreak_score_candidate_local = float(np.clip(
                    0.22 * float(pb_tight_local)
                    + 0.18 * float(pb_vol_tight_local)
                    + 0.18 * float(defended_pb_local)
                    + 0.16 * float(trigger_ready_pb_local)
                    + 0.12 * float(close_stack_pb_local)
                    + 0.14 * float(wick_ok_pb_local)
                    + 0.10 * float(strong_pressure_local)
                    + 0.10 * float(trend_strength_local >= 0.56)
                    + 0.10 * float(legit_local >= 0.58),
                    0.0,
                    1.0,
                ))
                classic_prebreak_ready_candidate_local = bool(
                    (not near_break_local)
                    and (not near_pullback_local)
                    and prefers_breakout_phase_local
                    and pb_tight_local
                    and defended_pb_local
                    and trigger_ready_pb_local
                    and pb_vol_tight_local
                    and wick_ok_pb_local
                    and float(raw_session_risk) <= 3.0
                    and float(impulse_quality or 0.0) >= 0.50
                    and legit_local >= 0.56
                    and trend_strength_local >= 0.56
                    and (strong_pressure_local or float(adx_slope if 'adx_slope' in locals() else 0.0) >= 0.0)
                    and (not bool(multibar_extension_profile_local.get('path_stretched') or False))
                    and (not bool(multibar_extension_profile_local.get('stalling') or False))
                    and (not bool(multibar_extension_profile_local.get('fading') or multibar_extension_profile_local.get('momentum_fade') or False))
                    and prebreak_score_candidate_local >= (0.64 if _pb_label_local == "fast" else 0.66)
                    and dist_br_local <= (1.22 if _pb_label_local == "fast" else 1.32) * max(prox_dist_local, 0.32 * float(atr_last))
                )
                anticipatory_prebreak_ready_candidate_local = bool(
                    prefers_breakout_phase_local
                    and pb_tight_local
                    and defended_pb_local
                    and trigger_ready_pb_local
                    and anticipatory_side_ok_local
                    and anticipatory_dist_ok_local
                    and anticipatory_structure_ok_local
                    and anticipatory_pressure_ok_local
                    and pb_vol_tight_local
                    and wick_ok_pb_local
                    and float(raw_session_risk) <= 3.0
                    and float(impulse_quality or 0.0) >= 0.52
                    and legit_local >= 0.58
                    and trend_strength_local >= 0.58
                    and float(extension_penalty_local) < 0.72
                    and breakout_proximity_ratio_local <= (1.14 if _pb_label_local == "fast" else 1.20)
                    and (not bool(multibar_extension_profile_local.get('path_stretched') or False))
                    and (not bool(multibar_extension_profile_local.get('stalling') or False))
                    and (not bool(multibar_extension_profile_local.get('fading') or multibar_extension_profile_local.get('momentum_fade') or False))
                    and prebreak_score_candidate_local >= (0.62 if _pb_label_local == "fast" else 0.64)
                )
                _prebreak_windows_local[_pb_label_local] = {
                    "ready": bool(classic_prebreak_ready_candidate_local or anticipatory_prebreak_ready_candidate_local),
                    "classic": bool(classic_prebreak_ready_candidate_local),
                    "anticipatory": bool(anticipatory_prebreak_ready_candidate_local),
                    "score": float(prebreak_score_candidate_local),
                }

            if _prebreak_windows_local:
                _best_prebreak_label_local = max(_prebreak_windows_local, key=lambda k: float(_prebreak_windows_local[k].get("score") or 0.0))
                _best_prebreak_local = _prebreak_windows_local[_best_prebreak_label_local]
                prebreak_compression_score_local = float(_best_prebreak_local.get("score") or 0.0)
                prebreak_compression_ready_local = bool(any(bool(v.get("ready") or False) for v in _prebreak_windows_local.values()))
                if any(bool(v.get("anticipatory") or False) for v in _prebreak_windows_local.values()):
                    prebreak_compression_score_local = float(max(prebreak_compression_score_local, 0.68))
                if prebreak_compression_ready_local:
                    prebreak_bias_bonus_local = float(np.clip(0.45 * prebreak_compression_score_local + 0.35 * trend_strength_local + 0.20 * vol_side_q_local, 0.0, 1.0))
                    near_break_local = True
                    stale_breakout_local = False
        except Exception:
            prebreak_compression_ready_local = False
            prebreak_compression_score_local = 0.0
            prebreak_bias_bonus_local = 0.0

        # Enhancement A: persistent trend recognition. Some volatile names do not
        # offer a true pullback once directional control is established. Instead of
        # blindly forcing a deep retrace, allow a controlled continuation lane when
        # the trend is clearly persistent, clean, and still close enough to the break.
        try:
            recent_persist_local = df.tail(int(min(5, len(df)))).copy()
            rp_open_local = pd.to_numeric(recent_persist_local['open'], errors='coerce').dropna()
            rp_close_local = pd.to_numeric(recent_persist_local['close'], errors='coerce').dropna()
            rp_high_local = pd.to_numeric(recent_persist_local['high'], errors='coerce').dropna()
            rp_low_local = pd.to_numeric(recent_persist_local['low'], errors='coerce').dropna()
            if min(len(rp_open_local), len(rp_close_local), len(rp_high_local), len(rp_low_local)) >= 4 and float(atr_last) > 0:
                persist_body_local = float(np.mean(np.abs(rp_close_local.values - rp_open_local.values) / np.maximum(1e-9, rp_high_local.values - rp_low_local.values)))
                persist_range_atr_local = float((float(rp_high_local.max()) - float(rp_low_local.min())) / max(1e-9, float(atr_last)))
                if direction == 'LONG':
                    persist_close_stack_local = bool(float(rp_close_local.iloc[-1]) >= float(rp_close_local.iloc[-2]) - 0.03 * float(atr_last) and float(rp_close_local.iloc[-2]) >= float(rp_close_local.iloc[-3]) - 0.05 * float(atr_last))
                    persist_wick_local = float(np.mean((rp_high_local.values - np.maximum(rp_close_local.values, rp_open_local.values)) / np.maximum(1e-9, rp_high_local.values - rp_low_local.values)))
                    accept_hold_persist_local = bool(float(rp_low_local.tail(int(min(4, len(rp_low_local)))).min()) >= float(accept_line) - 0.14 * float(atr_last))
                else:
                    persist_close_stack_local = bool(float(rp_close_local.iloc[-1]) <= float(rp_close_local.iloc[-2]) + 0.03 * float(atr_last) and float(rp_close_local.iloc[-2]) <= float(rp_close_local.iloc[-3]) + 0.05 * float(atr_last))
                    persist_wick_local = float(np.mean((np.minimum(rp_close_local.values, rp_open_local.values) - rp_low_local.values) / np.maximum(1e-9, rp_high_local.values - rp_low_local.values)))
                    accept_hold_persist_local = bool(float(rp_high_local.tail(int(min(4, len(rp_high_local)))).max()) <= float(accept_line) + 0.14 * float(atr_last))
                persistent_trend_score_local = float(np.clip(
                    0.28 * trend_strength_local
                    + 0.18 * float(np.clip((float(adx_last) - 35.0) / 18.0, 0.0, 1.0))
                    + 0.14 * legit_local
                    + 0.12 * vol_side_q_local
                    + 0.10 * accept_quality_local
                    + 0.08 * float(persist_close_stack_local)
                    + 0.05 * persist_body_local
                    + 0.05 * float(np.clip(1.0 - persist_wick_local, 0.0, 1.0))
                    + 0.08 * float(accept_hold_persist_local),
                    0.0,
                    1.0,
                ))
                persistent_trend_ready_local = bool(
                    (not near_pullback_local)
                    and prefers_breakout_phase_local
                    and trend_strength_local >= 0.70
                    and float(adx_last) >= 38.0
                    and legit_local >= 0.58
                    and vol_side_q_local >= 0.46
                    and accept_quality_local >= 0.52
                    and float(raw_session_risk) <= 3.0
                    and accept_hold_persist_local
                    and persist_close_stack_local
                    and persist_body_local >= 0.42
                    and persist_wick_local <= 0.24
                    and persist_range_atr_local <= 2.6
                    and breakout_proximity_ratio_local <= 1.22
                    and (not stale_breakout_local)
                    and (not bool(multibar_extension_profile_local.get('path_stretched') or False))
                    and (not bool(multibar_extension_profile_local.get('stalling') or False))
                    and (not bool(multibar_extension_profile_local.get('fading') or multibar_extension_profile_local.get('momentum_fade') or False))
                    and persistent_trend_score_local >= 0.68
                )
                if persistent_trend_ready_local:
                    persistent_bias_bonus_local = float(np.clip(0.52 * persistent_trend_score_local + 0.28 * trend_strength_local + 0.20 * vol_side_q_local, 0.0, 1.0))
                    near_break_local = True
            else:
                persistent_trend_score_local = 0.0
        except Exception:
            persistent_trend_ready_local = False
            persistent_trend_score_local = 0.0
            persistent_bias_bonus_local = 0.0

        actionable_local = bool((near_pullback_local or near_break_local or prebreak_compression_ready_local or persistent_trend_ready_local or defense_entry_ready_local) and not stale_breakout_local)
        elite_runaway_local = False

        breakout_ref_local = None
        if direction == "LONG":
            if impulse_type_hint == "ORB" and orb_high is not None:
                breakout_ref_local = float(orb_high)
            elif impulse_type_hint == "VWAP" and ref_vwap is not None:
                breakout_ref_local = float(ref_vwap)
            elif impulse_type_hint == "PIVOT" and swing_hi is not None:
                breakout_ref_local = float(swing_hi)
        else:
            if impulse_type_hint == "ORB" and orb_low is not None:
                breakout_ref_local = float(orb_low)
            elif impulse_type_hint == "VWAP" and ref_vwap is not None:
                breakout_ref_local = float(ref_vwap)
            elif impulse_type_hint == "PIVOT" and swing_lo is not None:
                breakout_ref_local = float(swing_lo)

        recent_break_closes_local = df["close"].astype(float).iloc[-int(min(2, len(df))):]
        breakout_acceptance_quality_local = {"accepted": bool(accept_ok), "clean_accept": bool(accept_ok), "rejection": False, "wick_ratio": 0.0, "close_finish": 0.5, "last_close_vs_ref": 0.0}
        if breakout_ref_local is not None and len(recent_break_closes_local):
            breakout_acceptance_quality_local = _compute_breakout_acceptance_quality(
                df,
                direction=str(direction),
                breakout_ref=float(breakout_ref_local),
                atr_last=float(atr_last) if atr_last is not None else None,
                buffer=float(buffer),
            )
            breakout_acceptance_ok_local = bool(breakout_acceptance_quality_local.get("accepted") or False)
            breakout_clean_accept_local = bool(breakout_acceptance_quality_local.get("clean_accept") or False)
        else:
            breakout_acceptance_ok_local = bool(accept_ok)
            breakout_clean_accept_local = bool(accept_ok)

        # Controlled ignition override:
        # When an elite continuation is clearly running away without offering a normal
        # pullback, allow a breakout-style entry instead of collapsing to CHOP solely
        # because legacy near-entry geometry was missed.
        try:
            last_bar_high_local = float(df["high"].iloc[-1])
            last_bar_low_local = float(df["low"].iloc[-1])
            last_bar_open_local = float(df["open"].iloc[-1])
            last_bar_close_local = float(df["close"].iloc[-1])
            last_bar_range_local = max(1e-9, last_bar_high_local - last_bar_low_local)
            body_ratio_local = abs(last_bar_close_local - last_bar_open_local) / last_bar_range_local
            if direction == "LONG":
                opposite_wick_ratio_local = max(0.0, last_bar_high_local - max(last_bar_open_local, last_bar_close_local)) / last_bar_range_local
                directional_close_local = (last_bar_close_local - last_bar_low_local) / last_bar_range_local
                directional_dom_local = float(di_p or 0.0) - float(di_m or 0.0)
                hold_above_accept_local = bool(float(df["low"].astype(float).tail(int(min(3, len(df)))).min()) >= float(accept_line) - 0.15 * float(atr_last))
                no_immediate_failure_local = bool(last_bar_close_local >= float(break_trigger_local) - 0.10 * float(atr_last))
            else:
                opposite_wick_ratio_local = max(0.0, min(last_bar_open_local, last_bar_close_local) - last_bar_low_local) / last_bar_range_local
                directional_close_local = (last_bar_high_local - last_bar_close_local) / last_bar_range_local
                directional_dom_local = float(di_m or 0.0) - float(di_p or 0.0)
                hold_above_accept_local = bool(float(df["high"].astype(float).tail(int(min(3, len(df)))).max()) <= float(accept_line) + 0.15 * float(atr_last))
                no_immediate_failure_local = bool(last_bar_close_local <= float(break_trigger_local) + 0.10 * float(atr_last))
            vol_impulse_local = bool(vol_ok or (med30 > 0 and float(vol.iloc[-1]) >= 1.8 * float(med30)))
            elite_runaway_local = bool(
                (not actionable_local)
                and (not near_pullback_local)
                and (not prefers_pullback_phase_local)
                and high_conviction_local and float(conviction_score or 0.0) >= 95.0
                and float(impulse_quality or 0.0) >= 0.68
                and float(impulse_legitimacy or 0.0) >= 0.62
                and directional_dom_local >= 8.0
                and vol_impulse_local
                and (structure_phase_detail_local in ("BREAK_AND_HOLD", "EARLY_ACCEPTANCE", "ACCEPTANCE_ATTEMPT", "PERSISTENT_CONTINUATION", "COMPRESSION_BUILDUP", "PRE_COMPRESSION", "RE_COMPRESSION", "RECOVERY_BUILD"))
                and (not bool(multibar_extension_profile_local.get("path_stretched") or False))
                and (not bool(multibar_extension_profile_local.get("stalling") or False))
                and (not bool(multibar_extension_profile_local.get("fading") or False))
                and body_ratio_local >= 0.55
                and directional_close_local >= 0.62
                and opposite_wick_ratio_local <= 0.22
                and hold_above_accept_local
                and no_immediate_failure_local
                and (dist_br_local <= 1.75 * max(prox_dist_local, 0.35 * float(atr_last)))
                and (float(multibar_extension_profile_local.get("dist_accept_atr") or 0.0) <= 2.8)
            )
            if elite_runaway_local:
                near_break_local = True
                stale_breakout_local = False
                actionable_local = True
        except Exception:
            elite_runaway_local = False

        entry_mode_local = None
        entry_price_local = None
        chase_line_local = None
        entry_base_local = None
        tape_metrics_local = {"eligible": False, "readiness": 0.0, "tightening": 0.0, "structural_hold": 0.0, "pressure": 0.0, "release_proximity": 0.0}
        tape_score_bonus_local = 0
        tape_breakout_bias_bonus_local = 0
        tape_prefers_breakout_local = False
        tape_rejection_penalty_local = {"penalty": 0.0, "stuffing": False}
        tape_breakout_urgency_local = {"score": 0.0, "urgent": False}
        tape_pullback_unlikelihood_local = {"score": 0.0, "unlikely": False}
        coiled_continuation_local = {"score": 0.0, "coiled": False, "compression": 0.0, "trend_persist": 0.0, "accept_hold": 0.0, "trigger_prox": 0.0}
        continuation_mode_local = False
        breakout_extension_state_local = {"penalty": float(multibar_extension_profile_local.get("penalty") or 0.0), "extended": False, "exhausted": bool((multibar_extension_profile_local.get("penalty") or 0.0) >= 1.0), "dist_accept_atr": float(multibar_extension_profile_local.get("dist_accept_atr") or 0.0), "dist_vwap_atr": 0.0, "momentum_fade": bool(multibar_extension_profile_local.get("momentum_fade") or multibar_extension_profile_local.get("fading") or False), "stalling": bool(multibar_extension_profile_local.get("stalling") or False), "path_stretched": bool(multibar_extension_profile_local.get("path_stretched") or False)}
        breakout_bias_score_local = 0
        extension_penalty_local = float(breakout_extension_state_local.get("penalty") or 0.0)
        extension_exhausted_local = bool(breakout_extension_state_local.get("exhausted") or False)
        combined_rejection_penalty_local = 0.0
        hard_rejection_local = False
        soft_rejection_local = False
        if actionable_local:
            ext_above_accept = max(0.0, float(last_price - accept_line)) if direction == "LONG" else max(0.0, float(accept_line - last_price))
            ext_above_pullback = max(0.0, float(last_price - pb1_local)) if direction == "LONG" else max(0.0, float(pb1_local - last_price))
            breakout_extension_ok = bool(ext_above_accept <= 0.60 * float(atr_last) and ext_above_pullback <= 0.35 * float(atr_last))
            breakout_elite_pre = bool(stage == "PRE" and (impulse_type_hint in ("ORB", "VWAP")) and impulse_legitimacy >= 0.86 and accept_ok and breakout_acceptance_ok_local and breakout_clean_accept_local and breakout_extension_ok)
            breakout_confirmed_ok = bool(stage == "CONFIRMED" and (impulse_type_hint in ("ORB", "VWAP")) and impulse_legitimacy >= 0.72 and breakout_acceptance_ok_local and breakout_clean_accept_local and breakout_extension_ok)
            breakout_margin_ok = bool(dist_br_local <= (0.75 * max(dist_pb_band_local, 1e-9)))

            breakout_bias_score_local = 0
            try:
                hist_tail_local = pd.to_numeric(df["macd_hist"].tail(int(min(4, len(df)))), errors="coerce").dropna()
            except Exception:
                hist_tail_local = pd.Series(dtype=float)
            if breakout_acceptance_ok_local:
                breakout_bias_score_local += 1
                if breakout_clean_accept_local:
                    breakout_bias_score_local += 1
            if float(adx_modifier or 0.0) >= 4.0:
                breakout_bias_score_local += 2
            elif float(adx_modifier or 0.0) > 0.0:
                breakout_bias_score_local += 1
            if len(hist_tail_local) >= 3:
                if direction == "LONG":
                    if hist_tail_local.iloc[-1] > hist_tail_local.iloc[-2] > hist_tail_local.iloc[-3] and hist_tail_local.iloc[-1] > 0:
                        breakout_bias_score_local += 2
                    elif hist_tail_local.iloc[-1] > hist_tail_local.iloc[-2] > hist_tail_local.iloc[-3]:
                        breakout_bias_score_local += 1
                else:
                    if hist_tail_local.iloc[-1] < hist_tail_local.iloc[-2] < hist_tail_local.iloc[-3] and hist_tail_local.iloc[-1] < 0:
                        breakout_bias_score_local += 2
                    elif hist_tail_local.iloc[-1] < hist_tail_local.iloc[-2] < hist_tail_local.iloc[-3]:
                        breakout_bias_score_local += 1
            recent_hold_closes_local = df["close"].astype(float).tail(int(min(3, len(df))))
            if direction == "LONG":
                shallow_retests_local = bool(float(df["low"].astype(float).tail(int(min(4, len(df)))).min()) >= float(accept_line) - 0.18 * float(atr_last))
                hold_progress_local = bool((recent_hold_closes_local >= float(accept_line) - buffer).all())
            else:
                shallow_retests_local = bool(float(df["high"].astype(float).tail(int(min(4, len(df)))).max()) <= float(accept_line) + 0.18 * float(atr_last))
                hold_progress_local = bool((recent_hold_closes_local <= float(accept_line) + buffer).all())
            if shallow_retests_local and hold_progress_local:
                breakout_bias_score_local += 1
            if near_break_local and dist_pb_band_local > max(1.10 * prox_dist_local, 0.28 * float(atr_last)):
                breakout_bias_score_local += 1
            if breakout_proximity_score_local >= 0.95:
                breakout_bias_score_local += 2
            elif breakout_proximity_score_local >= 0.75:
                breakout_bias_score_local += 1
            elif breakout_proximity_score_local <= 0.10:
                breakout_bias_score_local -= 2
            elif breakout_proximity_penalty_local > 0.0:
                breakout_bias_score_local -= 1

            if tape_mode_enabled:
                tape_metrics_local = _compute_tape_readiness(
                    df,
                    direction=str(direction),
                    atr_last=float(atr_last) if atr_last is not None else None,
                    release_level=float(break_trigger_local),
                    structural_level=float(accept_line),
                    trigger_near=bool(near_break_local or (breakout_proximity_ratio_local <= 1.20)),
                    baseline_ok=bool(
                        (impulse_quality >= 0.45)
                        and (impulse_legitimacy >= 0.52)
                        and (accept_ok or hold_ok or breakout_acceptance_ok_local)
                        and (not stale_breakout_local)
                    ),
                )
                tape_score_bonus_local = _tape_bonus_from_readiness(
                    float(tape_metrics_local.get("readiness") or 0.0),
                    cap=4,
                    thresholds=(5.0, 6.0, 7.0, 8.0),
                )
                tape_rejection_penalty_local = _compute_release_rejection_penalty(
                    df,
                    direction=str(direction),
                    atr_last=float(atr_last) if atr_last is not None else None,
                    release_level=float(break_trigger_local),
                )
                tape_breakout_urgency_local = _compute_breakout_urgency(
                    df,
                    direction=str(direction),
                    atr_last=float(atr_last) if atr_last is not None else None,
                    release_level=float(break_trigger_local),
                )
                tape_pullback_unlikelihood_local = _compute_pullback_unlikelihood(
                    df,
                    direction=str(direction),
                    atr_last=float(atr_last) if atr_last is not None else None,
                    accept_line=float(accept_line),
                )
                breakout_extension_state_local = _compute_breakout_extension_state(
                    df,
                    direction=str(direction),
                    atr_last=float(atr_last) if atr_last is not None else None,
                    accept_line=float(accept_line),
                    ref_vwap=float(ref_vwap) if ref_vwap is not None else None,
                )
                breakout_extension_state_local["penalty"] = float(min(1.5, float(breakout_extension_state_local.get("penalty") or 0.0) + 0.55 * float(multibar_extension_profile_local.get("penalty") or 0.0)))
                breakout_extension_state_local["stalling"] = bool(breakout_extension_state_local.get("stalling") or multibar_extension_profile_local.get("stalling"))
                breakout_extension_state_local["momentum_fade"] = bool(breakout_extension_state_local.get("momentum_fade") or multibar_extension_profile_local.get("fading"))
                breakout_extension_state_local["path_stretched"] = bool(multibar_extension_profile_local.get("path_stretched") or False)
                acceptance_rejection_flag_local = bool(breakout_acceptance_quality_local.get("rejection") or False)
                acceptance_wick_local = float(breakout_acceptance_quality_local.get("wick_ratio") or 0.0)
                acceptance_close_finish_local = float(breakout_acceptance_quality_local.get("close_finish") or 0.5)
                acceptance_penalty_local = 0.0
                if acceptance_rejection_flag_local:
                    acceptance_penalty_local = 0.65 if (acceptance_wick_local < 0.42 and acceptance_close_finish_local >= 0.42) else 1.0
                combined_rejection_penalty_local = float(max(
                    float(tape_rejection_penalty_local.get("penalty") or 0.0),
                    acceptance_penalty_local,
                ))
                extension_penalty_local = float(breakout_extension_state_local.get("penalty") or 0.0)
                extension_exhausted_local = bool(breakout_extension_state_local.get("exhausted") or False)
                hard_rejection_local = bool(combined_rejection_penalty_local >= 0.95)
                soft_rejection_local = bool((combined_rejection_penalty_local >= 0.35) and not hard_rejection_local)
                tape_breakout_ready_local = bool(
                    float(tape_metrics_local.get("readiness") or 0.0) >= 6.0
                    and float(tape_metrics_local.get("pressure") or 0.0) >= 1.25
                    and float(tape_metrics_local.get("release_proximity") or 0.0) >= 1.0
                    and near_break_local
                    and shallow_retests_local
                    and hold_progress_local
                    and breakout_acceptance_ok_local
                    and breakout_clean_accept_local
                    and breakout_extension_ok
                    and not stale_breakout_local
                    and not hard_rejection_local
                    and combined_rejection_penalty_local <= 0.65
                    and extension_penalty_local < 1.0
                    and not extension_exhausted_local
                )
                tape_tiebreak_breakout_local = bool(
                    tape_breakout_ready_local
                    and float(tape_breakout_urgency_local.get("score") or 0.0) >= 1.0
                    and float(tape_pullback_unlikelihood_local.get("score") or 0.0) >= 1.0
                    and breakout_proximity_ratio_local <= 0.85
                )
                if tape_breakout_ready_local and bool(tape_breakout_urgency_local.get("urgent") or False):
                    tape_breakout_bias_bonus_local = 1
                if hard_rejection_local:
                    breakout_bias_score_local -= 2
                elif soft_rejection_local:
                    breakout_bias_score_local -= 1
                if extension_penalty_local >= 1.0:
                    breakout_bias_score_local -= 2
                elif extension_penalty_local >= 0.5:
                    breakout_bias_score_local -= 1
                breakout_bias_score_local += int(tape_breakout_bias_bonus_local)
                tape_prefers_breakout_local = bool(
                    tape_tiebreak_breakout_local
                    and bool(tape_pullback_unlikelihood_local.get("unlikely") or False)
                    and not hard_rejection_local
                    and combined_rejection_penalty_local < 0.80
                    and extension_penalty_local < 1.0
                    and not extension_exhausted_local
                )

                try:
                    if direction == "LONG":
                        no_real_pullback_local = bool(last_price >= pb1_local + 0.12 * float(atr_last))
                        accept_hold_local = bool(float(df["low"].astype(float).tail(int(min(4, len(df)))).min()) >= float(accept_line) - 0.15 * float(atr_last))
                    else:
                        no_real_pullback_local = bool(last_price <= pb1_local - 0.12 * float(atr_last))
                        accept_hold_local = bool(float(df["high"].astype(float).tail(int(min(4, len(df)))).max()) <= float(accept_line) + 0.15 * float(atr_last))
                    coiled_continuation_local = _assess_coiled_continuation(
                        direction=str(direction),
                        df=df,
                        accept_line=float(accept_line),
                        break_trigger=float(break_trigger_local),
                        atr_last=float(atr_last) if atr_last is not None else None,
                    )
                    standard_continuation_ok_local = bool(
                        (not elite_runaway_local)
                        and (not hard_rejection_local)
                        and (not extension_exhausted_local)
                        and float(extension_penalty_local) < 1.0
                        and breakout_acceptance_ok_local
                        and accept_hold_local
                        and (structure_phase_detail_local in ("BREAK_AND_HOLD", "EARLY_ACCEPTANCE", "ACCEPTANCE_ATTEMPT", "PERSISTENT_CONTINUATION", "COMPRESSION_BUILDUP", "PRE_COMPRESSION", "RE_COMPRESSION", "RECOVERY_BUILD", "EXTEND_THEN_PULLBACK", "STALLING_EXTENSION", "MATURE_ACCEPTANCE"))
                        and float(impulse_quality or 0.0) >= 0.60
                        and float(impulse_legitimacy or 0.0) >= 0.58
                        and float(tape_breakout_urgency_local.get("score") or 0.0) >= 1.0
                        and float(tape_pullback_unlikelihood_local.get("score") or 0.0) >= 1.25
                        and no_real_pullback_local
                        and dist_br_local <= 1.35 * max(prox_dist_local, 0.35 * float(atr_last))
                    )
                    coiled_continuation_ok_local = bool(
                        (not elite_runaway_local)
                        and (not hard_rejection_local)
                        and (not extension_exhausted_local)
                        and float(extension_penalty_local) < 0.85
                        and breakout_acceptance_ok_local
                        and accept_hold_local
                        and (structure_phase_detail_local in ("BREAK_AND_HOLD", "EARLY_ACCEPTANCE", "ACCEPTANCE_ATTEMPT", "PERSISTENT_CONTINUATION", "COMPRESSION_BUILDUP", "PRE_COMPRESSION", "RE_COMPRESSION", "RECOVERY_BUILD"))
                        and float(impulse_quality or 0.0) >= 0.44
                        and float(impulse_legitimacy or 0.0) >= 0.52
                        and float(tape_breakout_urgency_local.get("score") or 0.0) >= 0.85
                        and float(tape_pullback_unlikelihood_local.get("score") or 0.0) >= 0.95
                        and bool(coiled_continuation_local.get("coiled") or False)
                        and float(coiled_continuation_local.get("score") or 0.0) >= 0.60
                        and no_real_pullback_local
                        and dist_br_local <= 1.15 * max(prox_dist_local, 0.35 * float(atr_last))
                    )
                    continuation_mode_local = bool(standard_continuation_ok_local or coiled_continuation_ok_local)
                except Exception:
                    continuation_mode_local = False
                    coiled_continuation_local = {"score": 0.0, "coiled": False, "compression": 0.0, "trend_persist": 0.0, "accept_hold": 0.0, "trigger_prox": 0.0}
                if continuation_mode_local:
                    near_break_local = True
                    stale_breakout_local = False
                    actionable_local = True

                # Ignition continuation lane: some high-quality early acceptance breakouts
                # continue immediately without printing a tradable pullback. Recognize that
                # structure and allow a controlled continuation signal rather than silently
                # missing it.
                try:
                    if direction == 'LONG':
                        recent_lows_ign_local = pd.to_numeric(df['low'].tail(int(min(3, len(df)))), errors='coerce').dropna()
                        ignition_hold_local = bool(len(recent_lows_ign_local) and float(recent_lows_ign_local.min()) >= float(accept_line) - 0.14 * float(atr_last))
                        ignition_rejection_local = float(np.clip(rejection_wick_local, 0.0, 1.0))
                    else:
                        recent_highs_ign_local = pd.to_numeric(df['high'].tail(int(min(3, len(df)))), errors='coerce').dropna()
                        ignition_hold_local = bool(len(recent_highs_ign_local) and float(recent_highs_ign_local.max()) <= float(accept_line) + 0.14 * float(atr_last))
                        ignition_rejection_local = float(np.clip(rejection_wick_local, 0.0, 1.0))
                    ignition_continuation_score_local = float(np.clip(
                        0.14 * float(structure_phase_detail_local in ('EARLY_ACCEPTANCE', 'ACCEPTANCE_ATTEMPT')) + 0.10 * float(structure_phase_detail_local in ('COMPRESSION_BUILDUP', 'PRE_COMPRESSION', 'RE_COMPRESSION', 'RECOVERY_BUILD'))
                        + 0.18 * float(np.clip(structure_phase_confidence_local, 0.0, 1.0))
                        + 0.14 * float(np.clip(trend_strength_local, 0.0, 1.0))
                        + 0.12 * float(np.clip(impulse_quality, 0.0, 1.0))
                        + 0.10 * float(np.clip(impulse_legitimacy, 0.0, 1.0))
                        + 0.10 * float(np.clip(vol_side_q_local, 0.0, 1.0))
                        + 0.08 * float(np.clip(close_stack_q, 0.0, 1.0))
                        + 0.08 * float(np.clip(persistence_q, 0.0, 1.0))
                        + 0.08 * float(ignition_hold_local)
                        + 0.06 * float(np.clip(1.0 - ignition_rejection_local, 0.0, 1.0)),
                        0.0,
                        1.0,
                    ))
                    ignition_continuation_ready_local = bool(
                        (not actionable_local)
                        and (not near_pullback_local)
                        and (not stale_breakout_local)
                        and str(structure_phase_detail_local) in ('EARLY_ACCEPTANCE', 'ACCEPTANCE_ATTEMPT', 'COMPRESSION_BUILDUP', 'PRE_COMPRESSION', 'RE_COMPRESSION', 'RECOVERY_BUILD')
                        and float(structure_phase_confidence_local) >= 0.70
                        and float(trend_strength_local) >= 0.58
                        and float(impulse_quality or 0.0) >= 0.48
                        and float(impulse_legitimacy or 0.0) >= 0.52
                        and float(vol_side_q_local) >= 0.48
                        and float(close_stack_q) >= 0.52
                        and float(persistence_q) >= 0.50
                        and ignition_hold_local
                        and (not hard_rejection_local)
                        and (not extension_exhausted_local)
                        and float(extension_penalty_local) < 0.90
                        and breakout_proximity_ratio_local <= 1.18
                        and ignition_continuation_score_local >= 0.68
                    )
                    if ignition_continuation_ready_local:
                        ignition_bias_bonus_local = float(np.clip(
                            0.46 * ignition_continuation_score_local + 0.30 * trend_strength_local + 0.24 * vol_side_q_local,
                            0.0,
                            1.0,
                        ))
                        near_break_local = True
                        actionable_local = True
                except Exception:
                    ignition_continuation_ready_local = False
                    ignition_continuation_score_local = 0.0
                    ignition_bias_bonus_local = 0.0

            breakout_bias = bool((breakout_confirmed_ok or breakout_elite_pre) and near_break_local and not stale_breakout_local and breakout_margin_ok and breakout_proximity_ratio_local <= 1.18)
            if breakout_bias and (hard_rejection_local or extension_exhausted_local or float(extension_penalty_local) >= 1.0):
                breakout_bias = False
            if not breakout_bias:
                breakout_bias = bool(
                    near_break_local
                    and not stale_breakout_local
                    and breakout_bias_score_local >= (5 if stage == "PRE" else 4)
                    and breakout_acceptance_ok_local
                    and (breakout_clean_accept_local or breakout_bias_score_local >= 6)
                    and not hard_rejection_local
                    and not extension_exhausted_local
                    and float(extension_penalty_local) < 1.0
                    and breakout_proximity_ratio_local <= 1.32
                )
            if not breakout_bias and tape_prefers_breakout_local and near_break_local and (near_pullback_local or breakout_bias_score_local >= 4) and not hard_rejection_local and not extension_exhausted_local and float(extension_penalty_local) < 1.0 and breakout_proximity_ratio_local <= 1.22:
                breakout_bias = True
            if not breakout_bias and prebreak_compression_ready_local and near_break_local and not hard_rejection_local and float(extension_penalty_local) < 0.90 and breakout_proximity_ratio_local <= 1.28:
                breakout_bias = True
                breakout_bias_score_local += 1
            if not breakout_bias and compression_breakout_ready_local and near_break_local and not hard_rejection_local and float(extension_penalty_local) < 0.92 and breakout_proximity_ratio_local <= 1.22:
                breakout_bias = True
                breakout_bias_score_local += 1
            if not breakout_bias and persistent_trend_ready_local and near_break_local and not hard_rejection_local and float(extension_penalty_local) < 0.85 and breakout_proximity_ratio_local <= 1.24:
                breakout_bias = True
                breakout_bias_score_local += 1
            if not breakout_bias and continuation_mode_local:
                breakout_bias = True
            if not breakout_bias and ignition_continuation_ready_local and near_break_local and not hard_rejection_local and float(extension_penalty_local) < 0.90 and breakout_proximity_ratio_local <= 1.18:
                breakout_bias = True
                breakout_bias_score_local += 1
            if breakout_bias and ((soft_rejection_local and near_pullback_local and dist_pb_band_local <= 1.10 * prox_dist_local) or extension_exhausted_local or (float(extension_penalty_local) >= 1.0 and near_pullback_local)):
                breakout_bias = False

            phase_no_trade_local = bool(str(structure_phase_detail_local) in ("FAILED_EXTENSION",))
            mature_acceptance_local = bool(str(structure_phase_detail_local) == "MATURE_ACCEPTANCE")
            breakout_fresh_phase_local = bool(str(structure_phase_detail_local) in ("BREAK_AND_HOLD", "EARLY_ACCEPTANCE", "ACCEPTANCE_ATTEMPT", "PERSISTENT_CONTINUATION", "COMPRESSION_BUILDUP", "PRE_COMPRESSION", "RE_COMPRESSION", "RECOVERY_BUILD"))
            mature_compression_ready_local = bool(
                mature_acceptance_local
                and float(structure_phase_confidence_local) >= 0.58
                and not hard_rejection_local
                and not soft_rejection_local
                and not extension_exhausted_local
                and breakout_proximity_ratio_local <= 1.14
                and float(close_stack_q) >= 0.52
                and float(persistence_q) >= 0.48
                and float(vol_side_q_local) >= 0.44
                and (
                    (bool(near_break_local) and float(extension_penalty_local) < 0.92)
                    or (bool(prebreak_compression_ready_local))
                    or (0.38 <= float(dist_pb_band_local) / max(1e-9, float(prox_dist_local)) <= 1.05)
                )
            )
            phase_momentum_ready_local = bool(
                breakout_fresh_phase_local
                and float(structure_phase_confidence_local) >= (0.68 if str(structure_phase_detail_local) == "BREAK_AND_HOLD" else 0.70)
                and not hard_rejection_local
                and not extension_exhausted_local
                and float(extension_penalty_local) < 0.92
                and breakout_proximity_ratio_local <= (1.28 if str(structure_phase_detail_local) in ("PERSISTENT_CONTINUATION", "COMPRESSION_BUILDUP", "PRE_COMPRESSION", "RE_COMPRESSION", "RECOVERY_BUILD") else 1.18)
                and (
                    bool(ignition_continuation_ready_local)
                    or bool(persistent_trend_ready_local)
                    or bool(prebreak_compression_ready_local)
                    or bool(compression_breakout_ready_local)
                    or (bool(breakout_bias) and bool(near_break_local) and float(persistence_q) >= 0.50 and float(close_stack_q) >= 0.50)
                )
            )

            if phase_no_trade_local:
                choose_pullback = False
            elif defense_entry_ready_local:
                choose_pullback = True
            elif elite_runaway_local or continuation_mode_local or persistent_trend_ready_local or ignition_continuation_ready_local:
                choose_pullback = False
            elif mature_compression_ready_local:
                choose_pullback = False
            elif near_pullback_local and not near_break_local:
                choose_pullback = True
            elif near_break_local and not near_pullback_local:
                choose_pullback = bool((not breakout_bias) or prefers_pullback_phase_local)
            elif near_pullback_local and near_break_local:
                if prefers_breakout_phase_local and breakout_bias and not extension_exhausted_local and float(extension_penalty_local) < 1.0:
                    choose_pullback = False
                elif prefers_pullback_phase_local:
                    choose_pullback = True
                else:
                    choose_pullback = bool(not breakout_bias)
            else:
                choose_pullback = bool(near_pullback_local or prefers_pullback_phase_local)

            if acceleration_ready_local and breakout_bias and near_break_local and not near_pullback_local and breakout_proximity_ratio_local <= (1.10 + 0.15 * acceleration_boost_local) and float(extension_penalty_local) < 0.75:
                choose_pullback = False
            if prebreak_compression_ready_local and breakout_bias and near_break_local and float(extension_penalty_local) < 0.85 and breakout_proximity_ratio_local <= 1.28:
                choose_pullback = False
            if compression_breakout_ready_local and breakout_bias and near_break_local and float(extension_penalty_local) < 0.92 and breakout_proximity_ratio_local <= 1.22:
                choose_pullback = False
            if persistent_trend_ready_local and breakout_bias and near_break_local and float(extension_penalty_local) < 0.85 and breakout_proximity_ratio_local <= 1.24:
                choose_pullback = False
            if ignition_continuation_ready_local and breakout_bias and near_break_local and float(extension_penalty_local) < 0.90 and breakout_proximity_ratio_local <= 1.18:
                choose_pullback = False
            if phase_momentum_ready_local and not defense_entry_ready_local:
                choose_pullback = False

            if phase_no_trade_local:
                actionable_local = False
                entry_mode_local = "NO_TRADE"
                entry_base_local = None
                entry_price_local = None
                chase_line_local = None
            elif choose_pullback:
                entry_mode_local = "PULLBACK"
                entry_base_local = float(pullback_entry_local)
                entry_price_local = float(entry_base_local + entry_pad) if direction == "LONG" else float(entry_base_local - entry_pad)
                chase_line_local = float(break_trigger_local + entry_pad) if direction == "LONG" else float(break_trigger_local - entry_pad)
            else:
                if phase_momentum_ready_local:
                    entry_mode_local = "MOMENTUM"
                    if str(structure_phase_detail_local) == 'MATURE_ACCEPTANCE':
                        if direction == "LONG":
                            momentum_anchor_local = float(np.clip(last_price - 0.04 * atr_last, accept_line + 0.03 * atr_last, break_trigger_local + 0.08 * atr_last))
                        else:
                            momentum_anchor_local = float(np.clip(last_price + 0.04 * atr_last, break_trigger_local - 0.08 * atr_last, accept_line - 0.03 * atr_last))
                    else:
                        if direction == "LONG":
                            momentum_anchor_local = float(np.clip(last_price - 0.06 * atr_last, accept_line + 0.02 * atr_last, break_trigger_local + 0.10 * atr_last))
                        else:
                            momentum_anchor_local = float(np.clip(last_price + 0.06 * atr_last, break_trigger_local - 0.10 * atr_last, accept_line - 0.02 * atr_last))
                    entry_base_local = float(momentum_anchor_local)
                    entry_price_local = float(entry_base_local + entry_pad) if direction == "LONG" else float(entry_base_local - entry_pad)
                    chase_line_local = float(entry_price_local + (0.08 if str(structure_phase_detail_local) != 'MATURE_ACCEPTANCE' else 0.06) * atr_last + entry_pad) if direction == "LONG" else float(entry_price_local - (0.08 if str(structure_phase_detail_local) != 'MATURE_ACCEPTANCE' else 0.06) * atr_last - entry_pad)
                else:
                    entry_mode_local = "BREAKOUT"
                    if prebreak_compression_ready_local:
                        if direction == "LONG":
                            entry_base_local = float(np.clip(last_price, break_trigger_local - 0.24 * atr_last, break_trigger_local - 0.03 * atr_last))
                        else:
                            entry_base_local = float(np.clip(last_price, break_trigger_local + 0.03 * atr_last, break_trigger_local + 0.24 * atr_last))
                    elif compression_breakout_ready_local:
                        if direction == "LONG":
                            entry_base_local = float(np.clip(last_price, break_trigger_local - 0.10 * atr_last, break_trigger_local + 0.04 * atr_last))
                        else:
                            entry_base_local = float(np.clip(last_price, break_trigger_local - 0.04 * atr_last, break_trigger_local + 0.10 * atr_last))
                    elif persistent_trend_ready_local:
                        if direction == "LONG":
                            entry_base_local = float(np.clip(last_price, break_trigger_local - 0.06 * atr_last, break_trigger_local + 0.08 * atr_last))
                        else:
                            entry_base_local = float(np.clip(last_price, break_trigger_local - 0.08 * atr_last, break_trigger_local + 0.06 * atr_last))
                    else:
                        entry_base_local = float(max(break_trigger_local, last_price)) if direction == "LONG" else float(min(break_trigger_local, last_price))
                    entry_price_local = float(entry_base_local + entry_pad) if direction == "LONG" else float(entry_base_local - entry_pad)
                    if prebreak_compression_ready_local:
                        chase_line_local = float(break_trigger_local + 0.08 * atr_last + entry_pad) if direction == "LONG" else float(break_trigger_local - 0.08 * atr_last - entry_pad)
                    elif compression_breakout_ready_local:
                        chase_line_local = float(break_trigger_local + 0.10 * atr_last + entry_pad) if direction == "LONG" else float(break_trigger_local - 0.10 * atr_last - entry_pad)
                    elif persistent_trend_ready_local:
                        chase_line_local = float(break_trigger_local + 0.12 * atr_last + entry_pad) if direction == "LONG" else float(break_trigger_local - 0.12 * atr_last - entry_pad)
                    else:
                        chase_line_local = float(entry_price_local + 0.10 * atr_last + entry_pad) if direction == "LONG" else float(entry_price_local - 0.10 * atr_last - entry_pad)
                if direction == "LONG":
                    stop_local = float(min(stop_local, accept_line - 0.80 * atr_last))
                else:
                    stop_local = float(max(stop_local, accept_line + 0.80 * atr_last))

        return {
            "high_conviction": bool(high_conviction_local),
            "accept_quality": float(accept_quality_local),
            "accept_src_confidence": float(src_conf_local),
            "accept_extension_ratio": float(extension_loose_local),
            "pb_inner_mult": float(pb_inner_mult_local),
            "pb_outer_mult": float(pb_outer_mult_local),
            "dynamic_shallow_bias": float(dynamic_shallow_bias_local),
            "dynamic_deep_bias": float(dynamic_deep_bias_local),
            "late_trend_shallow_bias": float(late_trend_shallow_bias_local),
            "persistent_hold_quality": float(persistent_hold_quality_local),
            "trend_strength_local": float(trend_strength_local),
            "acceleration_ready": bool(acceleration_ready_local),
            "persistent_trend_ready": bool(persistent_trend_ready_local),
            "defense_entry_ready": bool(defense_entry_ready_local),
            "defense_entry_score": float(defense_entry_score_local),
            "persistent_trend_score": float(persistent_trend_score_local),
            "persistent_bias_bonus": float(persistent_bias_bonus_local),
            "ignition_continuation_ready": bool(ignition_continuation_ready_local),
            "ignition_continuation_score": float(ignition_continuation_score_local),
            "ignition_bias_bonus": float(ignition_bias_bonus_local),
            "adaptive_pullback_refine": float(adaptive_pullback_refine_local),
            "phase_pullback_tighten": float(locals().get("phase_pullback_tighten_local", 0.0)),
            "phase_pullback_anchor": float(locals().get("phase_pullback_anchor_local", accept_line)),
            "phase_momentum_ready": bool(locals().get("phase_momentum_ready_local", False)),
            "mature_compression_ready": bool(locals().get("mature_compression_ready_local", False)),
            "phase_no_trade": bool(locals().get("phase_no_trade_local", False)),
            "acceleration_boost": float(acceleration_boost_local),
            "prebreak_compression_ready": bool(prebreak_compression_ready_local),
            "prebreak_compression_score": float(prebreak_compression_score_local),
            "compression_breakout_ready": bool(locals().get("compression_breakout_ready_local", False)),
            "compression_breakout_score": float(locals().get("compression_breakout_score_local", 0.0)),
            "prebreak_bias_bonus": float(prebreak_bias_bonus_local),
            "breakout_proximity_ratio": float(breakout_proximity_ratio_local),
            "breakout_proximity_score": float(breakout_proximity_score_local),
            "breakout_proximity_penalty": float(breakout_proximity_penalty_local),
            "breakout_proximity_bucket": str(breakout_proximity_bucket_local),
            "break_trigger": float(break_trigger_local),
            "pb1": float(pb1_local),
            "pb2": float(pb2_local),
            "pullback_entry": float(pullback_entry_local),
            "stop": float(stop_local),
            "dist_pb_band": float(dist_pb_band_local),
            "dist_br": float(dist_br_local),
            "near_pullback": bool(near_pullback_local),
            "near_break": bool(near_break_local),
            "actionable": bool(actionable_local),
            "entry_mode": entry_mode_local,
            "entry_price": entry_price_local,
            "chase_line": chase_line_local,
            "breakout_acceptance_ok": bool(breakout_acceptance_ok_local),
            "breakout_bias_score": int(breakout_bias_score_local),
            "structure_phase": structure_phase_local,
            "structure_phase_detail": structure_phase_detail_local,
            "structure_phase_confidence": float(structure_phase_confidence_local),
            "structure_phase_interpretation": structure_phase_interpretation_local,
            "elite_runaway": bool(elite_runaway_local),
            "continuation_mode": bool(continuation_mode_local),
            "extension_profile": multibar_extension_profile_local,
            "tape_readiness": float(tape_metrics_local.get("readiness") or 0.0),
            "tape_tightening": float(tape_metrics_local.get("tightening") or 0.0),
            "tape_hold": float(tape_metrics_local.get("structural_hold") or 0.0),
            "tape_pressure": float(tape_metrics_local.get("pressure") or 0.0),
            "tape_release_proximity": float(tape_metrics_local.get("release_proximity") or 0.0),
            "tape_score_bonus": int(tape_score_bonus_local),
            "tape_breakout_bias_bonus": int(tape_breakout_bias_bonus_local),
            "tape_prefers_breakout": bool(tape_prefers_breakout_local),
            "tape_rejection_penalty": float(tape_rejection_penalty_local.get("penalty") or 0.0),
            "tape_stuffing": bool(tape_rejection_penalty_local.get("stuffing") or False),
            "tape_breakout_urgency": float(tape_breakout_urgency_local.get("score") or 0.0),
            "tape_pullback_unlikelihood": float(tape_pullback_unlikelihood_local.get("score") or 0.0),
            "breakout_soft_rejection": bool(locals().get("soft_rejection_local", False)),
            "breakout_hard_rejection": bool(locals().get("hard_rejection_local", False)),
            "breakout_extension_penalty": float(breakout_extension_state_local.get("penalty") or 0.0),
            "breakout_extended": bool(breakout_extension_state_local.get("extended") or False),
            "breakout_exhausted": bool(breakout_extension_state_local.get("exhausted") or False),
            "breakout_dist_accept_atr": float(breakout_extension_state_local.get("dist_accept_atr") or 0.0),
            "breakout_dist_vwap_atr": float(breakout_extension_state_local.get("dist_vwap_atr") or 0.0),
            "breakout_momentum_fade": bool(breakout_extension_state_local.get("momentum_fade") or False),
            "breakout_stalling": bool(breakout_extension_state_local.get("stalling") or False),
            "prox_atr_local": float(prox_atr_local),
        }

    provisional_geometry = _build_ride_entry_geometry(score)
    ride_tape_bonus = int(provisional_geometry.get("tape_score_bonus") or 0) if tape_mode_enabled else 0
    if ride_tape_bonus:
        pts += float(ride_tape_bonus)
    ride_entry_context_px = float(provisional_geometry["entry_price"]) if isinstance(provisional_geometry.get("entry_price"), (float, int)) else (float(provisional_geometry["pullback_entry"]) if isinstance(provisional_geometry.get("pullback_entry"), (float, int)) else float(provisional_geometry["break_trigger"]))
    provisional_macd_phase = _ride_macd_phase_utility(
        direction=str(direction),
        detail_phase=str(provisional_geometry.get("structure_phase_detail") or provisional_geometry.get("structure_phase") or 'UNSET'),
        macd_info=ride_macd_info,
        entry_mode=provisional_geometry.get("entry_mode"),
    )

    ride_zone_ctx = _evaluate_entry_zone_context(
        df, entry_price=ride_entry_context_px, direction=str(direction), atr_last=float(atr_last) if atr_last is not None else None, lookback=10
    )
    ride_zone_adj = 0.0
    fav_q = float(ride_zone_ctx.get("favorable_quality") or 0.0)
    host_q = float(ride_zone_ctx.get("hostile_quality") or 0.0)
    provisional_phase = str(provisional_geometry.get("structure_phase") or "UNSET")
    if bool(ride_zone_ctx.get("favorable")):
        ride_zone_adj += 4.0 + 3.0 * fav_q
        if provisional_phase in ("EXTEND_THEN_PULLBACK", "FAILED_EXTENSION") and str(provisional_geometry.get("entry_mode") or '').upper() == 'PULLBACK':
            ride_zone_adj += 1.0
    if bool(ride_zone_ctx.get("hostile")):
        hostile_pen = 6.0 + 4.0 * host_q
        if str(provisional_geometry.get("entry_mode") or '').upper() == 'BREAKOUT':
            hostile_pen += (2.5 + 2.0 * host_q)
        if provisional_phase in ("BREAK_AND_HOLD", "ACCEPT_AND_GO") and str(provisional_geometry.get("entry_mode") or '').upper() == 'BREAKOUT':
            hostile_pen += 1.0
        ride_zone_adj -= hostile_pen
    ride_cont_strength_adj = _ride_continuation_strength_adjustment(
        provisional_phase,
        provisional_geometry.get("entry_mode"),
        provisional_geometry.get("tape_breakout_urgency"),
        provisional_geometry.get("tape_pullback_unlikelihood"),
    )
    provisional_cont_phase3_adj, provisional_cont_phase3_note = _ride_phase3_continuation_adjustment(
        direction=str(direction),
        adx_ctx=adx_ctx,
        structure_phase=provisional_phase,
        entry_mode=provisional_geometry.get("entry_mode"),
        legitimacy=float(impulse_legitimacy),
        vwap_score=float(vwap_score_up if direction == "LONG" else vwap_score_dn),
        pivot_score=float(pivot_score_up if direction == "LONG" else pivot_score_dn),
        orb_score=float(orb_score_up if direction == "LONG" else orb_score_dn),
    )
    provisional_pressure_adj, provisional_pressure_note = _ride_indicator_pressure_adjustment(
        direction=str(direction),
        detail_phase=str(provisional_geometry.get("structure_phase_detail") or provisional_geometry.get("structure_phase") or "UNSET"),
        entry_mode=provisional_geometry.get("entry_mode"),
        adx_ctx=adx_ctx,
        macd_info=ride_macd_info,
        pressure_states=ride_pressure_states,
    )
    score = _cap_score(pts + ride_zone_adj + ride_cont_strength_adj + adx_modifier + provisional_cont_phase3_adj + provisional_pressure_adj + float(provisional_macd_phase.get("score_adj") or 0.0))

    final_geometry = _build_ride_entry_geometry(score)
    break_trigger = float(final_geometry["break_trigger"])
    pb1 = float(final_geometry["pb1"])
    pb2 = float(final_geometry["pb2"])
    pullback_entry = float(final_geometry["pullback_entry"])
    stop = float(final_geometry["stop"])
    dist_pb_band = float(final_geometry["dist_pb_band"])
    dist_br = float(final_geometry["dist_br"])
    near_pullback = bool(final_geometry["near_pullback"])
    near_break = bool(final_geometry["near_break"])
    actionable = bool(final_geometry["actionable"])
    entry_mode = final_geometry.get("entry_mode")
    entry_price = final_geometry.get("entry_price")
    chase_line = final_geometry.get("chase_line")
    breakout_acceptance_ok = bool(final_geometry["breakout_acceptance_ok"])
    high_conviction = bool(final_geometry["high_conviction"])
    pb_inner_mult = float(final_geometry["pb_inner_mult"])
    pb_outer_mult = float(final_geometry["pb_outer_mult"])


    ride_macd_phase = _ride_macd_phase_utility(
        direction=str(direction),
        detail_phase=str(final_geometry.get("structure_phase_detail") or final_geometry.get("structure_phase") or 'UNSET'),
        macd_info=ride_macd_info,
        entry_mode=entry_mode,
    )
    if bool(ride_macd_phase.get('hard_block') or False):
        actionable = False
        entry_price = None
        chase_line = None

    ride_entry_context_px = float(entry_price) if isinstance(entry_price, (float, int)) else (float(pullback_entry) if isinstance(pullback_entry, (float, int)) else float(break_trigger))
    ride_zone_ctx = _evaluate_entry_zone_context(
        df, entry_price=ride_entry_context_px, direction=str(direction), atr_last=float(atr_last) if atr_last is not None else None, lookback=10
    )
    ride_zone_adj = 0.0
    fav_q = float(ride_zone_ctx.get("favorable_quality") or 0.0)
    host_q = float(ride_zone_ctx.get("hostile_quality") or 0.0)
    structure_phase = str(final_geometry.get("structure_phase") or "UNSET")
    if bool(ride_zone_ctx.get("favorable")):
        ride_zone_adj += 4.0 + 3.0 * fav_q
        if structure_phase in ("EXTEND_THEN_PULLBACK", "FAILED_EXTENSION") and str(entry_mode or '').upper() == 'PULLBACK':
            ride_zone_adj += 1.0
    if bool(ride_zone_ctx.get("hostile")):
        hostile_pen = 6.0 + 4.0 * host_q
        if str(entry_mode or '').upper() == 'BREAKOUT':
            hostile_pen += (2.5 + 2.0 * host_q)
        if structure_phase in ("BREAK_AND_HOLD", "ACCEPT_AND_GO") and str(entry_mode or '').upper() == 'BREAKOUT':
            hostile_pen += 1.0
        ride_zone_adj -= hostile_pen
        if str(entry_mode or '').upper() == 'BREAKOUT' and host_q >= 0.70:
            actionable = False
            entry_price = None
            chase_line = None
    ride_cont_strength_adj = _ride_continuation_strength_adjustment(
        structure_phase,
        entry_mode,
        final_geometry.get("tape_breakout_urgency"),
        final_geometry.get("tape_pullback_unlikelihood"),
    )
    ride_cont_phase3_adj, ride_cont_phase3_note = _ride_phase3_continuation_adjustment(
        direction=str(direction),
        adx_ctx=adx_ctx,
        structure_phase=structure_phase,
        entry_mode=entry_mode,
        legitimacy=float(impulse_legitimacy),
        vwap_score=float(vwap_score_up if direction == "LONG" else vwap_score_dn),
        pivot_score=float(pivot_score_up if direction == "LONG" else pivot_score_dn),
        orb_score=float(orb_score_up if direction == "LONG" else orb_score_dn),
    )
    # Final pressure adjustment must be based on the settled RIDE phase/entry mode,
    # not only the provisional geometry. This keeps ADX/DI + MACD + volume pressure
    # aligned with the same phase label used by the final entry router.
    final_pressure_adj, final_pressure_note = _ride_indicator_pressure_adjustment(
        direction=str(direction),
        detail_phase=str(final_geometry.get("structure_phase_detail") or final_geometry.get("structure_phase") or "UNSET"),
        entry_mode=entry_mode,
        adx_ctx=adx_ctx,
        macd_info=ride_macd_info,
        pressure_states=ride_pressure_states,
    )
    score = _cap_score(
        pts
        + ride_zone_adj
        + ride_cont_strength_adj
        + adx_modifier
        + ride_cont_phase3_adj
        + final_pressure_adj
        + float(ride_macd_phase.get('score_adj') or 0.0)
    )

    if bool(ride_macd_phase.get('soft_caution') or False) and str(entry_mode or '').upper() == 'BREAKOUT' and str(structure_phase or '').upper() in ('BREAK_AND_HOLD', 'ACCEPT_AND_GO', 'EXTEND_THEN_PULLBACK'):
        if actionable and float(final_geometry.get('dist_br') or 0.0) > 0.0:
            # keep valid setups valid, but do not chase late momentum fades
            if float(final_geometry.get('dist_br') or 0.0) > 0.65 * max(1e-9, float(atr_last)):
                actionable = False
                entry_price = None
                chase_line = None

    # Trader-edge late continuation guard: block breakout/momentum after the edge has moved.
    try:
        _mode_u = str(entry_mode or '').upper()
        _late_ratio = float(final_geometry.get('breakout_proximity_ratio') or 99.0)
        _ext_pen = float(final_geometry.get('extension_penalty') or 0.0)
        _prebreak_ok = bool(final_geometry.get('prebreak_compression_ready') or False)
        _defense_ok = bool(final_geometry.get('defense_entry_ready') or False)
        if actionable and _mode_u in ('BREAKOUT', 'MOMENTUM') and not _prebreak_ok and not _defense_ok:
            if _late_ratio > (1.16 if stage == 'CONFIRMED' else 1.08) or _ext_pen >= 0.92:
                actionable = False
                entry_price = None
                chase_line = None
                reason = (reason + '; ' if reason else '') + 'Late RIDE continuation blocked'
    except Exception:
        pass

    # Accept-line realism guard: if the raw structural accept line was far enough away to require
    # heavy clamping, treat the displayed continuation geometry more skeptically.
    accept_realism_penalty = 0
    if accept_line_synthetic:
        if accept_clamp_delta_atr >= 0.85:
            accept_realism_penalty = 8 if regime in ("mature_trend", "exhausting") or float(impulse_legitimacy) < 0.60 else 5
        elif accept_clamp_delta_atr >= 0.45:
            accept_realism_penalty = 4 if float(impulse_legitimacy) < 0.70 else 2
        score = _cap_score(float(score) - float(accept_realism_penalty))
        if stage == "CONFIRMED" and accept_clamp_delta_atr >= 0.60:
            stage = "PRE"
        if str(entry_mode or '').upper() == 'PULLBACK' and accept_clamp_delta_atr >= 0.95 and float(impulse_legitimacy) < 0.65:
            actionable = False
            entry_price = None
            chase_line = None

    # Session discipline: continuation off-window should need more proof, and midday/extended
    # sessions should be less eager to surface as CONFIRMED unless quality is exceptional.
    exceptional_session_quality = bool(float(impulse_quality) >= 0.72 and float(score) >= 86 and float(impulse_legitimacy) >= 0.68 and bool(vol_ok))
    if raw_session_risk > 0:
        score = _cap_score(float(score) - float(raw_session_risk))
        if stage == "CONFIRMED" and not exceptional_session_quality:
            stage = "PRE"
        if str(raw_session) in ("PREMARKET", "AFTERHOURS") and not exceptional_session_quality and float(impulse_quality) < 0.62:
            actionable = False
            entry_price = None
            chase_line = None
        if str(raw_session) == "MIDDAY" and not exceptional_session_quality and str(entry_mode or '').upper() == 'BREAKOUT' and float(impulse_quality) < 0.58:
            actionable = False
            entry_price = None
            chase_line = None

    # --- Targets: structure-first + monotonicity ---
    # TP0 should be a *real* liquidity/structure level (not a tiny tick), and TP ordering
    # must be monotonic (TP0 -> TP1 -> TP2 in the trade direction).
    hold_rng = float(df["high"].tail(6).max() - df["low"].tail(6).min())
    min_step = max(0.60 * float(atr_last), 0.35 * float(hold_rng))

    if direction == "LONG":
        cands = [x for x in [levels.get("prior_high"), levels.get("premarket_high"), swing_hi] if isinstance(x, (float, int))]
        cands = [float(x) for x in cands if float(x) > break_trigger + 0.10 * atr_last]
        tp0 = float(min(cands)) if cands else float(break_trigger + 0.90 * atr_last)
        # ensure tp0 isn't a meaningless "tick" target
        if float(tp0) - float(last_price) < 0.25 * float(atr_last):
            tp0 = float(last_price + 0.80 * atr_last)

        tp1 = float(tp0 + max(min_step, 0.70 * hold_rng))
        tp2 = float(tp1 + max(1.00 * atr_last, 0.90 * hold_rng))
    else:
        cands = [x for x in [levels.get("prior_low"), levels.get("premarket_low"), swing_lo] if isinstance(x, (float, int))]
        cands = [float(x) for x in cands if float(x) < break_trigger - 0.10 * atr_last]
        tp0 = float(max(cands)) if cands else float(break_trigger - 0.90 * atr_last)
        if float(last_price) - float(tp0) < 0.25 * float(atr_last):
            tp0 = float(last_price - 0.80 * atr_last)

        tp1 = float(tp0 - max(min_step, 0.70 * hold_rng))
        tp2 = float(tp1 - max(1.00 * atr_last, 0.90 * hold_rng))

    # Optional runner target (TP3): simple, monotonic extension.
    if direction == "LONG":
        tp3 = float(tp2 + max(1.25 * atr_last, 1.10 * hold_rng))
    else:
        tp3 = float(tp2 - max(1.25 * atr_last, 1.10 * hold_rng))

    # Trader-edge TP0 room gate based on actual executable entry.
    try:
        if actionable and entry_price is not None and stop is not None and tp0 is not None:
            if direction == "LONG":
                _reward0 = float(tp0) - float(entry_price); _risk0 = float(entry_price) - float(stop)
            else:
                _reward0 = float(entry_price) - float(tp0); _risk0 = float(stop) - float(entry_price)
            _rr0 = float(_reward0 / max(1e-9, _risk0)); _reward_pct0 = float(_reward0 / max(1e-9, float(entry_price)))
            _mode_u = str(entry_mode or '').upper()
            _early_mode = bool(_mode_u == 'PULLBACK' or final_geometry.get('prebreak_compression_ready') or final_geometry.get('defense_entry_ready'))
            _min_rr = 0.64 if _early_mode else (0.86 if _mode_u == 'BREAKOUT' else 0.96)
            _min_reward_pct = 0.0048 if _early_mode else (0.0060 if _mode_u == 'BREAKOUT' else 0.0070)
            extras_room_gate = {'tp0_rr': float(round(_rr0, 4)), 'tp0_reward_pct': float(round(_reward_pct0, 5)), 'tp0_room_min_rr': float(_min_rr), 'tp0_room_min_reward_pct': float(_min_reward_pct)}
            if _reward0 <= 0 or _rr0 < _min_rr or _reward_pct0 < _min_reward_pct:
                actionable = False
                entry_price = None
                chase_line = None
                reason = (reason + '; ' if reason else '') + f'RIDE TP0 room too thin (RR={_rr0:.2f}, reward={_reward_pct0:.2%})'
        else:
            extras_room_gate = {}
    except Exception:
        extras_room_gate = {}

    # ETA to TP0 (minutes)
    liq_factor = 1.0
    if str(liquidity_phase).upper() in ("AFTERHOURS", "PREMARKET"):
        liq_factor = 1.6
    elif str(liquidity_phase).upper() in ("MIDDAY",):
        liq_factor = 1.25
    elif str(liquidity_phase).upper() in ("OPENING", "POWER"):
        liq_factor = 0.9
    eta_min = None
    try:
        dist = abs(float(tp0) - float(last_price))
        bars = dist / max(1e-6, float(atr_last))
        eta_min = float(bars * float(interval_mins) * liq_factor)
    except Exception:
        eta_min = None

    why = []
    why.append(f"Trend {trend_votes}/3 (ADX {adx_last:.1f})")
    if adx_modifier_note:
        why.append(adx_modifier_note)
    why.append(f"Impulse: {impulse_type_hint} L={impulse_legitimacy:.2f}")
    try:
        if isinstance(ride_zone_ctx, dict):
            if ride_zone_ctx.get("favorable") and ride_zone_ctx.get("favorable_type"):
                why.append(f"Entry near {ride_zone_ctx.get('favorable_type')}")
            if ride_zone_ctx.get("hostile") and ride_zone_ctx.get("hostile_type"):
                why.append(f"Entry near {ride_zone_ctx.get('hostile_type')}")
                if str(entry_mode or '').upper() == 'BREAKOUT':
                    why.append("Hostile zone penalizes breakout")
    except Exception:
        pass
    why.append(f"Accept: {accept_src}" + (" + retest" if stage == "CONFIRMED" else ""))
    if accept_line_synthetic:
        why.append(f"Accept clamp realism penalty (-{int(accept_realism_penalty)})")
    if vol_ok:
        why.append("Vol: expand→compress")
    elif float(hold_structure_quality) >= 0.50:
        why.append("Vol: participation fair")
    if tape_mode_enabled and float(final_geometry.get("tape_readiness") or 0.0) >= 4.0:
        why.append(f"Tape R={float(final_geometry.get('tape_readiness') or 0.0):.1f}")
    if float(ride_cont_strength_adj) <= -4.0:
        why.append("Weak continuation penalty")
    elif float(ride_cont_strength_adj) >= 4.0:
        why.append("Strong continuation bonus")
    if 'ride_cont_phase3_note' in locals() and ride_cont_phase3_note and abs(float(ride_cont_phase3_adj)) >= 1.0:
        why.append(str(ride_cont_phase3_note))
    if 'final_pressure_note' in locals() and final_pressure_note and abs(float(final_pressure_adj or 0.0)) >= 1.0:
        why.append(str(final_pressure_note))
    if exhausted:
        why.append("Exhaustion guard")
    if raw_session_risk > 0:
        why.append(f"Session caution {raw_session} (-{int(raw_session_risk)})")
        if exceptional_session_quality:
            why.append("Off-window quality override")
    if str(entry_mode or '').upper() == 'NO_TRADE':
        why.append("Phase blocks continuation entry")
    elif not actionable:
        why.append("Not near entry lines yet")
    macd_state_print = str((ride_macd_info or {}).get('aligned_state') or 'NEUTRAL_NOISE')
    macd_comment_print = str((ride_macd_phase or {}).get('comment') or '')
    why.append(f"Momentum: {macd_state_print}")
    if macd_comment_print:
        why.append(macd_comment_print)
    if bool((ride_macd_phase or {}).get('watch_early_build') or False) and str(structure_phase).upper() == 'UNSET':
        why.append('Early build only; structure not confirmed yet')
    # compact quality hint
    why.append(f"Q={impulse_quality:.2f}")

    bias = "RIDE_LONG" if direction == "LONG" else "RIDE_SHORT"

    return SignalResult(
        symbol=symbol,
        bias=bias if actionable else "CHOP",
        setup_score=score,
        reason="; ".join(why),
        entry=float(entry_price) if (actionable and entry_price is not None) else None,
        stop=stop if actionable else None,
        target_1r=tp0 if actionable else None,
        target_2r=tp1 if actionable else None,
        last_price=last_price,
        timestamp=last_ts,
        session=session,
        extras={
            "mode": "RIDE",
            "stage": stage if actionable else None,
            "actionable": actionable,
            "macd_hard_block": bool((ride_macd_phase or {}).get("hard_block") or False),
            **(extras_room_gate if isinstance(extras_room_gate, dict) else {}),
            "macd_soft_caution": bool((ride_macd_phase or {}).get("soft_caution") or False),
            "accept_line": float(accept_line),
            "accept_line_raw": float(accept_line_raw),
            "accept_clamp_delta_atr": float(accept_clamp_delta_atr),
            "accept_line_synthetic": bool(accept_line_synthetic),
            "accept_realism_penalty": int(accept_realism_penalty),
            "accept_src": accept_src,
            "accept_recent_diag": accept_recent_diag,
            "break_trigger": float(break_trigger),
            "pullback_entry": float(pullback_entry),
            "pb1": float(pb1),
            "pb2": float(pb2),
            "pb_inner_mult": float(pb_inner_mult),
            "pb_outer_mult": float(pb_outer_mult),
            "dynamic_shallow_bias": float(final_geometry.get("dynamic_shallow_bias") or 0.0),
            "dynamic_deep_bias": float(final_geometry.get("dynamic_deep_bias") or 0.0),
            "ride_trend_strength_local": float(final_geometry.get("trend_strength_local") or 0.0),
            "acceleration_ready": bool(final_geometry.get("acceleration_ready") or False),
            "acceleration_boost": float(final_geometry.get("acceleration_boost") or 0.0),
            "macd_breakout_bonus": float(final_geometry.get("macd_breakout_bonus") or 0.0),
            "pb_high_conviction": bool(high_conviction),
            "tp0": float(tp0),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "tp3": float(tp3),
            "entry_mode": entry_mode,
            "breakout_acceptance_ok": bool(breakout_acceptance_ok),
            "tape_mode_enabled": bool(tape_mode_enabled),
            "tape_readiness": float(final_geometry.get("tape_readiness") or 0.0),
            "tape_tightening": float(final_geometry.get("tape_tightening") or 0.0),
            "tape_hold": float(final_geometry.get("tape_hold") or 0.0),
            "tape_pressure": float(final_geometry.get("tape_pressure") or 0.0),
            "tape_release_proximity": float(final_geometry.get("tape_release_proximity") or 0.0),
            "tape_score_bonus": int(final_geometry.get("tape_score_bonus") or 0),
            "tape_breakout_bias_bonus": int(final_geometry.get("tape_breakout_bias_bonus") or 0),
            "tape_prefers_breakout": bool(final_geometry.get("tape_prefers_breakout") or False),
            "entry_zone_context": ride_zone_ctx,
            "entry_zone_score_adj": float(ride_zone_adj),
            "continuation_strength_score_adj": float(ride_cont_strength_adj),
            "adx_score_adj": float(adx_modifier),
            "chase_line": float(chase_line) if chase_line is not None else None,
            "eta_tp0_min": eta_min,
            "liquidity_phase": liquidity_phase,
            "trend_votes": trend_votes,
            "long_trend_votes": int(long_trend_votes),
            "short_trend_votes": int(short_trend_votes),
            "long_trend_ok": bool(long_trend_ok),
            "short_trend_ok": bool(short_trend_ok),
            "long_trend_context": float(long_trend_context),
            "short_trend_context": float(short_trend_context),
            "adx": adx_last,
            "di_plus": di_p,
            "di_minus": di_m,
            "impulse_quality": impulse_quality,
            "phase_preview_detail": str(phase_preview_detail),
            "phase_preview_confidence": float(phase_preview_conf),
            "phase_disp_floor": float(phase_disp_floor),
            "macd_momentum_state": str((ride_macd_info or {}).get("aligned_state") or "NEUTRAL_NOISE"),
            "macd_momentum_raw_state": str((ride_macd_info or {}).get("raw_state") or "NEUTRAL_NOISE"),
            "macd_momentum_comment": str((ride_macd_phase or {}).get("comment") or ""),
            "macd_momentum_score_adj": float((ride_macd_phase or {}).get("score_adj") or 0.0),
            "macd_watch_early_build": bool((ride_macd_phase or {}).get("watch_early_build") or False),
            "ride_pressure_states": ride_pressure_states,
            "ride_pressure_score_adj": float(final_pressure_adj if 'final_pressure_adj' in locals() else provisional_pressure_adj if 'provisional_pressure_adj' in locals() else 0.0),
            "disp_ratio": disp_ratio,
            "close_stack_q": float(close_stack_q),
            "vol_q": float(vol_q),
            "persistence_q": float(persistence_q),
            "retest_q": float(retest_q),
            "legitimacy_q": float(legitimacy_q),
            "impulse_legitimacy": float(impulse_legitimacy),
            "orb_score": float(impulse_scores_long.get("ORB") if direction == "LONG" else impulse_scores_short.get("ORB")),
            "pivot_score": float(impulse_scores_long.get("PIVOT") if direction == "LONG" else impulse_scores_short.get("PIVOT")),
            "vwap_score": float(impulse_scores_long.get("VWAP") if direction == "LONG" else impulse_scores_short.get("VWAP")),
            "swept_low_then_reclaimed": bool(swept_low_then_reclaim),
            "swept_high_then_rejected": bool(swept_high_then_reject),
            "compression_ok": bool(compression_ok),
            "impulse_type": impulse_type,
            "structure_phase": structure_phase,
            "structure_phase_detail": str(final_geometry.get("structure_phase_detail") or structure_phase),
            "structure_phase_confidence": float(final_geometry.get("structure_phase_confidence") or 0.0),
            "structure_phase_interpretation": str(final_geometry.get("structure_phase_interpretation") or ""),
            "extension_profile": final_geometry.get("extension_profile"),
            "impulse_level": float(impulse_level),
            "impulse_idx": int(impulse_idx) if impulse_idx is not None else None,
            "break_anchor_fallback": bool(impulse_idx is None),
            "slippage_mode": slippage_mode,
            "entry_slip_amount": float(entry_pad),
            "entry_model": entry_model,
            "htf_bias_value": htf_label,
            "htf_bias_effect": float(htf_effect),
            "vwap_logic": vwap_logic,
            "session_vwap_include_premarket": session_vwap_include_premarket,
            "raw_session": str(raw_session),
            "session_confirm_bump": int(session_confirm_bump),
            "session_penalty": int(raw_session_risk),
            "exceptional_session_quality": bool(exceptional_session_quality),
            **_vwap_basis_metadata(
                engine="RIDE",
                vwap_logic=vwap_logic,
                session_vwap_include_premarket=bool(session_vwap_include_premarket),
                session_vwap_include_afterhours=bool(allow_afterhours),
            ),
        },
    )

# =========================
# MSS / ICT (Strict) alerts
# =========================

def _last_pivot_level(df: pd.DataFrame, piv_bool: pd.Series, col: str, *, before_idx: int) -> Tuple[Optional[float], Optional[int]]:
    """Return the most recent pivot level and its index position strictly before `before_idx`."""
    try:
        idxs = np.where(piv_bool.values)[0]
        idxs = idxs[idxs < before_idx]
        if len(idxs) == 0:
            return None, None
        i = int(idxs[-1])
        return float(df[col].iloc[i]), i
    except Exception:
        return None, None


def _first_touch_after(df: pd.DataFrame, *, start_i: int, zone_low: float, zone_high: float) -> Optional[int]:
    """First index >= start_i where candle overlaps the zone."""
    try:
        h = df["high"].values
        l = df["low"].values
        for i in range(max(0, start_i), len(df)):
            if (l[i] <= zone_high) and (h[i] >= zone_low):
                return i
        return None
    except Exception:
        return None


def compute_mss_signal(
    symbol: str,
    df: pd.DataFrame,
    rsi5: Optional[pd.Series] = None,
    rsi14: Optional[pd.Series] = None,
    macd_hist: Optional[pd.Series] = None,
    *,
    interval: str = "1min",
    # Time / bar guards
    allow_opening: bool = True,
    allow_midday: bool = True,
    allow_power: bool = True,
    allow_premarket: bool = False,
    allow_afterhours: bool = False,
    use_last_closed_only: bool = False,
    bar_closed_guard: bool = True,
    # VWAP config (for context + some POI ranking)
    vwap_logic: str = "session",
    session_vwap_include_premarket: bool = False,
    # Fib/vol knobs
    fib_lookback_bars: int = 240,
    orb_minutes: int = 15,
    liquidity_weighting: float = 0.55,
    target_atr_pct: float | None = None,
) -> SignalResult:
    """Strict MSS/ICT alert family.

    Philosophy:
      - Very selective: only fire when we can explicitly see
        raid -> displacement -> MSS break -> POI retest/accept.
      - Output is actionability-oriented (pullback band + trigger + monotonic targets).

    Returns SignalResult with bias in {MSS_LONG, MSS_SHORT, CHOP}.
    """

    if df is None or len(df) < 80:
        return SignalResult(symbol, "CHOP", 0, "Not enough data", None, None, None, None, None, None, None, {"family": "MSS"})

    # Use last closed bar if requested (prevents half-formed candle artifacts)
    dfx = df.copy()
    if use_last_closed_only and len(dfx) >= 2:
        dfx = dfx.iloc[:-1].copy()

    last_ts = dfx.index[-1]
    session = classify_session(last_ts)
    liquidity_phase = classify_liquidity_phase(last_ts)

    # Respect user time-of-day filters (same pattern as other engines).
    allow = {
        "OPENING": allow_opening,
        "MIDDAY": allow_midday,
        "POWER": allow_power,
        "PREMARKET": allow_premarket,
        "AFTERHOURS": allow_afterhours,
        "CLOSED": allow_afterhours,
    }.get(session, True)
    if not allow:
        return SignalResult(symbol, "CHOP", 0, f"Time filter blocks MSS ({session})", None, None, None, None, None, None, session, {"family": "MSS", "session": session})

    # --- Core series ---
    atr14 = calc_atr(dfx[["high", "low", "close"]], period=14)
    atr_last = float(atr14.iloc[-1]) if len(atr14) else 0.0

    # Vol normalization baseline (optional)
    atr_pct = float(atr_last / float(dfx["close"].iloc[-1])) if float(dfx["close"].iloc[-1]) else 0.0
    atr_score_scale = 1.0
    baseline_atr_pct = None
    if target_atr_pct is not None and isinstance(target_atr_pct, (float, int)) and target_atr_pct > 0:
        baseline_atr_pct = float(target_atr_pct)
        try:
            atr_score_scale = float(np.clip(baseline_atr_pct / max(atr_pct, 1e-9), 0.75, 1.25))
        except Exception:
            atr_score_scale = 1.0

    # --- Pivots: external (structure) + internal (MSS) ---
    ext_l = 6 if interval in ("1min", "5min") else 8
    ext_r = ext_l
    int_l = 2
    int_r = 2

    piv_low_ext = rolling_swing_lows(dfx["low"], left=ext_l, right=ext_r)
    piv_high_ext = rolling_swing_highs(dfx["high"], left=ext_l, right=ext_r)
    piv_low_int = rolling_swing_lows(dfx["low"], left=int_l, right=int_r)
    piv_high_int = rolling_swing_highs(dfx["high"], left=int_l, right=int_r)

    # --- Find most recent raid (liquidity sweep) ---
    raid_search = min(180, len(dfx) - 10)
    raid_i = None
    raid_side = None  # "bull" means swept lows
    raid_level = None

    # scan from near-end backwards for a clean sweep
    lows = dfx["low"].values
    highs = dfx["high"].values
    closes = dfx["close"].values

    for i in range(len(dfx) - 2, max(10, len(dfx) - raid_search), -1):
        # bullish raid: take external pivot low, wick below, close back above pivot (reclaim)
        pl, pl_i = _last_pivot_level(dfx, piv_low_ext, "low", before_idx=i)
        if pl is not None and pl_i is not None:
            if lows[i] < pl and closes[i] > pl:
                # require meaningful sweep size
                if atr_last > 0 and (pl - lows[i]) >= 0.15 * atr_last:
                    raid_i = i
                    raid_side = "bull"
                    raid_level = float(pl)
                    break
        # bearish raid: take external pivot high, wick above, close back below pivot
        ph, ph_i = _last_pivot_level(dfx, piv_high_ext, "high", before_idx=i)
        if ph is not None and ph_i is not None:
            if highs[i] > ph and closes[i] < ph:
                if atr_last > 0 and (highs[i] - ph) >= 0.15 * atr_last:
                    raid_i = i
                    raid_side = "bear"
                    raid_level = float(ph)
                    break

    if raid_i is None or raid_side is None:
        return SignalResult(symbol, "CHOP", 0, "No clean liquidity raid found", None, None, None, None, None, None, session, {"family": "MSS", "stage": "OFF", "liquidity_phase": liquidity_phase})

    # --- Displacement after raid ---
    tr = (dfx["high"] - dfx["low"]).rolling(20).median().fillna(method="bfill")
    disp_i = None
    disp_ratio = None

    for j in range(raid_i + 1, min(len(dfx), raid_i + 15)):
        rng = float(dfx["high"].iloc[j] - dfx["low"].iloc[j])
        med = float(tr.iloc[j]) if float(tr.iloc[j]) else 0.0
        if med <= 0:
            continue
        body = float(abs(dfx["close"].iloc[j] - dfx["open"].iloc[j]))
        dr = rng / med
        # directionality
        bull_dir = dfx["close"].iloc[j] > dfx["open"].iloc[j]
        bear_dir = dfx["close"].iloc[j] < dfx["open"].iloc[j]
        if raid_side == "bull" and bull_dir and dr >= 1.35 and (body / max(rng, 1e-9)) >= 0.55:
            disp_i = j
            disp_ratio = dr
            break
        if raid_side == "bear" and bear_dir and dr >= 1.35 and (body / max(rng, 1e-9)) >= 0.55:
            disp_i = j
            disp_ratio = dr
            break

    if disp_i is None:
        return SignalResult(symbol, "CHOP", 0, "Raid found but no displacement", None, None, None, None, None, None, session, {"family": "MSS", "stage": "OFF", "liquidity_phase": liquidity_phase, "raid_i": raid_i})

    # --- MSS break: break of internal pivot in displacement direction ---
    if raid_side == "bull":
        # internal pivot high between raid and displacement
        mss_level, mss_piv_i = _last_pivot_level(dfx, piv_high_int, "high", before_idx=disp_i)
        if mss_level is None:
            return SignalResult(symbol, "CHOP", 0, "No internal pivot for MSS", None, None, None, None, None, None, session, {"family": "MSS", "stage": "OFF"})
        mss_break_i = None
        for k in range(disp_i, min(len(dfx), disp_i + 20)):
            if float(dfx["close"].iloc[k]) > float(mss_level):
                mss_break_i = k
                break
        if mss_break_i is None:
            return SignalResult(symbol, "CHOP", 0, "No MSS break yet", None, None, None, None, None, None, session, {"family": "MSS", "stage": "OFF", "mss_level": float(mss_level)})
        bias = "MSS_LONG"
    else:
        mss_level, mss_piv_i = _last_pivot_level(dfx, piv_low_int, "low", before_idx=disp_i)
        if mss_level is None:
            return SignalResult(symbol, "CHOP", 0, "No internal pivot for MSS", None, None, None, None, None, None, session, {"family": "MSS", "stage": "OFF"})
        mss_break_i = None
        for k in range(disp_i, min(len(dfx), disp_i + 20)):
            if float(dfx["close"].iloc[k]) < float(mss_level):
                mss_break_i = k
                break
        if mss_break_i is None:
            return SignalResult(symbol, "CHOP", 0, "No MSS break yet", None, None, None, None, None, None, session, {"family": "MSS", "stage": "OFF", "mss_level": float(mss_level)})
        bias = "MSS_SHORT"

    # --- POI selection (order block / FVG / breaker) from raid->break window ---
    window_df = dfx.iloc[max(0, raid_i - 5): mss_break_i + 1].copy()

    poi_low = None
    poi_high = None
    poi_src = None

    # Order block
    try:
        ob = find_order_block(window_df, atr14.loc[window_df.index], side=("bull" if raid_side == "bull" else "bear"))
        if ob and isinstance(ob, dict):
            poi_low = float(ob.get("low"))
            poi_high = float(ob.get("high"))
            poi_src = "OB"
    except Exception:
        pass

    # FVG (prefer if present and tighter)
    try:
        fvg = detect_fvg(window_df)
        if fvg and isinstance(fvg, dict):
            fl = float(fvg.get("low"))
            fh = float(fvg.get("high"))
            if (poi_low is None) or (fh - fl) < (poi_high - poi_low):
                poi_low, poi_high, poi_src = fl, fh, "FVG"
    except Exception:
        pass

    # Breaker fallback
    if poi_low is None or poi_high is None:
        try:
            br = find_breaker_block(window_df, atr14.loc[window_df.index], side=("bull" if raid_side == "bull" else "bear"))
            if br and isinstance(br, dict):
                poi_low = float(br.get("low"))
                poi_high = float(br.get("high"))
                poi_src = "BREAKER"
        except Exception:
            pass

    if poi_low is None or poi_high is None:
        # last resort: midpoint of displacement candle
        poi_low = float(min(window_df["open"].iloc[-1], window_df["close"].iloc[-1]))
        poi_high = float(max(window_df["open"].iloc[-1], window_df["close"].iloc[-1]))
        poi_src = "DISP_BODY"

    poi_low, poi_high = float(min(poi_low, poi_high)), float(max(poi_low, poi_high))
    poi_mid = 0.5 * (poi_low + poi_high)

    # --- Retest + accept (CONFIRMED) ---
    touch_i = _first_touch_after(dfx, start_i=mss_break_i, zone_low=poi_low, zone_high=poi_high)
    retest_ok = touch_i is not None
    accept_ok = False
    if retest_ok:
        after = min(len(dfx) - 1, int(touch_i) + 3)
        if bias == "MSS_LONG":
            accept_ok = float(dfx["close"].iloc[after]) >= poi_mid
        else:
            accept_ok = float(dfx["close"].iloc[after]) <= poi_mid

    # --- Actionability: ATR-distance to POI band or trigger ---
    last_price = float(dfx["close"].iloc[-1])
    atr = max(atr_last, 1e-9)

    # trigger is the MSS break level (for long, above; for short, below)
    break_trigger = float(mss_level)
    dist_to_poi = 0.0
    if last_price < poi_low:
        dist_to_poi = (poi_low - last_price)
    elif last_price > poi_high:
        dist_to_poi = (last_price - poi_high)

    dist_to_trigger = abs(last_price - break_trigger)
    actionable_gate = (min(dist_to_poi, dist_to_trigger) <= 0.75 * atr)

    stage = None
    if actionable_gate:
        stage = "PRE"
        if retest_ok and accept_ok:
            stage = "CONFIRMED"

    # --- Entries / stops (strict + practical) ---
    pullback_entry = float(poi_mid)
    pb1 = float(poi_high)
    pb2 = float(poi_low)

    raid_extreme = float(dfx["low"].iloc[raid_i]) if bias == "MSS_LONG" else float(dfx["high"].iloc[raid_i])
    strict_stop = raid_extreme - 0.05 * atr if bias == "MSS_LONG" else raid_extreme + 0.05 * atr
    practical_stop = (poi_low - 0.10 * atr) if bias == "MSS_LONG" else (poi_high + 0.10 * atr)
    stop = float(practical_stop)

    # --- Targets (monotonic, structure-first) ---
    tp0 = None
    tp1 = None
    tp2 = None

    if bias == "MSS_LONG":
        # nearest internal pivot high above last
        candidates = []
        for i in np.where(piv_high_int.values)[0]:
            if i < len(dfx) and float(dfx["high"].iloc[i]) > last_price:
                candidates.append(float(dfx["high"].iloc[i]))
        candidates = sorted(set(candidates))
        tp0 = candidates[0] if candidates else float(last_price + 1.0 * atr)

        # next external pivot high (bigger pool)
        ext_cand = []
        for i in np.where(piv_high_ext.values)[0]:
            if i < len(dfx) and float(dfx["high"].iloc[i]) > float(tp0):
                ext_cand.append(float(dfx["high"].iloc[i]))
        ext_cand = sorted(set(ext_cand))
        tp1 = ext_cand[0] if ext_cand else float(tp0 + 1.0 * atr)

        # measured move from displacement
        disp_range = float(dfx["high"].iloc[disp_i] - dfx["low"].iloc[disp_i])
        tp2 = float(max(tp1, pullback_entry + max(disp_range, 1.2 * atr)))

    else:
        candidates = []
        for i in np.where(piv_low_int.values)[0]:
            if i < len(dfx) and float(dfx["low"].iloc[i]) < last_price:
                candidates.append(float(dfx["low"].iloc[i]))
        candidates = sorted(set(candidates), reverse=True)
        tp0 = candidates[0] if candidates else float(last_price - 1.0 * atr)

        ext_cand = []
        for i in np.where(piv_low_ext.values)[0]:
            if i < len(dfx) and float(dfx["low"].iloc[i]) < float(tp0):
                ext_cand.append(float(dfx["low"].iloc[i]))
        ext_cand = sorted(set(ext_cand), reverse=True)
        tp1 = ext_cand[0] if ext_cand else float(tp0 - 1.0 * atr)

        disp_range = float(dfx["high"].iloc[disp_i] - dfx["low"].iloc[disp_i])
        tp2 = float(min(tp1, pullback_entry - max(disp_range, 1.2 * atr)))

    # ensure monotonic ordering
    if bias == "MSS_LONG":
        tp0 = float(max(tp0, last_price))
        tp1 = float(max(tp1, tp0))
        tp2 = float(max(tp2, tp1))
    else:
        tp0 = float(min(tp0, last_price))
        tp1 = float(min(tp1, tp0))
        tp2 = float(min(tp2, tp1))

    # --- Score (quality-driven) ---
    score = 0.0
    why_bits = []

    # Raid quality (size)
    try:
        if raid_side == "bull":
            raid_size = float(raid_level - lows[raid_i])
        else:
            raid_size = float(highs[raid_i] - raid_level)
        raid_q = float(np.clip(raid_size / max(atr, 1e-9), 0.0, 2.0))
    except Exception:
        raid_q = 0.0

    score += 20.0 * min(1.0, raid_q)
    why_bits.append("Raid+reclaim")

    # Displacement quality
    dq = float(np.clip((disp_ratio or 0.0) / 2.0, 0.0, 1.0))
    score += 25.0 * dq
    why_bits.append("Displacement")

    # MSS break
    score += 20.0
    why_bits.append("MSS break")

    # POI quality
    if poi_src in ("FVG", "OB", "BREAKER"):
        score += 10.0
        why_bits.append(f"POI={poi_src}")

    # Retest/accept
    if retest_ok:
        score += 10.0
        why_bits.append("Retest")
    if accept_ok:
        score += 10.0
        why_bits.append("Accept")

    # RSI exhaustion guard (prevents buying top / selling bottom)
    if rsi5 is not None and rsi14 is not None:
        try:
            r5 = float(rsi5.iloc[-1])
            r14 = float(rsi14.iloc[-1])
            if bias == "MSS_LONG" and (r5 > 88 and r14 > 72):
                score -= 12.0
                why_bits.append("RSI exhausted")
            if bias == "MSS_SHORT" and (r5 < 12 and r14 < 28):
                score -= 12.0
                why_bits.append("RSI exhausted")
            else:
                score += 5.0
        except Exception:
            pass

    score *= float(atr_score_scale)
    score_i = _cap_score(score)

    actionable = stage in ("PRE", "CONFIRMED") and bias in ("MSS_LONG", "MSS_SHORT")

    reason = " ".join(why_bits)
    if stage is None:
        reason = reason + "; Too far from POI/trigger (ATR gating)"

    # ETA TP0 using same concept as other engines
    eta_min = None
    try:
        if atr_last > 0:
            dist = abs(float(tp0) - last_price)
            # rough minutes per ATR based on liquidity phase
            pace = 7.0 if liquidity_phase == "RTH" else 11.0
            eta_min = float(max(1.0, (dist / atr_last) * pace))
    except Exception:
        eta_min = None

    return SignalResult(
        symbol=symbol,
        # Keep the MSS family bias namespace intact so app-side routing/alerting
        # can key off (MSS_LONG/MSS_SHORT) without ambiguity.
        bias=bias if actionable else "CHOP",
        setup_score=score_i,
        reason=(f"MSS {stage or 'OFF'} — {reason}"),
        last_price=last_price,
        entry=pullback_entry if actionable else None,
        stop=stop if actionable else None,
        target_1r=float(tp0) if actionable else None,
        target_2r=float(tp1) if actionable else None,
        timestamp=last_ts,
        session=session,
        extras={
            "family": "MSS",
            "stage": stage,
            "actionable": actionable,
            "poi_src": poi_src,
            "pb1": pb1,
            "pb2": pb2,
            "pullback_entry": pullback_entry,
            "break_trigger": break_trigger,
            "strict_stop": float(strict_stop),
            "tp0": float(tp0),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "eta_tp0_min": eta_min,
            "liquidity_phase": liquidity_phase,
            "raid_i": int(raid_i),
            "disp_i": int(disp_i),
            "mss_level": float(mss_level),
            "disp_ratio": float(disp_ratio) if disp_ratio is not None else None,
            "atr_pct": atr_pct,
            "baseline_atr_pct": baseline_atr_pct,
            "atr_score_scale": atr_score_scale,
            "vwap_logic": vwap_logic,
            "session_vwap_include_premarket": session_vwap_include_premarket,
            "raw_session": str(raw_session),
            "session_confirm_bump": int(session_confirm_bump),
            "session_penalty": int(raw_session_risk),
            "exceptional_session_quality": bool(exceptional_session_quality),
            **_vwap_basis_metadata(
                engine="MSS",
                vwap_logic=vwap_logic,
                session_vwap_include_premarket=bool(session_vwap_include_premarket),
                session_vwap_include_afterhours=bool(allow_afterhours),
                note="MSS currently uses VWAP metadata for diagnostics only; raid/displacement logic is structure-first.",
            ),
        },
    )
