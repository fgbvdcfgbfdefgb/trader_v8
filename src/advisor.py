"""
Turns the Market Analyser's outputs + raw indicators into a human-readable
"advisory" string for that day -- this is what gets drawn onto each epoch's
saved PNG so you can see what the system "thought" about that day's market.
"""
import numpy as np

TREND_LABELS = ["bearish / downtrend", "sideways / flat", "bullish / uptrend"]
VOL_LABELS = ["low volatility", "moderate volatility", "high volatility"]


def summarize_day(symbol, trend_probs, vol_probs, rsi_last, macd_last, bb_last, day_return_pct):
    trend = TREND_LABELS[int(np.argmax(trend_probs))]
    vol = VOL_LABELS[int(np.argmax(vol_probs))]
    rsi_note = "overbought" if rsi_last > 70 else ("oversold" if rsi_last < 30 else "neutral")
    macd_note = "bullish momentum" if macd_last > 0 else "bearish momentum"
    bb_note = (
        "near upper band" if bb_last > 0.8 else
        "near lower band" if bb_last < 0.2 else "mid-range"
    )
    return (
        f"{symbol}: {trend}, {vol}. RSI={rsi_last:.1f} ({rsi_note}), "
        f"MACD {macd_note}, Bollinger {bb_note}. Day realized move: {day_return_pct:+.2f}%."
    )


def build_advisor_report(per_symbol_summaries: dict) -> str:
    lines = ["Market Analyser advisory for this trading day:"]
    for sym, s in per_symbol_summaries.items():
        lines.append("  - " + s)
    return "\n".join(lines)
