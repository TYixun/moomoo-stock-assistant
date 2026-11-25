"""guardrails.py - plain-Python checks. The model cannot change these."""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

ALLOWED = {"QQQ", "VOO", "SPY", "AAPL", "MSFT", "NVDA"}   # edit to taste
MAX_ORDER_USD = 30.0
MAX_ORDERS_PER_DAY = 2
MAX_PRICE_DEVIATION = 0.02      # limit price within 2% of the last quote
KILL_FILE = "STOP"              # create a file with this name to halt everything


def us_market_open(now=None):
    # Regular hours only; does not know about US holidays.
    now = now or datetime.now(ZoneInfo("America/New_York"))
    return now.weekday() < 5 and (9, 30) <= (now.hour, now.minute) < (16, 0)


def check(p, last_price, held_qty, orders_today):
    """Return a list of problems. An empty list means the proposal may go to the human."""
    errors = []
    if os.path.exists(KILL_FILE):
        errors.append("Kill switch is on.")
    try:
        qty, px = float(p["quantity"]), float(p["limit_price"])
    except (KeyError, TypeError, ValueError):
        return errors + ["Quantity or price missing or not a number."]
    ticker, side = str(p.get("ticker", "")).upper(), p.get("side")
    if ticker not in ALLOWED:
        errors.append(f"{ticker} is not on the allowed list.")
    if side not in ("BUY", "SELL"):
        errors.append("Side must be BUY or SELL.")
    if qty != int(qty):
        errors.append("Whole shares only.")
    if qty <= 0 or px <= 0:
        errors.append("Quantity and price must be positive.")
    if qty * px > MAX_ORDER_USD:
        errors.append(f"Order value ${qty * px:.2f} is above the ${MAX_ORDER_USD:.0f} cap.")
    if last_price and abs(px / last_price - 1) > MAX_PRICE_DEVIATION:
        errors.append("Limit price is too far from the last quote.")
    if side == "SELL" and qty > held_qty:
        errors.append("Cannot sell more than you hold.")
    if orders_today >= MAX_ORDERS_PER_DAY:
        errors.append("Daily order limit reached.")
    if not us_market_open():
        errors.append("US market is closed.")
    return errors