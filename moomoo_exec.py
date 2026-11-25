"""moomoo_exec.py - the ONLY file that can place orders."""
import csv, os, time
import moomoo as ft
from moomoo_client import HOST, PORT

LOG = "orders_log.csv"
LIVE_ENABLED = os.getenv("MOOMOO_LIVE") == "1"   # leave unset during the paper phase


class Executor:
    def __init__(self, security_firm=None):
        kwargs = {"filter_trdmarket": ft.TrdMarket.US, "host": HOST, "port": PORT}
        if security_firm is not None:
            kwargs["security_firm"] = security_firm
        self.trade = ft.OpenSecTradeContext(**kwargs)

    def place(self, code, side, qty, price, env="SIMULATE", reason=""):
        if env != "SIMULATE" and not LIVE_ENABLED:
            raise PermissionError("Live trading is switched off (MOOMOO_LIVE is not 1).")
        trd_env = ft.TrdEnv.SIMULATE if env == "SIMULATE" else ft.TrdEnv.REAL
        if trd_env == ft.TrdEnv.REAL:
            # Password comes from your environment, never from the model or the chat.
            self.trade.unlock_trade(password=os.environ["MOOMOO_TRADE_PWD"])
        ret, data = self.trade.place_order(
            price=price, qty=qty, code=code,
            trd_side=ft.TrdSide.BUY if side == "BUY" else ft.TrdSide.SELL,
            order_type=ft.OrderType.NORMAL, trd_env=trd_env,
        )
        with open(LOG, "a", newline="") as f:
            csv.writer(f).writerow([time.strftime("%F %T"), env, code, side, qty, price,
                                    "OK" if ret == ft.RET_OK else "FAIL", reason, str(data)[:200]])
        if ret != ft.RET_OK:
            raise RuntimeError(f"Order failed: {data}")
        return data