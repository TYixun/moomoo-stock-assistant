"""moomoo_client.py - read-only wrapper around OpenD. No order methods on purpose."""
import moomoo as ft

HOST, PORT = "127.0.0.1", 11111


class MoomooReadOnly:
    def __init__(self, env: str = "SIMULATE", security_firm=None):
        # Default to the paper account. Switch to "REAL" only after Step 8.
        self.env = ft.TrdEnv.SIMULATE if env == "SIMULATE" else ft.TrdEnv.REAL
        self.quote = ft.OpenQuoteContext(host=HOST, port=PORT)
        kwargs = {"filter_trdmarket": ft.TrdMarket.US, "host": HOST, "port": PORT}
        if security_firm is not None:   # look up the value for Moomoo MY in the SDK docs
            kwargs["security_firm"] = security_firm
        self.trade = ft.OpenSecTradeContext(**kwargs)

    @staticmethod
    def _ok(ret, data):
        if ret != ft.RET_OK:
            raise RuntimeError(f"Moomoo API error: {data}")
        return data

    def snapshot(self, codes):
        """Latest quote rows, e.g. codes=['US.QQQ', 'US.NVDA']."""
        return self._ok(*self.quote.get_market_snapshot(codes))

    def accounts(self):
        return self._ok(*self.trade.get_acc_list())

    def account_info(self):
        return self._ok(*self.trade.accinfo_query(trd_env=self.env))

    def positions(self):
        return self._ok(*self.trade.position_list_query(trd_env=self.env))

    def orders(self):
        return self._ok(*self.trade.order_list_query(trd_env=self.env))

    def close(self):
        self.quote.close()
        self.trade.close()


if __name__ == "__main__":
    c = MoomooReadOnly()
    print(c.accounts())
    print(c.account_info())
    print(c.positions())
    print(c.snapshot(["US.QQQ"]))
    c.close()