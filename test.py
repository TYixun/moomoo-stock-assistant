import moomoo as ft

ctx = ft.OpenQuoteContext(host="127.0.0.1", port=11111)
ret, data = ctx.get_market_snapshot(["US.QQQ"])
print("OK" if ret == ft.RET_OK else "ERROR")
print(data)   # a table on success, the error message otherwise
ctx.close()