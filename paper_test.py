from moomoo_client import MoomooReadOnly
from moomoo_exec import Executor

mm = MoomooReadOnly()
code = "US.QQQ"
last = float(mm.snapshot([code])["last_price"].iloc[0])
print("Last price:", last)

ex = Executor()
# 1 share, limit 2% below the last price, in the PAPER account
print(ex.place(code, "BUY", 0.04, round(last * 0.98, 2), env="SIMULATE", reason="first test"))

print(mm.orders())
mm.close()