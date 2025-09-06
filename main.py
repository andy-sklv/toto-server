from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Toto Mini Server", version="0.1.0")

class ForecastReq(BaseModel):
    series: List[float]
    horizon: int = 3
    metric: str = "chars_per_sec"  # или "latency_ms"

class ForecastResp(BaseModel):
    predictions: List[float]
    engine: str = "toto-py-demo"

def ema(series: List[float], alpha: float = 0.35) -> float:
    if not series:
        return 0.0
    s = series[0]
    for i in range(1, len(series)):
        s = alpha * series[i] + (1 - alpha) * s
    return float(s)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/forecast", response_model=ForecastResp)
def forecast(req: ForecastReq):
    s = [float(x) for x in req.series if isinstance(x, (int, float))]
    if len(s) < 2:
        return ForecastResp(predictions=[], engine="toto-py-demo")

    base = ema(s)
    # простая линейная дрейф-оценка
    left = s[: max(1, len(s) // 2)]
    right = s[max(1, len(s) // 2) :]
    drift = (ema(right) - ema(left)) if left and right else 0.0

    preds = []
    for i in range(req.horizon):
        preds.append(round(base + drift * (i + 1), 4))
    return ForecastResp(predictions=preds, engine="toto-py-demo")
