# main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Tuple
import os
import numpy as np

VERSION = "1.1.1"

app = FastAPI(title="toto-server", version=VERSION)

# ========================== Pydantic схемы (v2) ==========================

class ForecastIn(BaseModel):
    # Вариант 1: один ряд
    series: Optional[List[float]] = None
    # Вариант 2: несколько рядов
    seriesMulti: Optional[List[List[float]]] = None

    horizon: int = Field(3, ge=1, le=4096)
    confidence: float = Field(0.9, ge=0.5, le=0.995)
    samples: int = Field(256, ge=1, le=1024)           # для Toto (если доступен)
    intervalSeconds: int = Field(60, ge=1, le=86400)   # шаг между точками

    @model_validator(mode="after")
    def _validate(self) -> "ForecastIn":
        if self.series is None and self.seriesMulti is None:
            raise ValueError("Provide either `series` or `seriesMulti`.")
        if self.series is not None and self.seriesMulti is not None:
            raise ValueError("Provide only one of `series` or `seriesMulti`.")

        if self.series is not None:
            if len(self.series) < 3:
                raise ValueError("`series` must contain at least 3 points.")
        else:
            # seriesMulti
            assert self.seriesMulti is not None
            if not isinstance(self.seriesMulti, list) or not self.seriesMulti:
                raise ValueError("`seriesMulti` must be non-empty list of lists.")
            L = None
            for i, row in enumerate(self.seriesMulti):
                if not isinstance(row, list) or len(row) < 3:
                    raise ValueError(f"`seriesMulti[{i}]` must be list with >=3 values")
                if L is None:
                    L = len(row)
                elif len(row) != L:
                    raise ValueError("All rows in `seriesMulti` must have equal length.")
        return self


class ForecastOut(BaseModel):
    engine: str
    device: str
    horizon: int
    predictions: List[float]
    lower: List[float]
    upper: List[float]
    lastValue: float
    predictionsMulti: Optional[List[List[float]]] = None
    lowerMulti: Optional[List[List[float]]] = None
    upperMulti: Optional[List[List[float]]] = None


class BacktestIn(BaseModel):
    series: List[float] = Field(..., min_items=8)
    horizon: int = Field(3, ge=1, le=128)
    window: int = Field(24, ge=6, le=4096)
    confidence: float = Field(0.9, ge=0.5, le=0.995)


class BacktestOut(BaseModel):
    mae: float
    mape: float
    coverage: float
    samples: int
    engine: str


# ========================== Глобалы/модель ==========================

ENGINE_NAME = "ema-fallback"
DEVICE = os.getenv("TOTO_DEVICE", "cpu")
TOTO_READY = False
TOTO_LOAD_ERR: Optional[str] = None

# Поздние импорты Toto (если есть)
_torch = None
_Toto = None
_TotoForecaster = None
_MaskedTimeseries = None


def try_load_toto() -> None:
    """Пробуем подключить toto-ts; при неудаче — остаёмся на EMA fallback."""
    global ENGINE_NAME, DEVICE, TOTO_READY, TOTO_LOAD_ERR
    global _torch, _Toto, _TotoForecaster, _MaskedTimeseries

    try:
        import torch  # type: ignore
        from toto.model.toto import Toto  # type: ignore
        from toto.inference.forecaster import TotoForecaster  # type: ignore
        from toto.data.util.dataset import MaskedTimeseries  # type: ignore

        _torch = torch
        _Toto = Toto
        _TotoForecaster = TotoForecaster
        _MaskedTimeseries = MaskedTimeseries

        if DEVICE == "cuda" and not torch.cuda.is_available():
            DEVICE = "cpu"

        ckpt = os.getenv("TOTO_CHECKPOINT", "Datadog/Toto-Open-Base-1.0")

        # ВАЖНО: НЕ передаём memory_efficient_attention — это и падало в логах
        model = Toto.from_pretrained(ckpt)
        model = model.to(DEVICE)
        try:
            model.compile()
        except Exception:
            pass

        app.state.toto_model = model
        app.state.toto_forecaster = TotoForecaster(model.model)
        ENGINE_NAME = "toto-open-1.0"
        TOTO_READY = True
        TOTO_LOAD_ERR = None
    except Exception as e:
        # Фоллбек
        TOTO_LOAD_ERR = f"{type(e).__name__}: {e}"
        app.state.toto_model = None
        app.state.toto_forecaster = None
        ENGINE_NAME = "ema-fallback"
        TOTO_READY = False


@app.on_event("startup")
def _on_startup():
    try_load_toto()


# ========================== Утилиты прогноза ==========================

def _ema_forecast(series: np.ndarray, horizon: int, alpha: float = 0.3) -> List[float]:
    ema = float(series[0])
    for x in series[1:]:
        ema = alpha * float(x) + (1 - alpha) * ema
    return [ema] * horizon


def _bands_from_residuals(series: np.ndarray, confidence: float) -> float:
    kernel = np.ones(3) / 3.0
    smooth = np.convolve(series, kernel, mode="same")
    resid = series - smooth
    sigma = float(np.std(resid if np.isfinite(resid).any() else series) or 1e-6)
    if confidence >= 0.99:
        z = 2.58
    elif confidence >= 0.95:
        z = 1.96
    elif confidence >= 0.9:
        z = 1.64
    else:
        z = 1.28
    return z * sigma


def _ensure_multi(x: ForecastIn) -> List[List[float]]:
    if x.series is not None:
        return [x.series]
    assert x.seriesMulti is not None
    return x.seriesMulti


def _run_toto_forecast(
    series_multi: List[List[float]],
    horizon: int,
    confidence: float,
    samples: int,
    interval_s: int,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """Возвращает (median_multi, lower_multi, upper_multi) через Toto."""
    if not TOTO_READY or app.state.toto_forecaster is None:
        raise RuntimeError("Toto is not available")

    import torch  # локально

    N = len(series_multi)
    x = torch.tensor(series_multi, dtype=torch.float32)
    inputs = _MaskedTimeseries(
        series=x,
        padding_mask=torch.full_like(x, True, dtype=torch.bool),
        id_mask=torch.zeros_like(x),
        timestamp_seconds=torch.zeros_like(x),
        time_interval_seconds=torch.full((N,), int(interval_s), dtype=torch.int32),
    )

    fc = app.state.toto_forecaster.forecast(
        inputs,
        prediction_length=int(horizon),
        num_samples=int(samples),
        samples_per_batch=int(min(samples, 256)),
    )

    median = fc.median.cpu().numpy()
    q_low = (1.0 - confidence) / 2.0
    q_high = 1.0 - q_low
    lower = fc.quantile(q_low).cpu().numpy()
    upper = fc.quantile(q_high).cpu().numpy()

    return (
        [list(map(float, row)) for row in median],
        [list(map(float, row)) for row in lower],
        [list(map(float, row)) for row in upper],
    )


# ========================== Эндпоинты ==========================

@app.get("/")
def index():
    return {
        "name": "toto-server",
        "version": VERSION,
        "engine": ENGINE_NAME,
        "device": DEVICE,
        "ready": TOTO_READY,
        "load_error": TOTO_LOAD_ERR,
        "endpoints": ["/health", "/forecast", "/backtest"],
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "engine": ENGINE_NAME,
        "device": DEVICE,
        "ready": TOTO_READY,
        "load_error": TOTO_LOAD_ERR,
    }


@app.post("/forecast", response_model=ForecastOut)
def forecast(body: ForecastIn):
    series_multi = _ensure_multi(body)
    horizon = int(body.horizon)
    conf = float(body.confidence)
    interval_s = int(body.intervalSeconds)
    samples = int(body.samples)

    try:
        if TOTO_READY:
            predM, lowM, upM = _run_toto_forecast(
                series_multi, horizon, conf, samples, interval_s
            )
        else:
            raise RuntimeError("Toto not ready")
    except Exception:
        # EMA fallback
        predM, lowM, upM = [], [], []
        for row in series_multi:
            s = np.asarray(row, dtype=float)
            preds = _ema_forecast(s, horizon)
            band = _bands_from_residuals(s, conf)
            predM.append(preds)
            lowM.append([p - band for p in preds])
            upM.append([p + band for p in preds])

    first_pred = predM[0]
    first_low = lowM[0]
    first_up = upM[0]
    last_value = float(series_multi[0][-1])

    return ForecastOut(
        engine=ENGINE_NAME,
        device=DEVICE,
        horizon=horizon,
        predictions=[float(x) for x in first_pred],
        lower=[float(x) for x in first_low],
        upper=[float(x) for x in first_up],
        lastValue=last_value,
        predictionsMulti=[[float(x) for x in row] for row in predM],
        lowerMulti=[[float(x) for x in row] for row in lowM],
        upperMulti=[[float(x) for x in row] for row in upM],
    )


@app.post("/backtest", response_model=BacktestOut)
def backtest(body: BacktestIn):
    s = np.asarray(body.series, dtype=float)
    H = int(body.horizon)
    W = int(body.window)
    if len(s) < W + H + 1:
        W = max(6, min(W, len(s) - H - 1))

    if W <= 6:
        return BacktestOut(mae=0.0, mape=0.0, coverage=1.0, samples=0, engine=ENGINE_NAME)

    maes, mapes, covs = [], [], []
    for i in range(len(s) - W - H + 1):
        hist = s[i : i + W]
        preds = np.asarray(_ema_forecast(hist, H))
        band = _bands_from_residuals(hist, body.confidence)
        lower = preds - band
        upper = preds + band
        actual = s[i + W : i + W + H]
        err = np.abs(actual - preds)
        maes.append(float(np.mean(err)))
        mapes.append(float(np.mean(err / (np.abs(actual) + 1e-6)) * 100.0))
        covs.append(float(np.mean((actual >= lower) & (actual <= upper))))

    return BacktestOut(
        mae=float(np.mean(maes)),
        mape=float(np.mean(mapes)),
        coverage=float(np.mean(covs)),
        samples=len(maes),
        engine=ENGINE_NAME,
    )
