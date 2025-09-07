# toto-server  
_Мини-сервис на FastAPI для прогноза временных рядов (демо под Datadog Toto)_

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi)
![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI-4B8BBE)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Привет! 👋 Это мой компактный сервер-прогнозист: принимает небольшой временной ряд (например, **скорость генерации ответа** из UI) и возвращает **прогноз** на несколько шагов вперёд. Делал его под демонстрацию связки:

- быстрый запуск на сервере (venv/systemd/Docker),
- простая интеграция с фронтом (**ai-v5-demo**),
- понятный REST-контракт, за которым можно прятать любую модель (от EMA до Toto/ML).

В проде сервис пробует подключить **Datadog Toto** (если доступен), а при недоступности — автоматически использует лёгкий **EMA-fallback**. Контракт ответа стабильный, UI продолжает работать.

---

## 📚 Оглавление

- [🎯 Что делает сервис](#-что-делает-сервис)
- [🔌 API](#-api)
  - [GET /](#get-)
  - [GET /health](#get-health)
  - [POST /forecast](#post-forecast)
  - [POST /backtest](#post-backtest)
  - [Примеры curl](#примеры-curl)
- [🚀 Быстрый старт](#-быстрый-старт)
  - [Вариант A — локально (venv + uvicorn)](#вариант-a--локально-venv--uvicorn)
  - [Вариант B — systemd (прод)](#вариант-b--systemd-прод)
  - [Вариант C — Docker](#вариант-c--docker)
- [🔗 Интеграция с ai-v5-demo](#-интеграция-с-ai-v5-demo)
- [⚙️ Конфигурация](#-конфигурация)
- [🗂 Структура проекта](#-структура-проекта)
- [📝 Особенности реализации](#-особенности-реализации)
- [🩹 Траблшутинг](#-траблшутинг)
- [🗺️ Дорожная карта](#-дорожная-карта)
- [🙌 Ссылки](#-ссылки)

---

## 🎯 Что делает сервис

- Принимает **один** или **несколько** рядов (`series` или `seriesMulti`).  
- Возвращает **прогноз** на `horizon` шагов вперёд — медиану + доверительный интервал.  
- Если **Datadog Toto** доступен — прогноз с него, иначе **EMA-fallback**, чтобы UI не падал.  
- В связке с моим UI (**ai-v5-demo**) прогнозируются:
  - **Chars/sec** (скорость печати),
  - **End-to-End Latency** (мс).

---

## 🔌 API

### GET `/`

Краткая информация о сервисе и состоянии бэкенда.

**Ответ (пример):**
```json
{
  "name": "toto-server",
  "version": "1.1.1",
  "engine": "toto-open-1.0",
  "device": "cpu",
  "ready": true,
  "load_error": null,
  "endpoints": ["/health", "/forecast", "/backtest"]
}
```

### GET `/health`

Проверка статуса.

**Ответ (пример):**
```json
{
  "ok": true,
  "engine": "ema-fallback",
  "device": "cpu",
  "ready": false,
  "load_error": "ImportError: ..."
}
```

> `engine`: `toto-open-1.0` (если Toto доступен) или `ema-fallback`.  
> `ready`: готов ли Toto-движок.

### POST `/forecast`

**Вход (один ряд):**
```json
{
  "series": [0.82, 1.05, 0.97, 1.21],
  "horizon": 3,
  "confidence": 0.9,
  "samples": 256,
  "intervalSeconds": 60
}
```

**Вход (несколько рядов):**
```json
{
  "seriesMulti": [
    [180, 220, 240, 210, 250, 270, 260, 275],
    [1200, 1100, 1050, 1300, 900, 980, 990, 1010]
  ],
  "horizon": 3,
  "confidence": 0.9,
  "samples": 256,
  "intervalSeconds": 60
}
```

**Поля:**
- `series` — один временной ряд (≥ 3 точки).  
- `seriesMulti` — список рядов (все одинаковой длины, каждый ≥ 3 точки).  
- `horizon` — глубина прогноза.  
- `confidence` — доверительная вероятность (0.5…0.995).  
- `samples` — число выборок для Toto (если доступен).  
- `intervalSeconds` — шаг между точками (для Toto).

**Ответ (пример):**
```json
{
  "engine": "toto-open-1.0",
  "device": "cpu",
  "horizon": 3,
  "predictions": [1.10, 1.08, 1.06],
  "lower": [0.95, 0.94, 0.93],
  "upper": [1.25, 1.22, 1.19],
  "lastValue": 1.21,
  "predictionsMulti": [[...], [...]],
  "lowerMulti": [[...], [...]],
  "upperMulti": [[...], [...]]
}
```

Если Toto недоступен — поля те же, прогноз формируется **EMA-fallback**, чтобы фронт не ломался.

### POST `/backtest`

Оценка качества прогноза по скользящему окну (на EMA-fallback).

**Вход:**
```json
{
  "series": [180,220,240,210,250,270,260,275,285,295,305,290,300],
  "horizon": 3,
  "window": 8,
  "confidence": 0.9
}
```

**Ответ (пример):**
```json
{
  "mae": 12.34,
  "mape": 4.56,
  "coverage": 0.83,
  "samples": 5,
  "engine": "ema-fallback"
}
```

### Примеры `curl`

```bash
# состояние
curl -s http://127.0.0.1:8000/ | jq
curl -s http://127.0.0.1:8000/health | jq

# прогноз (один ряд)
curl -s http://127.0.0.1:8000/forecast \
  -H 'content-type: application/json' \
  -d '{"series":[0.9,1.0,0.95,1.1],"horizon":3,"confidence":0.9}' | jq

# прогноз (несколько рядов)
curl -s http://127.0.0.1:8000/forecast \
  -H 'content-type: application/json' \
  -d '{"seriesMulti":[[180,220,240,210,250,270,260,275],[1200,1100,1050,1300,900,980,990,1010]],"horizon":3,"confidence":0.9}' | jq

# бэктест
curl -s http://127.0.0.1:8000/backtest \
  -H 'content-type: application/json' \
  -d '{"series":[180,220,240,210,250,270,260,275,285,295,305,290,300],"horizon":3,"window":8,"confidence":0.9}' | jq
```

---

## 🚀 Быстрый старт

### Вариант A — локально (venv + uvicorn)

```bash
git clone git@github.com:andy-sklv/toto-server.git
cd toto-server

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# запуск
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Вариант B — systemd (прод)

Файлы на сервере, например: `/opt/toto-server`.

```ini
# /etc/systemd/system/toto.service
[Unit]
Description=Toto Forecast API (FastAPI)
After=network.target

[Service]
WorkingDirectory=/opt/toto-server
Environment="PATH=/opt/toto-server/.venv/bin"
ExecStart=/opt/toto-server/.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable toto
sudo systemctl start toto
sudo systemctl status toto
journalctl -u toto -f
```

### Вариант C — Docker

```bash
docker build -t toto-server:latest .
docker run -d --name toto-server -p 8000:8000 toto-server:latest
```

---

## 🔗 Интеграция с ai-v5-demo

Во фронте **ai-v5-demo**:
```bash
NEXT_PUBLIC_TOTO_ENABLED=true
TOTO_PROXY_URL=http://host.docker.internal:8000
```

Фронт шлёт в `/api/toto` → прокси на `${TOTO_PROXY_URL}/forecast`.  
Если Toto недоступен — UI всё равно работает (можно скрыть панель).

---

## ⚙️ Конфигурация

Необязательные переменные окружения:

- `TOTO_DEVICE` — `cpu`/`cuda` (если GPU).  
- `TOTO_CHECKPOINT` — чекпойнт Toto (по умолчанию `Datadog/Toto-Open-Base-1.0`).

> Специально не передаю `memory_efficient_attention` (именно он падал в логах на некоторых версиях Toto).

---

## 🗂 Структура проекта

```text
toto-server/
 ├─ main.py              # FastAPI + EMA fallback + (опц.) Toto
 ├─ requirements.txt     # fastapi, uvicorn, numpy, toto (опц.)
 ├─ Dockerfile           # образ с uvicorn
 ├─ README.md
 └─ .gitignore           # обязательно: .venv/, __pycache__/, *.pyc
```

---

## 📝 Особенности реализации

- Грейсфул-фоллбек: если Toto не загрузился — автоматически EMA, контракт тот же.  
- Мульти-серии: можно предсказывать сразу несколько рядов одним запросом.  
- Интервалы: доверительные границы из остатков (EMA-подход), у Toto — квантильные прогнозы.  
- Стартап-инициализация: загрузка Toto в startup-хуке, health показывает состояние.  
- Сценарии эксплуатации: локально/прод через systemd/Docker.

---

## 🩹 Траблшутинг

**Problem:** `ModuleNotFoundError: numpy`  
**Fix:** Вы точно в venv?  
```bash
source .venv/bin/activate && pip install -r requirements.txt
```

**Problem:** `TypeError: TotoBackbone.__init__() got an unexpected keyword argument 'memory_efficient_attention'`  
**Fix:** Используйте текущий `main.py`. Я удалил этот аргумент в загрузке Toto — теперь ок.

**Problem:** Сервис падает в systemd  
**Fix:** Проверьте `Environment="PATH=/opt/toto-server/.venv/bin"` и права на папку. Логи:  
```bash
journalctl -u toto -n 200 --no-pager
```

**Problem:** CORS  
**Fix:** Обычно зову сервер через бэкенд-прокси (Next API Route), CORS не нужен. Если зовёте напрямую из браузера — включите CORS в FastAPI.

---

## 🗺️ Дорожная карта

- `/forecast/batch` (несколько метрик пакетом).  
- Доп. модели: ARIMA/Prophet/ML и выбор движка по метрике.  
- Экспорт метрик в Prometheus/OpenTelemetry.  
- Конфигурируемые пайплайны нормализации/детрендинга.  
- Нагрузочные тесты и e2e.

---

## 🙌 Ссылки

- UI (фронт): https://github.com/andy-sklv/ai-v5-demo  
- Этот сервер: https://github.com/andy-sklv/toto-server

---