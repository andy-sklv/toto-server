# toto-server  
_Мини-сервис на FastAPI для прогноза временных рядов (демо под Datadog Toto)_

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi)
![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI-4B8BBE)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Привет! 👋 Это мой маленький сервер-прогнозист: принимает короткий временной ряд (например, **скорость генерации текста** из UI), и возвращает **прогноз** на несколько шагов вперёд. Я сделал его максимально простым, чтобы:

- быстро поднять на сервере;
- легко интегрировать с фронтом (**ai-v5-demo**);
- показать пайплайн: **Python + FastAPI + Uvicorn + systemd/Docker**.

> Архитектурно — это тонкий REST-слой, за которым можно спрятать что угодно: от скользящей средней до ARIMA/Prophet/ML. В демо используется лёгкая эвристика (например, EMA/скользящая средняя) — важен контракт и связка с фронтом.

---

## 📚 Оглавление

- [🎯 Что делает сервис](#-что-делает-сервис)
- [🔌 API](#-api)
  - [GET /health](#get-health)
  - [POST /forecast](#post-forecast)
  - [Примеры curl](#примеры-curl)
- [🚀 Быстрый старт](#-быстрый-старт)
  - [Вариант A — локально (venv + uvicorn)](#вариант-a--локально-venv--uvicorn)
  - [Вариант B — systemd (прод)](#вариант-b--systemd-прод)
  - [Вариант C — Docker](#вариант-c--docker)
- [🔗 Интеграция с ai-v5-demo](#-интеграция-с-ai-v5-demo)
- [⚙️ Конфигурация](#-конфигурация)
- [🗂 Структура проекта](#-структура-проекта)
- [📝 Тонкости и заметки](#-тонкости-и-заметки)
- [🩹 Траблшутинг](#-траблшутинг)
- [🗺️ Дорожная карта](#-дорожная-карта)
- [🙌 Ссылки](#-ссылки)

---

## 🎯 Что делает сервис

- Принимает массив чисел `series` (последние измерения метрики).
- Возвращает **прогноз** на `horizon` шагов вперёд (по умолчанию 3).
- Лёгкий REST-эндпоинт, чтобы фронт мог регулярно кормить метриками и мгновенно получать предсказания.
- В связке с моим UI (**ai-v5-demo**) можно прогнозировать:
  - **Chars/sec** (скорость генерации ответа),
  - **End-to-End latency** (мс).

---

## 🔌 API

### GET `/health`

Проверка живости сервиса.

**Ответ:**
```json
{ "ok": true }
```

### POST `/forecast`

**Вход:**
```json
{
  "series": [0.82, 1.05, 0.97, 1.21],
  "horizon": 3,
  "metric": "chars_per_sec"
}
```
- `series` — массив чисел (последние наблюдения).  
- `horizon` — сколько шагов предсказать вперёд (по умолчанию: 3).  
- `metric` — произвольная подпись (для логов/маршрутизации), опционально.

**Выход (пример):**
```json
{
  "engine": "toto-py-demo",
  "horizon": 3,
  "predictions": [1.10, 1.08, 1.06],
  "lastValue": 1.21
}
```
- `engine` — идентификатор внутреннего движка (для прозрачности).  
- `predictions` — массив прогноза длины `horizon`.  
- `lastValue` — последнее наблюдение из `series` (удобно для UI/логов).

> Нота: структура ответа стабильная и дружелюбная к фронту. Внутри можно легко заменить реализацию на ARIMA/Prophet/ML/LLM — **контракт не меняется**.

### Примеры `curl`

```bash
# health
curl -s http://127.0.0.1:8000/health

# forecast
curl -s -X POST http://127.0.0.1:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"series":[0.9,1.0,0.95,1.1],"horizon":3,"metric":"chars_per_sec"}'
```

---

## 🚀 Быстрый старт

### Вариант A — локально (venv + uvicorn)

Требуется: **Python 3.10+**

```bash
git clone git@github.com:andy-sklv/toto-server.git
cd toto-server

# виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# зависимости
pip install --upgrade pip
pip install -r requirements.txt

# запуск (форграунд)
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# проверка
curl -s http://127.0.0.1:8000/health

# для разработки: авто-перезагрузка
python -m uvicorn main:app --reload --port 8000
```

### Вариант B — systemd (прод)

1) Разворачиваем в целевой директории, например `/opt/toto-server`:

```bash
sudo mkdir -p /opt/toto-server
sudo chown -R $USER:$USER /opt/toto-server
git clone git@github.com:andy-sklv/toto-server.git /opt/toto-server
cd /opt/toto-server

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Юнит `/etc/systemd/system/toto.service`:

```ini
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

3) Запуск и проверка:
```bash
sudo systemctl daemon-reload
sudo systemctl enable toto
sudo systemctl start toto
sudo systemctl status toto

# логи:
journalctl -u toto -f
```

### Вариант C — Docker

```bash
# сборка образа
docker build -t toto-server:latest .

# запуск контейнера
docker run -d --name toto-server -p 8000:8000 toto-server:latest

# проверка
curl -s http://127.0.0.1:8000/health
```

> Можно прикрутить docker-compose и пробросить конфиги/сети — демо живёт и без этого.

---

## 🔗 Интеграция с ai-v5-demo

Во фронте **ai-v5-demo** достаточно указать URL сервиса (если фронт в Docker, а сервер на хосте — удобно так):

```env
# .env в ai-v5-demo
NEXT_PUBLIC_TOTO_ENABLED=true
TOTO_PROXY_URL=http://host.docker.internal:8000
```
Фронт стучится в `/api/toto` → прокси на `TOTO_PROXY_URL/forecast`.  
Если Toto недоступен — UI остаётся работоспособным (можно скрыть панель).

---

## ⚙️ Конфигурация

Сервис не требует обязательных переменных окружения. Возможные параметры, если заложите их в код:

- `PORT` — порт приложения (согласуйте с systemd/Docker).
- `LOG_LEVEL` — уровень логирования (`INFO`/`DEBUG`).
- `CORS_ORIGINS` — список origin'ов для CORS, если планируете прямой вызов из браузера.

> В демо я чаще использую бэкенд-прокси во фронте (Next API Route), поэтому CORS обычно не нужен.

---

## 🗂 Структура проекта

```text
toto-server/
 ├─ main.py              # FastAPI-приложение (эндпоинты /health и /forecast)
 ├─ requirements.txt     # зависимости (fastapi, uvicorn, numpy и т.д.)
 ├─ Dockerfile           # лёгкий образ с uvicorn
 ├─ README.md            # этот файл
 └─ (опц.) core/ utils/  # прогнозный модуль, утилиты и т.п.
```

В `main.py` — небольшой “движок” прогноза (EMA/скользящая средняя). Хотите серьёзнее — замените реализацию, **не трогая контракт**.

---

## 📝 Тонкости и заметки

- **Стабильность интерфейса:** контракт запроса/ответа фиксирован, фронту не нужно адаптироваться под внутренние правки.
- **Производительность:** Uvicorn спокойно держит сотни RPS для простых эвристик. Для ML-моделей вынесите загрузку в стартап-хуки.
- **Расширяемость:** легко добавить `/forecast/batch`, `/metrics`, разные горизонты, нормализацию.
- **Безопасность:** за reverse-proxy (nginx/traefik), включить rate-limit и auth (token/header), если сервис наружу.

---

## 🩹 Траблшутинг

**Command 'uvicorn' not found**  
Вы в venv? Проверьте:
```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn main:app --port 8000
```

**Порт занят**  
`sudo lsof -i :8000` → остановите лишний процесс или смените порт.

**CORS ошибки в браузере**  
Включите CORS в FastAPI или вызывайте сервер через ваш бэкенд-прокси (как в ai-v5-demo).

**systemd: “failed”**  
`journalctl -u toto -f` — смотрим логи; часто проблема в `PATH`/виртуальном окружении.

---

## 🗺️ Дорожная карта

- `/forecast/batch` (несколько метрик одним запросом).
- Поддержка ARIMA/Prophet/ML-моделей.
- Экспорт внутренних метрик Prometheus/OpenTelemetry.
- Конфигурируемые `horizon`, сглаживание, нормализация.
- E2E и нагрузочные тесты.

---

## 🙌 Ссылки

- UI (фронт): <https://github.com/andy-sklv/ai-v5-demo>
- Этот сервер: <https://github.com/andy-sklv/toto-server>
