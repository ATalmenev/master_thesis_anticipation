# Анализ поведения провайдеров ликвидности на Hyperliquid

Скрипты для регрессионного анализа поведения профессиональных
провайдеров ликвидности на децентрализованной перпетуальной бирже
Hyperliquid в окне 19:00-22:45 UTC 10 октября 2025 года.

## Содержание

```
reproducibility/
├── src/                  скрипты анализа
├── data/processed/       входные панели (parquet/json)
├── requirements.txt
└── README.md
```

## Настройка

Python 3.10+.

```
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
python3 unpack.py
```

`unpack.py` один раз распаковывает входные `.parquet` в `.csv`. После
этого запускайте entry-скрипты из корня пакета.

## Запуск

Каждый скрипт пишет результат в `data/processed/<имя>_summary.csv` и
печатает ключевые числа в stdout. Inference — studentized block
bootstrap-t, L=8 (40-second blocks), B=3000, ошибки кластеризуются на
21 fund master entities. Каждый скрипт CPU-bound, 5-30 минут.

### Intensive margin: specification ladder, outcome log(1+ALO)

Восемь колонок: коэффициент при per-coin signed pressure × pre-event
exposure, добавление FE, лагов и контролей по одному.

```
python3 src/horserace_ladder.py
```

Пишет `wallet_horserace_extended_summary.csv`.

### Extensive margin: presence

Outcome: `1{buy or sell order > 0}` на уровне wallet-coin-5s,
когорта tier-1+, linear trend.

```
python3 src/presence_regression.py
```

Пишет `wallet_presence_summary.csv`.

### Order-type selectivity, bid-ask symmetry, trading margin

Та же wallet-панель, те же FE, linear trend. Четыре outcome
(ALO как референс, GTC, buy, sell) плюс trading margin по
maker fills.

```
python3 src/selectivity_symmetry.py      # ALO, GTC, buy, sell
python3 src/trading_margin_fills.py      # maker fills
```

Пишут `wallet_gtc_symmetry_summary.csv` и
`wallet_fills_on_pressure_summary.csv`.

## Inference engine и общие модули

- `inference.py` — `run()`: FWL projection, two-way FE absorption,
  studentized block bootstrap-t.
- `bootstrap_utils.py` — `iter_demean`, `slope_se`.
- `cohort.py` — загрузка tier-1+ cohort и агрегация wallet → master.
- `pressure_signal.py` — построение per-coin signed pressure π_{c,t}.

## Входные данные

| Файл | Описание |
|------|----------|
| `fivesec_user_coin_panel_4h_20251010.csv` | Основная wallet-coin-5s panel, 19:00-22:45 UTC |
| `wallet_spread_panel_4h_20251010.csv` | Per-wallet maker fill counts для trading margin |
| `hlp_inventory_pressure_stock_by_coin_4h.csv` | Per-coin серия HLP inventory (numerator давления) |
| `tier_anchored_cohort_actual_labels.csv` | 228 tier-1+ кошельков с rebate tier |
| `lp_master_full_subs.json` | Master → sub-account map из `/info userRole` |
| `l2_controls_minute_4h_20251010.csv` | Coin-minute контроли по depth и funding |
