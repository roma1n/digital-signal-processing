# Digital signal processing (homework)

Как установить venv и зависимости:

```
poetry install
```

## Модуль 1

Вычислить скорректированный белый шум, импульсный отклик и свернуть gt с откликом:

```
poetry run sound equalizer
```

## Модуль 2

Вывести метрики:

```
poetry run sound metrics
```

Метрики:

|    | файл                              |       SDR |   SI-SDR |    PESQ |    DNSMOS | MOS          |
|---:|:----------------------------------|----------:|---------:|--------:|----------:|:-------------|
|  0 | data/gt.wav                       | 130.83    | 101.865  | 4.54864 |   3.44877 | set_mannualy |
|  1 | data/gt_recording.wav             |  -6.40356 | -14.5361 | 1.64996 |   1.86377 | set_mannualy |
|  2 | data/gt_transformed_resp_5000.wav |  -9.31557 | -24.1579 | 1.5475  |   2.32739 | set_mannualy |
|  3 | data/gt_transformed.wav           | -20.6623  | -51.2892 | 1.17754 | nan       | set_mannualy |


## Модуль 3

```
poetry run sound reverbration
```
