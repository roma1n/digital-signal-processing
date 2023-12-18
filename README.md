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

К исходному алгоритму добавил два улучшения:
1. Ограничение гейнов: в первых подоходах для некоторых полос гейны получались на 5-6 порядков больше, чем для других. Из-за этого белый шум превращался в звук определенной частоты. Позже нашел баг в коде и выключил ограничение. Тем не менее, все равно может быть полезно в некоторых кейсах
2. Ограничение времени отклика: без ограничения отклик был слишком долгий и речь в gt.wav сливалась в единый гул. Пробовал ограничивать 0.1 и 0.2 секунды, остановился на 0.1 для дальнейших измерений

## Модуль 2

Вывести метрики:

```
poetry run sound metrics
```

Метрики:

|    | файл                              |       SDR |   SI-SDR |    PESQ |    DNSMOS | MOS          |
|---:|:----------------------------------|----------:|---------:|--------:|----------:|:-------------|
|  0 | data/gt.wav                       | 130.83    | 101.865  | 4.54864 |   3.44877 | 5 |
|  1 | data/gt_recording.wav             |  -6.40356 | -14.5361 | 1.64996 |   1.86377 | 4 |
|  2 | data/gt_transformed_resp_5000.wav |  -9.31557 | -24.1579 | 1.5475  |   2.32739 | 3 |
|  3 | data/gt_transformed.wav           | -20.6623  | -51.2892 | 1.17754 | nan       | 1 |


## Модуль 3

Запуск:

```
poetry run sound reverberation
```

Метрики:

|    | файл                                                 |       SDR |   SI-SDR |    PESQ |    DNSMOS | MOS          |
|---:|:-----------------------------------------------------|----------:|---------:|--------:|----------:|:-------------|
|  0 | data/gt.wav                                          | 130.83    | 101.865  | 4.54864 |   3.44877 | 5 |
|  1 | data/gt_recording.wav                                |  -6.40356 | -14.5361 | 1.64996 |   1.86377 | 4 |
|  2 | data/gt_transformed_resp_5000.wav                    |  -9.42737 | -25.0661 | 1.57305 |   2.26615 | 2 |
|  3 | data/gt_transformed.wav                              | -23.8184  | -47.8303 | 1.09421 |   1.06475 | 1 |
|  4 | data/reverberation/gt_est1.wav                       |  15.7618  |   9.6743 | 3.3637  |   3.39322 | 4 |
|  5 | data/reverberation/gt_recording_est1.wav             |  -4.49469 | -16.9171 | 1.66143 |   2.14973 | 4 |
|  6 | data/reverberation/gt_transformed_resp_5000_est1.wav |  -9.78219 | -25.6921 | 1.47645 |   2.36188 | 3 |
|  7 | data/reverberation/gt_transformed_est1.wav           | -26.228   | -35.5367 | 1.13301 | nan       | 1 |

[Сетка-улучшатель](https://huggingface.co/cankeles/ConvTasNet_WHAMR_enhsingle_16k) не сильно улучшила записи. Для gt.wav, наоборот, стало хуже
