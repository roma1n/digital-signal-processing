import json

import numpy as np
from scipy import (
    fft,
    signal,
)

from digital_signal_processing import utils


GAIN_CLIP = None
RESP_LEN = 10000


def get_band_index(spectrum, num_bands=64):
    return ((np.arange(spectrum.shape[0]) / (spectrum.shape[0] + 1)) * num_bands).astype(int)


def get_band_mean(spectrum, num_bands=64):
    '''
    Бьем частотную область на полосы (aka бэнды будущего эквалайзера) - лучше 32 полосы, можно 16, меньше 16 не стоит, больше 32 можно
    В каждой полосе берем либо центральное, либо среднее значение амплитуды
    '''
    band_index = get_band_index(spectrum, num_bands)
    bands = np.zeros(num_bands)
    counts = np.zeros(num_bands)
    bands[band_index] += spectrum
    counts[band_index] += np.ones_like(spectrum)
    return bands / counts


def get_gains(clip=None):
    '''
    Вычесляет гейны. Параметр clip позволяет ограничить выборсы
    '''
    sweep = utils.load_audio('data/sweep.wav')
    recording = utils.load_audio('data/sweep_recording.wav')

    recording = utils.align(sweep, recording)
    assert sweep.shape == recording.shape, 'Expected equal shapes'

    # Переводим оригинальный свипер и записанный свипер в частотную область
    sweep_fft = np.abs(fft.rfft(sweep))
    recording_fft = np.abs(fft.rfft(recording))

    # Делим один набор значений для записанного свипера на набор значений для оригинального - получаем набор гейнов эквалайзера
    # Добавляем epsilon в знаменатель, чтобы избавиться от деления на 0
    gains = get_band_mean(recording_fft) / (get_band_mean(sweep_fft) + 1e-10)
    if clip:
        gains = np.clip(gains, 1/clip, clip)
    return gains


def correct_spectrum(spectrum, gains):
    corrected = spectrum.copy()
    corrected[:utils.MAX_FREQ] *= gains[get_band_index(spectrum, num_bands=gains.shape[0])[:utils.MAX_FREQ]]
    corrected[utils.MAX_FREQ:] = 0
    return corrected


def process_white_noise(gains, gain_clip=None):
    '''
    Корректирует белый шум с помощью гейнов
    '''
    white_noise = utils.load_audio('data/white_noise.wav')
    white_noise_fft = fft.rfft(white_noise)

    corrected_white_noise_fft = correct_spectrum(white_noise_fft, gains)
    corrected_white_noise = fft.irfft(corrected_white_noise_fft)

    fname = f'data/corrected_white_noise_clip_{gain_clip}.wav' if gain_clip else 'data/corrected_white_noise.wav'
    audio = utils.write_audio(
        audio=corrected_white_noise,
        fname=fname,
    )



def get_corrected_white_noise(gain_clip=None):
    gains = get_gains(clip=gain_clip)
    with open(f'data/gains_clip_{gain_clip}.json' if gain_clip else 'data/gains.json', 'w') as f:
        f.write(json.dumps({'gains': gains.tolist()}, indent=4))
    process_white_noise(gains, gain_clip)


def get_response():
    '''
    Вычисляет импульсный отклик
    '''
    white_noise = utils.load_audio('data/white_noise.wav')
    recording = utils.load_audio('data/corrected_white_noise_recording.wav')
    recording = utils.align(white_noise, recording)
    return signal.deconvolve(recording, white_noise)[1]


def transform_gt(resp_len=None):
    '''
    Сворачивает gt с импульсным откликом
    '''
    gt = utils.load_audio('data/gt.wav')
    response = get_response()
    if resp_len:
        response = response[:resp_len]
    gt_transformed = signal.convolve(gt, response, mode='same')

    fname = f'data/gt_transformed_resp_{resp_len}.wav' if resp_len else 'data/gt_transformed.wav'
    utils.write_audio(
        audio=gt_transformed,
        fname=fname,
    )


def equalizer():
    get_corrected_white_noise(gain_clip=GAIN_CLIP)
    transform_gt(resp_len=RESP_LEN)


if __name__ == '__main__':
    equalizer()
