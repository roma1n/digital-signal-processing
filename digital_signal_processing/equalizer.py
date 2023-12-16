import json

import librosa
import numpy as np
import soundfile as sf
from scipy import (
    fft,
    signal,
)
import tqdm

MAX_FREQ = 24000
# GAIN_CLIP = 5
GAIN_CLIP = None
RESP_LEN = 5000


def load_audio(path):
    audio, _ = librosa.load(path, sr=MAX_FREQ)
    audio = librosa.resample(audio, orig_sr=MAX_FREQ, target_sr=2 * MAX_FREQ)
    return audio


def get_band_index(spectrum, num_bands=64):
    return ((np.arange(spectrum.shape[0]) / (spectrum.shape[0] + 1)) * num_bands).astype(int)


def get_band_mean(spectrum, num_bands=64):
    '''
    Бьем частотную область на полосы (aka бэнды будущего эквалайзера) - лучше 32 полосы, можно 16, меньше 16 не стоит, больше 32 можно
    В каждой полосе берем либо центральное, либо среднее значение амплитуды
    '''
    band_index = ((np.arange(spectrum.shape[0]) / (spectrum.shape[0] + 1)) * num_bands).astype(int)
    bands = np.zeros(num_bands)
    counts = np.zeros(num_bands)
    bands[band_index] += spectrum
    counts[band_index] += np.ones_like(spectrum)
    return bands / counts


def align(orig, recording):
    '''
    Выравнивает оригинальный файл и запись, максимизируя корреляцию
    '''
    print('Running align')
    left = 0
    right = recording.shape[0] - orig.shape[0]
    mid = None
    print(f'Initial range for alignment selection: [{left}, {right}]')
    for step in [1000, 100, 10, 1]:
        vals = []
        shifts = []
        for shift in tqdm.tqdm(range(left, right, step)):
            vals.append(np.corrcoef(orig, recording[shift:orig.shape[0] + shift])[0, 1])
            shifts.append(shift)
        print(f'Step: {step}. Correlation: {np.max(vals):.4f}')
        mid = shifts[np.argmax(vals)]
        left = mid - 3 * step
        right = mid + 3 * step
    print(f'Selected shift: {mid}')
    return recording[mid:orig.shape[0] + mid]



def get_gains(clip=None):
    '''
    Вычесляет гейны. Параметр clip позволяет ограничить выборсы
    '''
    sweep = load_audio('data/sweep.wav')
    recording = load_audio('data/sweep_recording.wav')

    recording = align(sweep, recording)
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
    corrected[:MAX_FREQ] *= gains[get_band_index(spectrum[:MAX_FREQ], num_bands=gains.shape[0])]
    return corrected


def process_white_noise(gains, gain_clip=None):
    '''
    Корректирует белый шум с помощью гейнов
    '''
    white_noise = load_audio('data/white_noise.wav')
    white_noise_fft = fft.rfft(white_noise)

    corrected_white_noise_fft = correct_spectrum(white_noise_fft, gains)
    corrected_white_noise = fft.irfft(corrected_white_noise_fft)

    fname = f'data/corrected_white_noise_clip_{gain_clip}.wav' if gain_clip else 'data/corrected_white_noise.wav'
    sf.write(
        file=fname,
        data=corrected_white_noise,
        samplerate=2 * MAX_FREQ,
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
    white_noise = load_audio('data/white_noise.wav')
    recording = load_audio('data/corrected_white_noise_recording.wav')
    recording = align(white_noise, recording)
    return signal.deconvolve(recording, white_noise)[1]


def transform_gt(resp_len=None):
    '''
    Пре
    '''
    gt = load_audio('data/gt.wav')
    response = get_response()
    if resp_len:
        response = response[:resp_len]
    gt_transformed = signal.convolve(gt, response)

    fname = f'data/gt_transformed_resp_{resp_len}.wav' if resp_len else 'data/gt_transformed.wav'
    sf.write(
        file=fname,
        data=gt_transformed,
        samplerate=2 * MAX_FREQ,
    )


def equalizer():
    get_corrected_white_noise(gain_clip=GAIN_CLIP)
    transform_gt(resp_len=RESP_LEN)


if __name__ == '__main__':
    equalizer()
