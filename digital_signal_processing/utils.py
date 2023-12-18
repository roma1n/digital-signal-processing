import librosa
import numpy as np
import soundfile as sf
import tqdm


MAX_FREQ = 24000


def load_audio(path):
    audio, _ = librosa.load(path, sr=MAX_FREQ)
    audio = librosa.resample(audio, orig_sr=MAX_FREQ, target_sr=2 * MAX_FREQ)
    return audio


def write_audio(audio, fname):
    sf.write(
        file=fname,
        data=audio,
        samplerate=2 * MAX_FREQ,
    )


def align(orig, recording):
    '''
    Выравнивает оригинальный файл и запись, максимизируя корреляцию
    '''
    print('Running align')

    assert orig.shape[-1] <= recording.shape[-1], 'Expected orig to be longer than recording'
    if orig.shape[-1] == recording.shape[-1]:
        return recording

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
        left = max(mid - 3 * step, 0)
        right = min(mid + 3 * step, recording.shape[0] - orig.shape[0])
    print(f'Selected shift: {mid}')

    ## найс

    return recording[mid:orig.shape[0] + mid]
