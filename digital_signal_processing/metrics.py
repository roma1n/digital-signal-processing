import librosa
import pandas as pd
from speechmos import dnsmos as dnsmos_impl
import torch
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ScaleInvariantSignalDistortionRatio,
    SignalDistortionRatio,
)

from digital_signal_processing import utils

PESQ_SAMPLING_FREQ = 8000
DNSMOS_SAMPLING_FREQ = 16000


TARGET_FILE = 'data/gt.wav'

FILES = [
    'data/gt.wav',
    'data/gt_recording.wav',
    'data/gt_transformed_resp_5000.wav',
    'data/gt_transformed.wav',
]


def signal_distortion_ratio(preds, target):
    return SignalDistortionRatio()(
        preds=torch.Tensor(preds),
        target=torch.Tensor(target),
    ).item()


def scale_invariant_signal_distortion_ratio(preds, target):
    return ScaleInvariantSignalDistortionRatio()(
        preds=torch.Tensor(preds),
        target=torch.Tensor(target),
    ).item()


def perceptual_evaluation_speech_quality(preds, target):
    return PerceptualEvaluationSpeechQuality(PESQ_SAMPLING_FREQ, 'nb')(
        preds=torch.Tensor(librosa.resample(preds, orig_sr=2 * utils.MAX_FREQ, target_sr=PESQ_SAMPLING_FREQ)),
        target=torch.Tensor(librosa.resample(target, orig_sr=2 * utils.MAX_FREQ, target_sr=PESQ_SAMPLING_FREQ)),
    ).item()


def dnsmos(preds):
    try:
        return dnsmos_impl.run(
            sample=librosa.resample(preds, orig_sr=2 * utils.MAX_FREQ, target_sr=DNSMOS_SAMPLING_FREQ),
            sr=DNSMOS_SAMPLING_FREQ,
        )['ovrl_mos']
    except ValueError as error:
        pass


def metrics():
    target = utils.load_audio(TARGET_FILE)

    result = []

    for fname in FILES:
        preds = utils.load_audio(fname)
        preds = utils.align(target, preds)

        result.append({
            'файл': fname,
            'SDR': signal_distortion_ratio(preds, target),
            'SI-SDR': scale_invariant_signal_distortion_ratio(preds, target),
            'PESQ': perceptual_evaluation_speech_quality(preds, target),
            'DNSMOS': dnsmos(preds),
            'MOS': 'set_mannualy',
        })
    
    result = pd.DataFrame(result)
    print(result.to_markdown())


if __name__ == '__main__':
    metrics()
