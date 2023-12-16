from asteroid.models import BaseModel

from digital_signal_processing import utils


model = BaseModel.from_pretrained("cankeles/ConvTasNet_WHAMR_enhsingle_16k")


FILES = [
    'data/gt.wav',
    'data/gt_recording.wav',
    'data/gt_transformed_resp_5000.wav',
    'data/gt_transformed.wav',
]


def reverberation():
    for fname in FILES:
        model.separate(
            wav=fname,
            output_dir='data/reverberation',
            force_overwrite=False,
            resample=True,
        )


if __name__ == '__main__':
    reverberation()
