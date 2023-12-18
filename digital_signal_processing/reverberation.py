from asteroid.models import BaseModel

from digital_signal_processing import utils


model = BaseModel.from_pretrained("cankeles/ConvTasNet_WHAMR_enhsingle_16k")
## сетка 16кГц, а гоним через нее все подряд - у тебя файлы с совершенно разными полосами (8 видел, 12 видел, 24 видел), хотя все в 48кГц
## а сетка обучена только на полосу 8
## конечно она мусор выдает
## я поэтому и предлагал дипфильтрнет, он во-первых обучен на 48кГц файлах, а во-вторых более-менее устойчив к порезанной полосе


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
