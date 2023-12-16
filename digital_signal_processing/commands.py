import fire

from digital_signal_processing import (
    equalizer,
    metrics,
    reverberation,
)


class Sound:
    def __init__(self):
        print('Running sound')

    def test(self):
        print('Hello world!')

    def equalizer(self):
        equalizer.equalizer()

    def metrics(self):
        metrics.metrics()

    def reverberation(self):
        reverberation.reverberation()


def main():
    fire.Fire(Sound())


if __name__ == '__main__':
    main()
