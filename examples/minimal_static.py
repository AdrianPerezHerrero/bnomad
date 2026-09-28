"""The supported static formulation with explicit small-data settings."""
from hdpgpc import HDPGPC
from hdpgpc.examples import configuration, waveforms

if __name__ == '__main__':
    model = HDPGPC(configuration('static')).fit(waveforms())
    print(model.labels_)
