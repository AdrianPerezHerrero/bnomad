"""A complete tiny offline fit using only public APIs."""
from hdpgpc import HDPGPC
from hdpgpc.examples import configuration, waveforms

if __name__ == '__main__':
    model = HDPGPC(configuration('offline')).fit(waveforms())
    print(model.labels_)
    print(model.predictive(0).mean)
