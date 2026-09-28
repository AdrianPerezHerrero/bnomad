"""Shared offline assignments for a two-output synthetic sequence."""
from hdpgpc import HDPGPC
from hdpgpc.examples import configuration, waveforms

if __name__ == '__main__':
    model = HDPGPC(configuration('multi_output')).fit(waveforms(outputs=2))
    print(model.labels_)
    print([state.filtered_mean for state in model.posterior(0)])
