"""Run after pip install .; no dataset downloads or path modifications."""
from hdpgpc import HDPGPC
from hdpgpc.examples import configuration, waveforms

if __name__ == '__main__':
    model = HDPGPC(configuration('online'))
    for observation in waveforms()[:4]:
        result = model.update(observation)
        print(result.assigned_cluster, result.new_cluster)
    print(model.labels_)
