"""Small executable fit/streaming example using only the public API.

Run after installing the package: python examples/public_api.py
The underlying research implementation still prints progress.
"""

from pathlib import Path

import numpy as np
from hdpgpc import HDPGPC, HDPGPCConfig


def main():
    configurations = Path(__file__).with_name('configurations')
    a = np.array([0., .4, 1., .4, 0.])
    b = np.array([0., -.6, -1.2, -.6, 0.])
    X = np.stack([a, a + .01, b, b - .01])

    config = HDPGPCConfig.from_yaml((configurations / 'static.yaml').read_text())
    model = HDPGPC(config).fit(X)
    print('Batch labels:', model.labels_)

    config = HDPGPCConfig.from_yaml((configurations / 'online.yaml').read_text())
    stream = HDPGPC(config)
    for x in X:
        result = stream.update(x, include_prediction=True)
        print('Assignment:', result.assigned_cluster, 'New cluster:', result.new_cluster)
    print('Current cluster mean:', result.prediction.mean)


if __name__ == '__main__':
    main()
