"""Dataset-independent reports for clustered sequence experiments."""

from collections import Counter

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster_label_statistics(assignments, labels):
    """Return in-sample majority-label scores and a true/predicted confusion table.

    The returned tuple is ``(scores, classes, confusion, cluster_labels)``.
    Majority labels are fitted and evaluated on the supplied data; their
    accuracy is descriptive, not a held-out classification estimate.
    """
    assignments = np.asarray(assignments).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if not len(assignments) or len(assignments) != len(labels):
        raise ValueError('assignments and labels must have the same nonzero length')
    clusters = np.unique(assignments)
    classes = np.unique(labels)
    contingency = np.zeros((len(clusters), len(classes)), dtype=np.int64)
    for row, cluster in enumerate(clusters):
        counts = Counter(labels[assignments == cluster].tolist())
        for column, label in enumerate(classes):
            contingency[row, column] = counts[label]
    majority_labels = classes[np.argmax(contingency, axis=1)]
    cluster_labels = dict(zip(clusters.tolist(), majority_labels.tolist()))
    assigned_labels = majority_labels[np.searchsorted(clusters, assignments)]
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(confusion, (np.searchsorted(classes, labels),
                          np.searchsorted(classes, assigned_labels)), 1)
    correct = int(np.trace(confusion))
    scores = {
        'purity': correct / len(labels),
        'majority_label_accuracy_in_sample': correct / len(labels),
        'adjusted_rand_index': float(adjusted_rand_score(labels, assignments)),
        'normalized_mutual_information': float(normalized_mutual_info_score(
            labels, assignments, average_method='arithmetic')),
    }
    return scores, classes, confusion, cluster_labels


def format_confusion_matrix(classes, matrix, *, true_name='true label',
                            assigned_name='cluster-majority label'):
    """Format counts with true classes in rows and assigned classes in columns."""
    classes = np.asarray(classes).reshape(-1)
    matrix = np.asarray(matrix)
    if matrix.shape != (len(classes), len(classes)):
        raise ValueError('matrix must be square with one row and column per class')
    row_width = max(13, *(len(str(label)) for label in classes))
    column_names = [str(label) for label in classes]
    cell_width = max(5, len(str(matrix.sum())), *(len(name) for name in column_names))
    lines = [f'Confusion matrix (rows={true_name}, columns={assigned_name}; counts):',
             f"  {'Real/assigned':>{row_width}} | "
             + ' '.join(f'{name:>{cell_width}}' for name in column_names)
             + f" | {'Total':>{cell_width}}"]
    for label, row in zip(classes, matrix):
        lines.append(f'  {str(label):>{row_width}} | '
                     + ' '.join(f'{int(count):>{cell_width}}' for count in row)
                     + f' | {int(row.sum()):>{cell_width}}')
    lines.append(f"  {'Total':>{row_width}} | "
                 + ' '.join(f'{int(total):>{cell_width}}' for total in matrix.sum(axis=0))
                 + f' | {int(matrix.sum()):>{cell_width}}')
    return '\n'.join(lines)


def format_lds_priors(*, dimension, model_type, observation_noise,
                      process_noise, degrees_of_freedom, source, policies):
    """Describe the selected LDS prior centers and strengths for a run log."""
    def matrix_center(name):
        value = policies[name].value
        return f'I_{dimension}' if value is None else f'custom {np.asarray(value).shape} matrix'

    def noise_center(name, scalar):
        value = policies[name].value
        return (f'{scalar:.6g} * I_{dimension}' if value is None
                else f'custom {np.asarray(value).shape} covariance')

    lines = ['------ Initial LDS parameter priors ------',
             f'Source={source}; LDS dimension={dimension}']
    if model_type == 'dynamic':
        for matrix, noise, scale in (('A', 'Gamma', process_noise),
                                     ('C', 'Sigma', observation_noise)):
            matrix_policy, noise_policy = policies[matrix], policies[noise]
            n0 = degrees_of_freedom * noise_policy.strength
            lines.append(
                f'{matrix}: {matrix_policy.mode}, center={matrix_center(matrix)}, '
                f'column prior precision={matrix_policy.strength:g} * I_{dimension}'
                if matrix_policy.mode != 'fixed' else
                f'{matrix}: fixed at {matrix_center(matrix)}')
            lines.append(
                f'{noise}: {noise_policy.mode}, center={noise_center(noise, scale)}, '
                f'excess degrees n0={n0:g}' if noise_policy.mode != 'fixed' else
                f'{noise}: fixed at {noise_center(noise, scale)}')
    else:
        lines.extend([
            'A: absent (static dynamics)',
            f"C: {policies['C'].mode} at {matrix_center('C')}",
            f'Gamma: 0_{dimension} (static model; no A/Gamma posterior KL)',
            f"Sigma: {policies['Sigma'].mode}, "
            f"center={noise_center('Sigma', observation_noise)}",
        ])
    lines.append('------------------------------------------')
    return '\n'.join(lines)


class Tee:
    """Forward console writes to both a terminal and a log stream."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)
