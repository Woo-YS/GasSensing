import numpy as np

SEED = 36

def build_samples(df: dict, gas_to_label: dict):
    samples = []
    labels = []

    for gas, label in gas_to_label.items():
        merge = df[gas]["merge"].to_numpy()
        time = merge[:, 0]
        signal = merge[:, 1:]

        for i in range(signal.shape[1]):
            samples.append(signal[:, i].astype(np.float32))
            labels.append(label)

    x = np.stack(samples)
    num_classes = len(gas_to_label)
    y_index = np.array(labels)
    y_onehot = np.eye(num_classes, dtype=np.float32)[labels]

    print("x shape", x.shape)
    print("y_index shape", y_index.shape)
    print("y_onehot shape", y_onehot.shape)
    return x, y_index, y_onehot