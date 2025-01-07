import pandas as pd
from sklearn.utils import resample

def fit(data, target_column, rate):
    class_counts = data[target_column].value_counts()
    total_samples = len(data)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    w = sum(class_weights.values())
    balanced_data = pd.DataFrame()

    for cls, count in class_counts.items():
        ni_prime = max(int((count * class_weights[cls] / w) * rate), 2)

        if ni_prime <= count:
            undersampled_data = resample(
                data[data[target_column] == cls],
                replace=False,
                n_samples=ni_prime,
                random_state=42
            )
            balanced_data = pd.concat([balanced_data, undersampled_data], ignore_index=True)
        else:
            oversampled_data = resample(
                data[data[target_column] == cls],
                replace=True,
                n_samples=ni_prime - count,
                random_state=42
            )
            balanced_data = pd.concat([balanced_data, data[data[target_column] == cls], oversampled_data],
                                      ignore_index=True)

    return balanced_data
