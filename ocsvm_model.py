from sklearn.svm import OneClassSVM
import joblib
import numpy as np

def train_ocsvm_per_joint(data, seq_len=30, nu=0.1):
    # data: (total_frames, num_joints)
    models = []
    for j in range(data.shape[1]):
        # извлекаем окна для сустава
        X = []
        for i in range(len(data)-seq_len+1):
            X.append(data[i:i+seq_len, j])
        X = np.array(X)
        model = OneClassSVM(kernel='rbf', nu=nu)
        model.fit(X)
        models.append(model)
    return models