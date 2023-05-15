import numpy as np
from sklearn.model_selection import train_test_split

def loadData():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48 * 48))
    ytr = np.atleast_2d(np.load("age_regression_ytr.npy")).T
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48 * 48))
    yte = np.atleast_2d(np.load("age_regression_yte.npy")).T
    return X_tr, ytr, X_te, yte


def data_PreProcessing(X_tr, y_tr):
    # split training and validation into 2 parts
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=8)
    return X_train, y_train, X_val, y_val


def sgd(X_train, y_train, X_val, y_val):
    # epoch, batch or mini batch, alphas and learning rate are the various hyperparameters
    num_epochs = [20, 30, 40, 50]
    mini_batches = [50, 100, 150, 200]
    alphas = [0.1, 0.01, 0.001, 0.0001]
    learning = [0.001, 0.0001, 0.00001, 0.01]
    fmse_tuned = 10000000000000000000000
    epoch_tuned = 8
    batch_tuned = 8
    alpha_tuned = 8
    eps_tuned = 8
    w_tuned = 8
    b_tuned = 8

    for epoch in num_epochs:
        for batch in mini_batches:
            for alpha in alphas:
                for eps in learning:
                    # resetting the w and b values for each epoch and each mini batch
                    w = np.mat(np.random.randn(X_train.shape[1])).T
                    b = np.random.randn(1, 1)

                    for e in range(len(num_epochs)):
                        mb_size = int(X_train.shape[0] // batch)
                        for i in range(mb_size):
                            beg = i * mb_size
                            end = beg + mb_size

                            X_train_mb = X_train[beg:end, :]
                            y_train_mb = y_train[beg:end]

                            yHat = np.dot(X_train_mb, w) + b
                            diff = yHat - y_train_mb
                            grad = ((X_train_mb.T @ diff) + (alpha * w)) / X_train_mb.shape[0]
                            w_new = w - (eps * grad)
                            w = w_new

                            b_grad = np.mean(diff)
                            b_new = b - (eps * b_grad)
                            b = b_new

                    y_hat_val = np.dot(X_val, w) + b
                    fmseVal = ((1 / (2 * len(y_hat_val))) * (np.sum(np.square(y_hat_val - y_val))))
                    regVal = np.dot(w.T, w)
                    fmseReg = fmseVal + (alpha / 2) * regVal

                    if (fmseReg < fmse_tuned):
                        fmse_tuned = fmseReg
                        epoch_tuned = e
                        batch_tuned = i
                        alpha_tuned = alpha
                        eps_tuned = eps
                        w_tuned = w
                        b_tuned = b
    return epoch_tuned, batch_tuned, alpha_tuned, eps_tuned, w_tuned, b_tuned


X_train, ytrain, X_val, y_val = data_PreProcessing(loadData()[0], loadData()[1])

epoch, batch, alpha, eps, w, b = sgd(X_train, ytrain, X_val, y_val)
print("epoch", epoch, "b", b, "eps", eps, "w", w, "alpha", alpha, "mini_batch", batch)
y_hat_Test = np.dot(loadData()[2], w) + b
fmse = (1 / (2 * y_hat_Test.shape[0])) * (np.sum(np.square(y_hat_Test - loadData()[3])))
print('the fmse value is : ',fmse)