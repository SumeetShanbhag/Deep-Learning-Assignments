import numpy as np

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return np.dot(A,B) - C

def problem_1c (A, B, C):
    return A * B + np.transpose(C) 

def problem_1d (x, y):
    return np.dot(np.transpose(x), y)

def problem_1e (A, x):
    return np.linalg.solve(A,x)

def problem_1f (A, i):
    N = np.sum(A[i, ::2])
    return N

def problem_1g (A, c, d):
    greater_than_c = A[np.nonzero(A > c)]
    less_than_d = greater_than_c[np.nonzero(greater_than_c < d)]
    return np.mean(less_than_d)

def problem_1h (A, k):
    w,v = np.linalg.eig(A)
    return np.delete(v[:,np.argsort(w)], np.s_[:(len(A) - k)], axis = 1)

def problem_1i (x, k, m, s):
    z = np.ones(len(x))
    return np.transpose(np.random.multivariate_normal(x + m * z, s * np.eye(len(x)), k))

def problem_1j (A):
    return np.random.shuffle(A)

def problem_1k (x):
    return (x - np.mean(x)) / np.std(x)

def problem_1l (x, k):
    return np.repeat(x[: , np.newaxis], k, axis =1)

def problem_1m (X, Y):
    X_3d = np.atleast_3d(X)
    Y_3d = np.atleast_3d(Y)
    X_3d = np.repeat(X_3d, Y.shape[0], axis=1)
    Y_3d = np.repeat(Y_3d, X.shape[0], axis=0)
    D_ij = np.sum((X_3d - Y_3d)**2, axis=-1)
    return np.sqrt(D_ij)

def problem_1n (matrices):
    C = 1
    for i in range(len(matrices)-1):
        C *= matrices[i].shape[0] * matrices[i+1].shape[1]
    return C

def linear_regression (X_tr, y_tr):
    ones = np.ones((X_tr.shape[0],1))
    X_w = np.hstack((X_tr,ones))
    a=np.matmul(np.transpose(X_w),X_w)
    c=np.dot(np.linalg.inv(a),X_w.T)
    w_hat = c.dot(y_tr)
    return w_hat[:-1],w_hat[-1]
    # return w, b

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    w, b = linear_regression(X_tr, ytr)
    # Report fMSE cost on the training and testing data (separately)
    y_hat_tr = (X_tr @ w)+b
    y_hat_te = (X_te @ w)+b
    
    MSE_tr = np.sum(np.square(y_hat_tr - ytr))/(2*X_tr.shape[0])
    MSE_te = np.sum(np.square(y_hat_te - yte))/(2*X_te.shape[0])
    print("Mean Squared Error for the testing set: ", MSE_te)
    print("Mean Squared Error for the training set: ",MSE_tr)

train_age_regressor()
