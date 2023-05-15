import numpy as np


# Training Dataset
X_train = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 28*28))
Y_train = np.load("fashion_mnist_train_labels.npy")

# Converting Y_train to a 10 different vectors for 10 different classes
Y_data_onehot=np.eye(10)[Y_train] 

# Splitting into Training and Validation
X_tr, X_va = np.split(X_train,[int(.8 * len(X_train))])
X_tr = X_tr.T
X_va = X_va.T
Y_tr, Y_va = np.split(Y_data_onehot, [int(.8 * len(X_train))])

# Loading Test Dataset
X_test = np.reshape(np.load("fashion_mnist_test_images.npy"), (-1, 28*28))
X_test = X_test.T
Y_test = np.load("fashion_mnist_test_labels.npy")

# Converting Y_test to a 10 different vectors for 10 different classes
Y_test_onehot = np.eye(10)[Y_test]

def SGD(X, Y, w, b, epoch, alpha, eps, mb, N):
    # Stochastic Gradient Descent
    for i in range(epoch):
        for j in range(0,N,mb):
            # This is sliced this way as the transpose is taken above
            X_mb = X[:,j:j+mb]
            Y_mb = Y[j:j+mb]
        
            # Calculating Predicted labels
            z = np.dot(X_mb.T,w)+b
            Y_mb_hat = np.exp(z)/np.sum(np.exp(z),axis=1)[:,None]
            
            # Gradients
            dw = (np.dot(X_mb,(Y_mb_hat-Y_mb))+alpha*w)/mb 
            db = np.sum(Y_mb_hat-Y_mb)/mb

            # Updating the Weights and Bias
            w-= eps*dw
            b-= eps*db
    return w,b


# Length of Training Examples
N = np.shape(X_tr)[1]

# Hyperparameters
# Below values are tuned values for the same for highest accuracy
Mb = [32]
Epoch = [35]
Alpha = [0.001]
Eps = [0.00001]

# Comment above lists and uncomment below lists to see 3*3*3*3 combinations

# Mb = [20, 30, 32] # Mini Batch
# Epoch = [5, 25, 35] # Epochs
# Alpha = [0.1, 0.01, 0.001] # Regularization Strength
# Eps = [0.00001, 0.00005, 0.000001] # Learning Rate

# This is done to ensure that the minimum cost value will be less 
# than the cost value obtained for the first iteration below
initial_cost = np.Inf 

H_star = np.array(4) #Initializing an array to store Best Hyperparameters

print("Finding best hyperparameters from 3*3*3*3 combinations")

W = np.zeros((np.shape(X_tr)[0],10))  #Randomly intializing the Weights
B = np.zeros(10) #Initializing bias = 0

for epoch in Epoch:
    for alpha in Alpha:
        for eps in Eps:
            for mb in Mb:
                # Stochastic Gradient Descent
                w,b = SGD(X_tr, Y_tr, W, B, epoch, alpha, eps, mb, N)

                # Calculating Predicted labels
                z_va = np.dot(X_va.T,w)+b 
                Y_va_hat = np.exp(z_va)/np.sum(np.exp(z_va),axis=1)[:,None]
                
                Y_va_hat_log = np.log(Y_va_hat, out=np.zeros_like(Y_va_hat))
                
                # Calculating MSE Loss on Validation Setimport warnings
                cost = -(np.trace(np.dot(Y_va,Y_va_hat_log.T)))/(np.shape(Y_va)[0]) + alpha*np.trace(np.dot(w.T,w))/(2*np.shape(Y_va)[0])

                # Updating Cost and Hyperparameters
                if(cost < initial_cost):
                    initial_cost = cost
                    H_star = [epoch, alpha, eps, mb]
                                     

print("Best Hyperparameters:\n") 
print("Epochs = ",  H_star[0], "\nAlpha = ",  H_star[1], "\nLearning Rate = ",  H_star[2], "\nMini Batch Size = ",  H_star[3])
print("Cost function value using above hyperparameter values ", initial_cost)
print("\nNow combining Training and validation Sets\n")

X_train = X_train.T

# Length of Training Examples
N_train = np.shape(X_train)[1]

# Stochastic Gradient Descent
w, b = SGD(X_train, Y_data_onehot, W, B, H_star[0], H_star[1], H_star[2], H_star[3], N_train)

# Calculating Predicted labels
z_te = np.dot(X_test.T,w)+b  
Y_test_hat = np.exp(z_te)/np.sum(np.exp(z_te),axis=1)[:,None]
Y_test_hat_log = np.log(Y_test_hat, out=np.zeros_like(Y_test_hat))
Y_pred = np.argmax(Y_test_hat,axis=1)

# Calculating MSE Loss on Test Set  
cost = -(np.trace(np.dot(Y_test_onehot,Y_test_hat_log.T))/(np.shape(Y_test)[0]))
accuracy = (np.sum(Y_pred==Y_test)/np.shape(Y_test)[0])
print("Cost & Accuracy values\n")
print(" Cost = ", cost)
print(" Accuracy = %.2f" % (accuracy*100), "%")
print("\n")