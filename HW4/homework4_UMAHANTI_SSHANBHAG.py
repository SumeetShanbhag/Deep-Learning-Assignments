#!/usr/bin/env python
# coding: utf-8

# In[6]:

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import warnings
warnings.filterwarnings('ignore')

NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = NUM_HIDDEN_LAYERS * [ 64 ]
NUM_OUTPUT = 10

# In[7]:


def unpack (weights):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN[0] 
    W = weights[start:end]
    Ws.append(W)

    # Unpack the weight matrices as vectors
    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i]*NUM_HIDDEN[i+1]
        W = weights[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN[-1]*NUM_OUTPUT
    W = weights[start:end]
    Ws.append(W)

    
    #print('Ws',np.shape(Ws[0]))     
          # Reshape the weight "vectors" into proper matrices
    Ws[0] = Ws[0].reshape(NUM_HIDDEN[0], NUM_INPUT)
    # print('Ws',np.shape(Ws[0]))
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN[i], NUM_HIDDEN[i-1])
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN[-1])

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN[0]
    b = weights[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i+1]
        b = weights[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weights[start:end]
    bs.append(b)

    return Ws, bs


# In[8]:


def fCE(X, Y, weights, alpha):
    Ws, bs = unpack(weights)
    n = len(Y)
    hs = []
    Zs = []
    h = X
    hold = 0
    for i in range(NUM_HIDDEN_LAYERS+1):
        z = np.dot(Ws[i], h.T).T + bs[i]
        Zs.append(z)
        h = relu(z)
        hs.append(h)
    y_hat = softmax(z)
    Fce = (-1/n) * np.sum(Y * np.log(y_hat))
    for i in range(NUM_HIDDEN_LAYERS+1):
        hold += np.sum(Ws[i]**2)
    regularization = Fce + (alpha/2)*hold
    return regularization, Zs, hs, y_hat
    

# In[9]:


def onehot(y):    
    Y_encode = np.zeros((y.size, y.max() + 1))
    Y_encode[np.arange(y.size), y] = 1
    return Y_encode        


# In[10]:


def gradCE(X, Y, weights, alpha):
    _, Zs, hs, y_hat = fCE(X, Y, weights, alpha)
    Ws, bs = unpack(weights)
    n = np.shape(Y)[0]
    dw, db = [], []
    g = (y_hat - Y) / n
    backprop_list = list(reversed(range(len(hs))))
    for i in backprop_list:
        db.append(np.sum(g.T, axis=1))
        if i == 0:
            dv_w = np.dot(g.T, X)
        else:
            dv_w = np.dot(g.T, hs[i - 1])
            g = np.dot(g, Ws[i])
            g = g * relu_pr(Zs[i - 1])
        dw.append(dv_w)

    dw = np.flip(np.asarray(dw))
    db = np.flip(np.asarray(db))

    dw = [dwi + alpha * Wi for dwi, Wi in zip(dw, Ws)]
    allGradientsAsVector = np.hstack([d_w.flatten() for d_w in dw] + [d_b.flatten() for d_b in db])
    return allGradientsAsVector


# In[11]:


def relu(Z):           
    relu = np.maximum(0,Z)
    return relu


# In[12]:


def relu_pr(Z):        
    relu_pr = np.where(Z > 0, 1, 0)
    return relu_pr


# In[13]:


def softmax(Z):        
    p = np.exp(Z)/np.sum(np.exp(Z), axis=1, keepdims=True)
    return p


# In[14]:


def accuracy(X,Y, weights,alpha):
    _,_,_, y_hat = fCE (X, Y, weights,alpha)
    # Ws,bs = unpack(weights)
    y_hat = np.argmax(y_hat, axis=1)
    label = np.argmax(Y, axis=1)
    acc = (np.mean(y_hat==label))*100
    return acc



# In[15]:


def show_W0 (W):
   Ws,bs = unpack(W)
   W = Ws[0]
   n = int(NUM_HIDDEN[0] ** 0.5)
   plt.imshow(np.vstack([
       np.hstack([ np.pad(np.reshape(W[idx1*n + idx2,:], [ 28, 28 ]), 2, mode='constant') for idx2 in range(n) ]) for idx1 in range(n)
   ]), cmap='gray'), plt.show()


# In[16]:


def SGD(allGradientsAsVector, lr, weights, num_epochs):
    dw, db = unpack(allGradientsAsVector)
    Ws, bs = unpack(weights)
    lr_new = learning_rate_decay(lr, 0.001, num_epochs)
    w_updated = [w - lr_new * d_w for w, d_w in zip(Ws, dw)]
    b_updated = [b - lr_new * d_b for b, d_b in zip(bs, db)]
    updated_grad = np.hstack([d_w.flatten() for d_w in w_updated] + [d_b.flatten() for d_b in b_updated])
    return updated_grad


# In[17]:


def learning_rate_decay(lr, decay_rate, num_epoch):
    return lr / (1 + decay_rate * num_epoch)


# In[18]:


def train (trainX, trainY, testX, testY):
    # TODO: implement me
    epochs = [5,10] #100
    alphas = [0.0001,0.05,0.001] #0.0001
    batch_sizes = [8,16,32] #16
    hidden_layers = [3,4,5] #3
    hidden_units = [30,40,50] #50 
    learning_rates = [0.005,0.0001,0.004] #0.005

    val_fce_star = np.float16('inf')
    hidden_layer_star = None
    hidden_unit_star = None
    alpha_star = None
    epoch_star = None
    batch_size_star = None
    learning_rate_star = None

    
    for layer in hidden_layers:
        for unit in hidden_units:
            global NUM_HIDDEN_LAYERS
            global NUM_HIDDEN
            NUM_HIDDEN_LAYERS = layer
            NUM_HIDDEN = NUM_HIDDEN_LAYERS * [unit]
            initial = initWeightsAndBiases()
            for alpha in alphas:
                for lr in learning_rates:
                    for num_epochs in epochs:
                        for batch_size in batch_sizes:
                            for epoch in range(num_epochs):
                                for i in range(0, len(trainY), batch_size):
                                    
                                    X_batch = trainX[i:i + batch_size, :]
                                    y_batch = trainY[i:i + batch_size]

                                    allgradasvector = gradCE(X_batch,y_batch,initial,alpha)   #unpacked
                    
                                    allupdated_grad = SGD(allgradasvector , lr , initial , num_epochs)
                                    initial = allupdated_grad
                                                      
                            val_FCE,_,_,_ = fCE(X_val, y_val, initial,alpha)          # validation MSE
                                
                            if (val_FCE < val_fce_star):
                                val_fce_star = val_FCE
                                val_acc = accuracy(X_val,y_val,initial,alpha)
                                print('val_acc = ', val_acc)
                                print('best_val_fce = ', val_fce_star)
                                alpha_star = alpha
                                epoch_star = num_epochs
                                batch_size_star = batch_size
                                best_weights = initial
                                learning_rate_star = lr
                                hidden_layer_star = layer
                                hidden_unit_star = unit
    print("The best hyper parameters are:")
    print("val fce = ", val_fce_star)
    print("alpha = ", alpha_star)
    print("hidden_layer = ", hidden_layer_star)
    print("hidden_unit = ", hidden_unit_star)
    print("num_epochs = ", epoch_star)
    print("batch_size = ", batch_size_star)
    print("learning_rate = ", learning_rate_star)

    
    NUM_HIDDEN_LAYERS = hidden_layer_star
    NUM_HIDDEN =  NUM_HIDDEN_LAYERS * [hidden_unit_star]


    test_fce,_,_,_ = fCE(testX, testY,best_weights,alpha)  
    acc_test = accuracy(testX, testY,best_weights,alpha)
    print("Regularized ce for Testing data = ", test_fce)
    print("Test accuracy = ", acc_test)
    print('Best hidden unit = ',hidden_unit_star)

    return best_weights, acc_test


# In[19]:


def initWeightsAndBiases ():
    Ws = []
    bs = []

    # Strategy:
    # Sample each weight from a 0-mean Gaussian with std.dev. of 1/sqrt(numInputs).
    # Initialize biases to small positive number (0.01).

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_HIDDEN[0], NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN[0])
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN[i], NUM_HIDDEN[i+1]))/NUM_HIDDEN[i]**0.5) - 1./NUM_HIDDEN[i]**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN[i+1])
        bs.append(b)

    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN[-1]))/NUM_HIDDEN[-1]**0.5) - 1./NUM_HIDDEN[-1]**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    in_weights = np.hstack([w.flatten() for w in Ws] + [b.flatten() for b in bs])
    #print(in_weights)
    return in_weights


# In[ ]:


if __name__ == "__main__":

    trainX = np.load('fashion_mnist_train_images.npy')/255
    trainY = onehot(np.load('fashion_mnist_train_labels.npy'))
    testX = np.load('fashion_mnist_test_images.npy')/255
    testY = onehot(np.load('fashion_mnist_test_labels.npy'))

    num_samples = trainX.shape[0]
    num_val = int(num_samples * 0.2)
    X_val = trainX[:num_val, :]
    y_val = trainY[:num_val]
    X_train = trainX[num_val:, :]
    y_train = trainY[num_val:]


#######3


    # # Load training data.
    # # Recommendation: divide the pixels by 255 (so that their range is [0-1]), and then subtract
    # # 0.5 (so that the range is [-0.5,+0.5]).

    Ws, bs = unpack(initWeightsAndBiases())

    # # "Pack" all the weight matrices and bias vectors into long one parameter "vector".
    weights = np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])
    
    weights = initWeightsAndBiases()

    # # On just the first 5 training examlpes, do numeric gradient check.
    # # Use just the first return value ([0]) of fCE, which is the cross-entropy.
    # # The lambda expression is used so that, from the perspective of
    # # check_grad and approx_fprime, the only parameter to fCE is the weights
    # # # themselves (not the training data).110
    
    print('---check grad---')
    print(scipy.optimize.check_grad(lambda weights_: fCE(X_train[0:5,:], y_train[0:5], weights_,0)[0],                                      lambda weights_: gradCE(X_train[0:5,:], y_train[0:5], weights_,0),                                     weights))
    print('---approx prime---')
    print(scipy.optimize.approx_fprime(weights, lambda weights_: fCE(X_train[0:5,:], y_train[0:5,:], weights_,0)[0], 1e-6))


    acc_test = train (trainX, trainY, testX, testY)
    print("Accuracy for Testing data = ", acc_test)
    
    weights,_ = train(X_train, y_train, testX, testY)
    show_W0(weights)


# %%
