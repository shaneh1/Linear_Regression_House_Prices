import numpy as np
import scipy.optimize

def linear_model(W, X):
    """Given intercept and coefficients W, and single row of the training data X,
    calculate yhat = w0 + w1x1 + w2x2 + ..."""
    n = W.shape[0] - 1
    return W[0] + sum((W[i+1] * X[i]) for i in range(n))

def dist(x, y):
    return np.linalg.norm(x - y)

def objective(W,X,y):
    """Given intercept and coefficients W, and the entire training data
    (X, y), calculate the value of the objective."""
    return sum((y[i] - linear_model(W,X[i]))**2 for i in range(len(X)))
    # sum of square errors between true values and equation of line

def derivative_1(W, X, y):
    """Given intercept and coefficients W, and a single point from the
    training data (X, y), calculate the value of the derivative of the
    objective, for use in SGD. Hint: calculate yhat first."""
    der = np.zeros_like(W)
    yhat = linear_model(W,X) #calculate initial prediction
    der[0] = 2 * (yhat - y) #derivative of first element (intercept)
    for i in range(n):
        der[i + 1] = 2 * (yhat - y) * X[i] #derivative of all other elements (coefficients)
    return der

def RMSE(a, b):
    """Root mean square error between two vectors a and b."""
    return np.sqrt((sum(a[i]-b[i])**2) for i in range(len(a)))
    #rmse between each pair of points in two vector.

def parallel_shuffle(a, b):
    """Shuffle two arrays of equal length, so that corresponding items ai
    and bi are still corresponding items aj and bj."""
    p = np.random.permutation(len(a))
    return a[p], b[p]

def SGD_LR(W0, X, y,
           alpha= 0.01, gamma=0.0,
           shuffle = False,
           tol=10**-7, maxits=1000):
    """Run stochastic gradient descent to solve a linear regression problem.
    Use a learning rate alpha and momentum."""

    if shuffle:
        X, y = parallel_shuffle(X, y)

    W = W0 # initial point
    v = np.zeros_like(W0) # initial value of the step, for use by momentum

    its = 0
    while its < maxits:
        idx = its % len(X) # make a "batch" consisting of just one point
        X_batch = X[idx]
        y_batch = y[idx]
        slope = derivative_1(W, X_batch, y_batch) #calculate the slope at that one point
        v = [(gamma * vi + slope) for vi, slope in zip(v, slope)] #velocity, influenced by gamma and slope.
        new_w = np.array([Wi - alpha * vi for Wi, vi in zip(W,v)]) # find new w value using learning rate and momentum.
        if dist(new_w, W) <= tol: #if step size is lower than tolerance, satisfied optimum has been reached.
            return new_w
        its += 1 #
        W = new_w #go in the direction of that step. Repeat until maxits exceeded or tol reached.
    return W

def read_portland_data():
    d = np.genfromtxt("portland_house_prices_standardised.csv", delimiter=",")
    X, y = d[:, 0:2], d[:, 2]
    return X, y

def make_test_data():
    X = np.array([ [1], [2], [3], [4], [5], [6], [7], [8], [9.]  ])
    y = np.array( [2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1] )
    return X, y

#X,y = make_test_data()
X, y = read_portland_data()

n = X.shape[1]
x0 = np.zeros(n+1)

SGD_LR = SGD_LR(x0, X, y, alpha = 0.001, gamma = 0.5, shuffle = False, tol=10**-7, maxits = 5000)

sse = objective(SGD_LR, X, y) #create new variable calling sse function, using SGD function
print("Optimum SSE value: {}".format(sse)) #print result
print("Optimum Coefficients of line: {}".format(SGD_LR))
print("Optimum SSE value (Nelder-Mead Method): ")
#Compare scipy.optimize.minimize results
res = scipy.optimize.minimize (objective, x0, args = (X,y), method = 'Nelder-Mead', tol=10**-7, options={"disp": True})
print("Optimum Coefficients of line (Nelder-Mead Method):")
print(res.x)
