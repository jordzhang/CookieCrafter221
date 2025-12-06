# cookie prediction model
# predicts cookie attributes from recipe ingredients using linear regression

import numpy as np
import pandas as pd

# settings
DATA_PATH = "data/normalized_cookie_dataset.tsv"
LR = 0.005  # learning rate for gradient descent
EPOCHS = 1000  # number of training iterations

# input columns - these are the recipe ingredients/parameters
FEATURES = ["salt_g", "flour_g", "butter_g", "brown_sugar_g", "white_sugar_g",
            "choc_chips_g", "bake_temp_F", "bake_time_min", "chill_time_min", "eggs_count"]

# output columns - these are the cookie attributes we want to predict
TARGETS = ["thick", "chewy", "crispy", "sweet", "chocolatey", "salty"]


def load_data():
    # read the dataset
    df = pd.read_csv(DATA_PATH, sep="\t")
    X = df[FEATURES].values  # input features
    Y = df[TARGETS].values   # target values
    
    # shuffle the data randomly
    np.random.seed(37)
    idx = np.random.permutation(len(X))
    X, Y = X[idx], Y[idx]
    
    # split into train/val/test (70/15/15)
    n = len(X)
    t1, t2 = int(n*0.7), int(n*0.85)
    
    X_tr, Y_tr = X[:t1], Y[:t1]
    X_val, Y_val = X[t1:t2], Y[t1:t2]
    X_te, Y_te = X[t2:], Y[t2:]
    
    # standardize features to have mean=0 and std=1
    # this helps gradient descent converge faster
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std[std==0] = 1  # avoid divide by zero
    
    X_tr = (X_tr - mean) / std
    X_val = (X_val - mean) / std
    X_te = (X_te - mean) / std
    
    return X_tr, Y_tr, X_val, Y_val, X_te, Y_te, mean, std


class LinearModel:
    # simple linear regression model: Y = X @ W + b
    
    def __init__(self, n_in, n_out):
        # initialize weights randomly (small values)
        self.W = np.random.randn(n_in, n_out) * 0.01
        # initialize biases to zero
        self.b = np.zeros(n_out)
    
    def forward(self, X):
        # compute predictions: Y = X @ W + b
        return X @ self.W + self.b
    
    def mse(self, pred, y):
        # mean squared error loss
        return np.mean((pred - y)**2)
    
    def step(self, X, Y):
        # one step of gradient descent
        
        # forward pass - get predictions
        pred = self.forward(X)
        n = len(X)
        err = pred - Y  # prediction error
        
        # compute gradients (derivatives of MSE loss)
        dW = (2/n) * X.T @ err
        db = (2/n) * err.sum(axis=0)
        
        # update weights and biases
        self.W -= LR * dW
        self.b -= LR * db
        
        return self.mse(pred, Y)


def run_baseline(Y_tr, Y_val, Y_te):
    # baseline model: just predict the average of training data
    # this is the simplest possible prediction - our model should beat this
    avg = Y_tr.mean(axis=0)
    
    # compute MSE for each split
    mse_tr = np.mean((avg - Y_tr)**2)
    mse_val = np.mean((avg - Y_val)**2)
    mse_te = np.mean((avg - Y_te)**2)
    
    print("Baseline (mean predictor):")
    print(f"  train: {mse_tr:.4f}, val: {mse_val:.4f}, test: {mse_te:.4f}")
    return mse_te


def train():
    # load and prepare data
    X_tr, Y_tr, X_val, Y_val, X_te, Y_te, mean, std = load_data()
    print(f"loaded {len(X_tr)} train, {len(X_val)} val, {len(X_te)} test samples")
    
    # run baseline first so we can compare
    baseline = run_baseline(Y_tr, Y_val, Y_te)
    
    # create the model
    model = LinearModel(X_tr.shape[1], Y_tr.shape[1])
    
    # training loop
    print(f"\ntraining for {EPOCHS} epochs...")
    for i in range(EPOCHS):
        # train on training data
        loss = model.step(X_tr, Y_tr)
        
        # check validation loss to monitor overfitting
        val_loss = model.mse(model.forward(X_val), Y_val)
        
        # print progress every 100 epochs
        if (i+1) % 100 == 0:
            print(f"epoch {i+1}: train={loss:.4f} val={val_loss:.4f}")
    
    # final evaluation on test set
    test_loss = model.mse(model.forward(X_te), Y_te)
    print(f"\ntest loss: {test_loss:.4f}")
    print(f"vs baseline: {(baseline-test_loss)/baseline*100:.1f}% better")
    
    return model, mean, std


def generate_recipe(model, target, mean, std):
    # given target attributes, find a recipe that produces them
    # uses optimization to search for the best input features
    from scipy.optimize import minimize
    
    # objective function: how far are predictions from target?
    def obj(x):
        pred = model.forward(x.reshape(1,-1))[0]
        return np.sum((pred - target)**2)
    
    # start from zero (average recipe in standardized space)
    x0 = np.zeros(len(FEATURES))
    
    # keep values within reasonable range (2 std devs)
    bounds = [(-2, 2)] * len(FEATURES)
    
    # run optimization
    res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
    
    # convert back to real units
    recipe = res.x * std + mean
    recipe = np.maximum(recipe, 0)  # no negative ingredients
    return recipe


if __name__ == "__main__":
    model, mean, std = train()

