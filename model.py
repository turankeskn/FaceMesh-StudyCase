import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Function to preprocess data, train a random forest regressor, and evaluate using k-fold cross-validation
def train_and_evaluate_skincare_model(X, y):
    scores = []
    kfold = KFold()
    
    # Flatten the landmarks into a 2D array
    X_flat = X.reshape((X.shape[0], -1))

    for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
        X_train, y_train = X_flat[train_index], y[train_index]
        X_test, y_test = X_flat[test_index], y[test_index]
        
        # Train a random forest regressor
        model = RandomForestRegressor() # We can also hyperparameters tuning with gridsearchcv
        model.fit(X_train, y_train)
        
        # Evaluate the model on the test set
        pred = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, pred))
        
        scores.append(score)
        print(f'Fold {i+1}, RMSE: {score}')
        
    print('{0} Average RMSE: {1:.3f}'.format(model.__class__.__name__, np.mean(scores)))  
    return np.mean(scores), model

# I couldn't find a real data set for this case because of that I create synthetic data (replace with real dataset)
def generate_synthetic_data(num_samples):
    X = np.random.rand(num_samples, 468, 3)  # 468 landmarks with (x, y, z) coordinates
    y = np.random.rand(num_samples)  # Random labels (replace with actual skincare labels)
    print(X, y)
    return X, y

