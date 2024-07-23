import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to preprocess data, train a ResNet, and evaluate using a single train-test split
def train_and_evaluate_skincare_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define a simple ResNet model
    inputs = tf.keras.Input(shape=(468, 3, 1))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Residual block
    res = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    res = tf.keras.layers.BatchNormalization()(res)
    res = tf.keras.layers.Activation('relu')(res)
    res = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(res)
    res = tf.keras.layers.BatchNormalization()(res)
    
    x = tf.keras.layers.add([x, res])
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    # Train the ResNet model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Evaluate the model on the test set
    pred = model.predict(X_test).flatten()
    score = np.sqrt(mean_squared_error(y_test, pred))
    
    print(f'ResNet RMSE: {score:.3f}')
    return score, model

# Generating synthetic data
def generate_synthetic_data(num_samples):
    X = np.random.rand(num_samples, 468, 3, 1)  # 468 landmarks with (x, y, z) coordinates reshaped for CNN
    y = np.random.rand(num_samples)  # Random labels (replace with actual skincare labels)
    return X, y

# Example usage
X, y = generate_synthetic_data(1000)
rmse_resnet, resnet_model = train_and_evaluate_skincare_model(X, y)