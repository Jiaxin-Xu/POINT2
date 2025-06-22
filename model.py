import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

import tensorflow_probability as tfp

#########################
# BNN
#########################

tfpl = tfp.layers
tfd = tfp.distributions

def build_bayesian_model(input_shape, hidden_units, output_shape):
    # Define the prior and posterior for the weights and biases
    def prior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfpl.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(
                loc=tf.zeros(n), scale_diag=tf.ones(n)))
        ])

    def posterior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
            tfpl.IndependentNormal(n)
        ])

    # Input layer: Ensure it's defined correctly
    print("Input shape:", input_shape)
    print("Expected shape for tf.keras.Input:", (input_shape[0],))

    inputs = tf.keras.Input(shape=(input_shape[0],))

    # Dense Variational layer
    hidden = tfpl.DenseVariational(units=hidden_units,
                                   make_prior_fn=prior,
                                   make_posterior_fn=posterior,
                                   kl_weight=1/input_shape[0],
                                   activation='relu')(inputs)

    # Output layer
    outputs = tfpl.DenseVariational(units=output_shape,
                                    make_prior_fn=prior,
                                    make_posterior_fn=posterior,
                                    kl_weight=1/input_shape[0],
                                    activation='linear')(hidden)

    # Build model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def compile_and_train_BNN(model, train_data, train_labels, num_epochs):
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(train_data, train_labels, epochs=num_epochs, verbose=1)
    return model

def bnn_predict_with_uncertainty(model, test_data, n_iter=100, quantiles=[5, 95]):
    # Perform stochastic forward passes
    pred_list = [model(test_data, training=True) for _ in range(n_iter)]
    pred_dist = tf.stack(pred_list)
    
    # Convert predictions to numpy for quantile calculations
    predictions = pred_dist.numpy()
    
    mean_pred = np.mean(predictions, axis=0)
    # Ensure mean_predictions is a 1D array
    if mean_pred.ndim > 1:
        mean_pred = mean_pred.squeeze()

    std_dev_pred = np.std(predictions, axis=0)
    # Ensure std_deviation is a 1D array
    if std_dev_pred.ndim > 1:
        std_dev_pred = std_dev_pred.squeeze()

    quantile_predictions = [np.percentile(predictions, q, axis=0) for q in quantiles]
    # Ensure each quantile prediction is a 1D array
    quantile_predictions = [q.squeeze() for q in quantile_predictions]

    return mean_pred, std_dev_pred, quantile_predictions



#########################
# MLP-DropOut
#########################
# Define a custom Dropout class that remains active during inference
class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

# Function to create the MLP model with MC Dropout layers
def create_mc_dropout_mlp_model(input_dim, dropout_rate=0.2):
    model = Sequential([
        Dense(512, input_dim=input_dim, activation='relu'),
        MCDropout(dropout_rate),
        Dense(128, activation='relu'),
        MCDropout(dropout_rate),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train the MLP model with a portion of the training data reserved for validation
def train_mlp_model(X_train, y_train, validation_split=0.1):
    model = create_mc_dropout_mlp_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=validation_split)
    return model

# Monte Carlo prediction function to estimate uncertainties
def mlp_predict_with_uncertainty(model, X, n_iter=100, quantiles=[5, 95]):
    predictions = [model.predict(X) for _ in range(n_iter)]
    predictions = np.array(predictions)

    mean_predictions = predictions.mean(axis=0)
    # Ensure mean_predictions is a 1D array
    if mean_predictions.ndim > 1:
        mean_predictions = mean_predictions.squeeze()

    std_deviation = predictions.std(axis=0)
    # Ensure std_deviation is a 1D array
    if std_deviation.ndim > 1:
        std_deviation = std_deviation.squeeze()

    quantile_predictions = [np.percentile(predictions, q, axis=0) for q in quantiles]
    # Ensure each quantile prediction is a 1D array
    quantile_predictions = [q.squeeze() for q in quantile_predictions]

    return mean_predictions, std_deviation, quantile_predictions

# # Example data generation
# np.random.seed(0)
# X = np.random.randn(1000, 10)
# y = X.sum(axis=1) + np.random.randn(1000) * 0.5  # Some function of X plus noise

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the model
# model = train_mlp_model(X_train, y_train)

# # Predict with uncertainty
# mean_predictions, std_deviation, quantile_predictions = mlp_predict_with_uncertainty(model, X_test, n_iter=100)

# print("Mean Predictions:", mean_predictions[:5])
# print("Standard Deviations:", std_deviation[:5])
# print("5th and 95th Quantiles:", quantile_predictions[5][:5], quantile_predictions[95][:5])
