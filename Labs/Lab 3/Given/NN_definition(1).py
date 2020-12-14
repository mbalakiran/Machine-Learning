# inputs = features
# outputs =  target values
# inputLayerNeurons = number of neurons in the input layer
# hiddenLayerNeurons = number of neurons in the hidden layer
# outputLayerNeurons = number of neurons in the output layer
# epochs = number of epochs
# lr = learning rate

# ----------------------------------- #

# Standard library import
import random

# Third-party library import
import numpy as np

# Local import
# your functions to import: sigmoid etc

# ------------------------------------ #


def Net(
    inputs,
    expected_output,
    inputLayerNeurons,
    hiddenLayerNeurons,
    outputLayerNeurons,
    epochs,
    lr,
):

    # Random weights and bias initialization
    hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
    hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
    output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
    output_bias = np.random.uniform(size=(1, outputLayerNeurons))

    print("Initializing Network Parameters:")
    print()
    print("Initial hidden weights: ", end="")
    print(*hidden_weights)
    print("Initial hidden biases: ", end="")
    print(*hidden_bias)
    print("Initial output weights: ", end="")
    print(*output_weights)
    print("Initial output biases: ", end="")
    print(*output_bias)

    # Training algorithm
    for _ in range(epochs):
        # Forward Propagation
        hidden_layer_activation = np.dot(inputs, hidden_weights)
        hidden_layer_activation += hidden_bias
        hidden_layer_output = sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, output_weights)
        output_layer_activation += output_bias
        predicted_output = sigmoid(output_layer_activation)

        # Backpropagation
        error = expected_output - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)

        error_hidden_layer = d_predicted_output.dot(output_weights.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Updating Weights and Biases
        output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
        output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
        hidden_weights += inputs.T.dot(d_hidden_layer) * lr
        hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

    return hidden_weights, hidden_bias, output_weights, output_bias, predicted_output

