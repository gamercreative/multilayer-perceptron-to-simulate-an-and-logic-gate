#include <iostream>
#include <cmath>   // For mathematical functions like exp()
#include <time.h>  // For seeding the random number generator
#include <vector>  // To use std::vector for dynamic arrays
using namespace std;

// Sigmoid activation function to squash input to range (0, 1)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function, used for backpropagation
double sigmoid_derv(double x) {
    return x * (1.0 - x);
}

// Define a neural network class
class neural_network {
public:
    // Constructor: initializes weights and biases randomly
    neural_network() {
        srand(time(0)); // Seed the random number generator

        // Initialize weights for the input-to-hidden layer
        weights_input_hidden = { 
            {random_weight(), random_weight()}, 
            {random_weight(), random_weight()} 
        };
        
        // Initialize weights for the hidden-to-output layer
        weights_hidden_output = {random_weight(), random_weight()};
    }

    // Generate random weight between 0 and 1
    double random_weight() {
        return static_cast<double>(rand()) / RAND_MAX;
    }

    // Forward pass: calculate the output given two inputs
    double ff(double in1, double in2) {
        // Calculate hidden layer activations using the sigmoid function
        vector<double> hidden_layer(2);
        hidden_layer[0] = sigmoid(in1 * weights_input_hidden[0][0] + 
                                  in2 * weights_input_hidden[1][0] + 
                                  bais_hidden[0]);

        hidden_layer[1] = sigmoid(in1 * weights_input_hidden[0][1] + 
                                  in2 * weights_input_hidden[1][1] + 
                                  bais_hidden[1]);

        // Calculate the final output using hidden layer activations
        double output = sigmoid(hidden_layer[0] * weights_hidden_output[0] + 
                                hidden_layer[1] * weights_hidden_output[1] + 
                                bais_output);
        return output;  // Return the output value
    }

    // Backpropagation: adjust weights to minimize error
    void bp(double in1, double in2, double target) {
        // Calculate hidden layer activations (same as in forward pass)
        vector<double> hidden_layer(2);
        hidden_layer[0] = sigmoid(in1 * weights_input_hidden[0][0] + 
                                  in2 * weights_input_hidden[1][0] + 
                                  bais_hidden[0]);

        hidden_layer[1] = sigmoid(in1 * weights_input_hidden[0][1] + 
                                  in2 * weights_input_hidden[1][1] + 
                                  bais_hidden[1]);

        // Calculate output based on current weights
        double output = sigmoid(hidden_layer[0] * weights_hidden_output[0] + 
                                hidden_layer[1] * weights_hidden_output[1] + 
                                bais_output);

        // Compute the output error and its delta (for gradient)
        double output_error = target - output;
        double output_error_delta = output_error * sigmoid_derv(output);

        // Calculate errors for the hidden layer nodes
        vector<double> error_hidden_layer(2);
        vector<double> error_hidden_layer_delta(2);

        error_hidden_layer[0] = output_error_delta * weights_hidden_output[0];
        error_hidden_layer[1] = output_error_delta * weights_hidden_output[1];

        // Calculate the delta for the hidden layer nodes
        error_hidden_layer_delta[0] = error_hidden_layer[0] * sigmoid_derv(hidden_layer[0]);
        error_hidden_layer_delta[1] = error_hidden_layer[1] * sigmoid_derv(hidden_layer[1]);

        // Update weights for hidden-to-output connections
        weights_hidden_output[0] += hidden_layer[0] * output_error_delta * learning_rate;
        weights_hidden_output[1] += hidden_layer[1] * output_error_delta * learning_rate;

        // Update weights for input-to-hidden connections
        weights_input_hidden[0][0] += error_hidden_layer_delta[0] * in1 * learning_rate;
        weights_input_hidden[1][0] += error_hidden_layer_delta[0] * in1 * learning_rate;
        weights_input_hidden[0][1] += error_hidden_layer_delta[1] * in2 * learning_rate;
        weights_input_hidden[1][1] += error_hidden_layer_delta[1] * in2 * learning_rate;

        // Update biases for the hidden and output layers
        bais_hidden[0] += error_hidden_layer_delta[0] * learning_rate;
        bais_hidden[1] += error_hidden_layer_delta[1] * learning_rate;
        bais_output += output_error_delta * learning_rate;
    }

private:
    // Hidden layer activations (not used directly in this version)
    vector<double> hidden_layer;

    // Weights between input and hidden layers
    vector<vector<double>> weights_input_hidden;

    // Weights between hidden and output layers
    vector<double> weights_hidden_output;

    // Biases for hidden layer nodes
    vector<double> bais_hidden = {0.1, 0.1};

    // Bias for the output node
    double bais_output = 0.1;

    // Learning rate for weight updates
    double learning_rate = 0.01;
};

int main() {
    // Create an instance of the neural network
    neural_network b3yte;

    // Dataset representing OR gate behavior: {inputs} -> target output
    vector<pair<pair<double, double>, double>> dataset = {
        {{0, 0}, 0}, {{0, 1}, 1}, {{1, 0}, 1}, {{1, 1}, 1}
    };

    // Train the network over 1 million iterations
    for (int i = 0; i < 1000000; i++) {
        for (auto data : dataset) {
            b3yte.bp(data.first.first, data.first.second, data.second);
        }
    }

    // Display the network's output for all possible inputs
    cout << "Output for (0,0): " << b3yte.ff(0, 0) << endl;
    cout << "Output for (0,1): " << b3yte.ff(0, 1) << endl;
    cout << "Output for (1,0): " << b3yte.ff(1, 0) << endl;
    cout << "Output for (1,1): " << b3yte.ff(1, 1) << endl;

    return 0;
}
