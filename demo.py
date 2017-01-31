from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a three layer network:
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.

        # First Layer, 4 neurons with 3 input connections and 4 output connections.
        self.synaptic_weights1 = 2 * random.random((3, 4)) - 1
        # Second Layer, 4 neurons with 4 input connections and 4 output connections.
        self.synaptic_weights2 = 2 * random.random((4, 4)) - 1
        # Third Layer, 4 neurons with 1 input connections and 4 output connections.
        self.synaptic_weights3 = 2 * random.random((4, 1)) - 1



    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output1, output2, output3 = self.think(training_set_inputs)

            # Calculate the error of the last layer (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output3
            delta3 = error * self.__sigmoid_derivative(output3)

            # Calculate the error of the second layer (The contribute of layer 2 to the layer 3 error as a function of
            # the weights)
            error2 = dot(delta3, self.synaptic_weights3.T)
            delta2 = error2 * self.__sigmoid_derivative(output2)

            # Calculate the error of the first layer (The contribute of layer 1 to the layer 2 error as a function of
            # the weights)
            error1 = dot(delta2, self.synaptic_weights2.T)
            delta1 = error1 * self.__sigmoid_derivative(output1)


            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            # Calculate how much to adjust the weights by
            adjustment1 = dot(training_set_inputs.T, delta1)
            adjustment2 = dot(output1.T, delta2)
            adjustment3 = dot(output2.T, delta3)


            # Adjust the weights.
            self.synaptic_weights1 += adjustment1
            self.synaptic_weights2 += adjustment2
            self.synaptic_weights3 += adjustment3



    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network.
        # output of the first layer
        out1 = self.__sigmoid(dot(inputs, self.synaptic_weights1))
        # takes the output of the first layer as input and compute the second output
        out2 = self.__sigmoid(dot(out1, self.synaptic_weights2))
        # takes the output of the first layer as input and compute the final output
        out3 = self.__sigmoid(dot(out2, self.synaptic_weights3))

        return out1, out2, out3


if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights1)
    print(neural_network.synaptic_weights2)
    print(neural_network.synaptic_weights3)



    # See https://medium.com/technology-invention-and-more/how-to-build-a-multi-layered-neural-network-in-python-53ec3d1d326a#.sb6sv57iu
    # and  https://iamtrask.github.io/2015/07/27/python-network-part2/
    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T


    # Train the neural network using a training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights1)
    print(neural_network.synaptic_weights2)
    print(neural_network.synaptic_weights3)


    # Test the neural network with a new situation.
    print("Considering a new situation [1, 1, 0] -> ?: ")
    _, _, output = neural_network.think(array([1, 1, 0]))
    print(output)
