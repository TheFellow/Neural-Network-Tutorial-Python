#
# Neural Network Trainer Class
#
import numpy as np
import NeuralNetwork as NN

#
# Define the Trainer class
#
class NetworkTrainer:
    """This is a class which trains neural networks"""
    
    #
    # Class Members
    #
    Network = None
    DataSet = []
    MaxIterations = 100000
    
    #
    # Class methods
    #
    def __init__(self, network, dataSet):
        """Set up the trainer class"""
        
        if not isinstance(network, NN.BackPropagationNetwork):
            raise ValueError("Bad parameter {network}")
        
        self.Network = network
        self.DataSet = dataSet
    
    def TrainNetwork(self, min_error=0.1, display=True, **kwags):
        """Train the network until a specific error is reached"""
        for i in range(self.MaxIterations):
            error = bpn.TrainEpoch(self.DataSet[0], self.DataSet[1], **kwags)
            if error <= min_error:
                break
            if error in [np.nan, np.inf]:
                print("An error occurred during training.")
                break
            if display and i % 500 == 0:
                print("Iteration {0:9d} Error: {1:0.6f}".format(i, error))
        
        return i

#
# Run main
#
if __name__ == "__main__":
    
    bitwidth = 3
    
    # Create the network
    bpn = NN.BackPropagationNetwork((bitwidth,2,1), [None, NN.TransferFunctions.tanh, NN.TransferFunctions.linear])
    
    # Create the data set
    input = []
    output = []
    for n in range(2**bitwidth):
        laInput = []
        for digit in reversed(range(bitwidth)):
            laInput.append((n&(2**digit))>>digit)
        input.append(np.array(laInput))
        output.append(np.array([(1 if sum(input[-1]) == 1 else 0)]))
        print("Input: {0} Output: {1}".format(input[-1], output[-1]))
    
    # Train it!
    Input = np.vstack(input)
    Output = np.vstack(output)
    
    trn = NetworkTrainer(bpn, [Input, Output])
    iter = trn.TrainNetwork(1e-4, True, trainingRate = 0.005, momentum = 0.75)
    
    # Run the network on trained data set
    Output = bpn.Run(Input)
    
    for i in range(Input.shape[0]):
        print("Input {0} Output {1}".format(Input[i], np.round(Output[i], 2)))
    
    print(bpn.weights[0])
    print(bpn.weights[1])
    
    
    