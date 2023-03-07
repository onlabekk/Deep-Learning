# Don't erase the template code, except "Your code here" comments.

import torch
import math                      # Pi

""" Task 1 """

def get_rho():
    # (1) Your code here; theta = ...
    assert theta.shape == (1000,)

    # (2) Your code here; rho = ...
    assert torch.is_same_size(rho, theta)
    
    # (3) Your code here; x = ...
    # (3) Your code here; y = ...

    return x, y

""" Task 2 """

def game_of_life_update_torch(alive_map):
    """
    PyTorch version of `game_of_life_update_reference()`.
    
    alive_map:
        `torch.tensor`, ndim == 2, dtype == `torch.int64`
        The game map containing 0s (dead) an 1s (alive).
    """
    # Your code here

""" Task 3 """

# This is a reference layout for encapsulating your neural network. You can add arguments and
# methods if you need to. For example, you may want to add a method `do_gradient_step()` that
# executes one step of an optimization algorithm (SGD / Adadelta / Adam / ...); or you can
# add an extra argument for which you'll find a good value during experiments and set it as
# default to preserve interface (e.g. `def __init__(self, num_hidden_neurons=100):`).
class NeuralNet:
    def __init__(self):
        # Your code here

    def predict(self, images):
        """
        images:
            `torch.tensor`, shape == `batch_size x height x width`, dtype == `torch.float32`
            A minibatch of images -- the input to the neural net.
        
        return:
        prediction:
            `torch.tensor`, shape == `batch_size x 10`, dtype == `torch.float32`
            The scores of each input image to belong to each of the dataset classes.
            Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
            belong to `j`-th class.
            These scores can be 0..1 probabilities, but for better numerical stability
            they can also be raw class scores after the last (usually linear) layer,
            i.e. BEFORE softmax.
        """
        # Your code here

    # Your code here

def accuracy(model, images, labels):
    """
    Use `NeuralNet.predict` here.
    
    model:
        `NeuralNet`
    images:
        `torch.tensor`, shape == `N x height x width`, dtype == `torch.float32`
    labels:
        `torch.tensor`, shape == `N`, dtype == `torch.int64`
        Class indices for each sample in `images`.
    
    return:
    value:
        `float`
        The fraction of samples from `images` correctly classified by `model`.
        `0 <= value <= 1`.
    """
    # Your code here

def train_on_notmnist(model, X_train, y_train, X_val, y_val):
    """
    Update `model`'s weights so that its accuracy on `X_val` is >=82%.
    `X_val`, `y_val` are provided for convenience and aren't required to be used.
    
    model:
        `NeuralNet`
    X_train:
        `torch.tensor`, shape == `N x height x width`, dtype == `torch.float32`
    y_train:
        `torch.tensor`, shape == `N`, dtype == `torch.int64`
        Class indices for each sample in `X_train`.
    X_val, y_val:
        Same as above, possibly with a different length.
    """
    # Your code here
