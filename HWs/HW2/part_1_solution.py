# Don't erase the template code, except "Your code here" comments.

import torch
import math                      # Pi

""" Task 1 """

def get_rho():
    theta = torch.linspace(-math.pi, math.pi, 1000)
    assert theta.shape == (1000,)

    rho = (1 + 0.9 * torch.cos(8 * theta)) * (1 + 0.1 * torch.cos(24 * theta)) * (0.9 + 0.05 * torch.cos(200 * theta)) * (1 + torch.sin(theta))
    assert torch.is_same_size(rho, theta)
    
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)

    return x, y

""" Task 2 """

def game_of_life_update_torch(alive_map):
    """
    PyTorch version of `game_of_life_update_reference()`.
    
    alive_map:
        `torch.tensor`, ndim == 2, dtype == `torch.int64`
        The game map containing 0s (dead) an 1s (alive).
    """
    #alive_map_numpy = np.random.choice([0, 1], p=(0.5, 0.5), size=(100, 100)).astype(np.int64)
    #alive_map_torch = torch.from_numpy(alive_map_numpy).float.clone()
    
    alive_map = alive_map.float()
    conv_kernel = torch.Tensor([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]])
    neighbors = torch.conv2d(alive_map.unsqueeze(0).unsqueeze(0),
                                 conv_kernel, padding=1).squeeze()
    born = (neighbors == 3) & (alive_map == 0)
    survived = ((neighbors == 2) | (neighbors == 3)) & (alive_map == 1)
    alive_map.copy_(born | survived)
    
""" Task 3 """

# This is a reference layout for encapsulating your neural network. You can add arguments and
# methods if you need to. For example, you may want to add a method `do_gradient_step()` that
# executes one step of an optimization algorithm (SGD / Adadelta / Adam / ...); or you can
# add an extra argument for which you'll find a good value during experiments and set it as
# default to preserve interface (e.g. `def __init__(self, num_hidden_neurons=100):`).
class NeuralNet:
    def __init__(self):
        self.input_size = 28*28
        self.output_size = 10
              
        self.mu = 0
        self.sigma = 0.02
        self.hidden_layer_size = 70 
        self.learning_rate = 2e-2 
        
        self.W1 = torch.zeros(self.input_size, self.hidden_layer_size, requires_grad = True)
        self.b1 = torch.zeros(self.hidden_layer_size, requires_grad = True)
        self.W2 = torch.zeros(self.hidden_layer_size, self.output_size, requires_grad = True)
        self.b2 = torch.zeros(self.output_size, requires_grad = True)
        
        self.W1.data.normal_(self.mu,self.sigma)
        self.W2.data.normal_(self.mu,self.sigma)
        self.b1.data.normal_(self.mu,self.sigma)
        self.b2.data.normal_(self.mu,self.sigma)

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
        return self.forward(self.reshaped(images))

    def reshaped(self, images):
        return  images.reshape([images.size(0), -1])
                            
    def forward(self, images):
        output = images.clone()
        mult_in = torch.matmul(output,self.W1)
        output = torch.sigmoid(torch.add(mult_in,self.b1))
        mult_out = torch.matmul(output,self.W2)
        return torch.softmax(torch.add(mult_out,self.b2), dim = 1)
        
    def do_gradient_step(self):
        l_r = self.learning_rate
        self.W1.data -= l_r * self.W1.grad.data
        self.b1.data -= l_r * self.b1.grad.data
        self.W2.data -= l_r * self.W2.grad.data
        self.b2.data -= l_r * self.b2.grad.data
        
        self.W1.grad = torch.zeros_like(self.W1)
        self.W2.grad = torch.zeros_like(self.W2)
        self.b1.grad = torch.zeros_like(self.b1)
        self.b2.grad = torch.zeros_like(self.b2)
        
    def loss(self, predicted, labels):
        return - torch.sum(labels * torch.log(predicted)) / labels.size(0)

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
    pred_probs = model.predict(images)
    pred_labels = torch.argmax(pred_probs, dim = 1)
    output = float(torch.sum(torch.eq(pred_labels, labels)))
    return output / labels.size(0)

def get_batches(X, y, batch_size):
    n_samples = X.size(0)
    indices = torch.randperm(n_samples)
    batch = torch.split(indices, batch_size)
    for batch_idx in batch:
        yield X[batch_idx], y[batch_idx]

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
    from tqdm import tqdm
    n_epoch = 200
    batch_size = 128
    for i in tqdm(range(n_epoch)):
        for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
            y_batch_oh = torch.eye(10)[y_batch]
            predicted = model.predict(X_batch)
            loss = model.loss(predicted, y_batch_oh)
            loss.backward()
            model.do_gradient_step()
