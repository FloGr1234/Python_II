import torch
import torch.nn as nn


class SimpleNetwork(nn.Module):

    def __init__(
            self,
            input_neurons: int,
            hidden_neurons: int,
            output_neurons: int,
            activation_function: nn.Module = nn.ReLU()
    ):
        super().__init__()
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.activation_function = activation_function

        self.input_layer = nn.Linear(self.input_neurons, self.hidden_neurons)
        self.hidden_layer_1 = nn.Linear(self.hidden_neurons, self.hidden_neurons)
        self.hidden_layer_2 = nn.Linear(self.hidden_neurons, self.hidden_neurons)
        self.output_layer = nn.Linear(self.hidden_neurons, self.output_neurons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_function(self.input_layer(x))
        x = self.activation_function(self.hidden_layer_1(x))
        x = self.activation_function(self.hidden_layer_2(x))
        x = self.output_layer(x)
        return x


class SimpleCNN1(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, num_hidden_layers: int, num_classes: int, kernel_size: int = 3,
                 activation_function: nn.Module = nn.ReLU()):
        super().__init__()

        # Create the convolutional Layers
        hidden_layers = []
        for i in range(num_hidden_layers):
            layer = nn.Conv2d(in_channels=input_channels*64*64, out_channels=hidden_channels, kernel_size=kernel_size, padding=1, stride=1)
            hidden_layers.append(layer)

            hidden_layers.append(activation_function)
            input_channels = hidden_channels

        # create the full connected outputlayer
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(in_features=hidden_channels * 64 * 64, out_features=num_classes)
        self.hidden_channals = hidden_channels

    def forward(self, x: torch.Tensor):
        x = self.hidden_layers(x)

        x = x.view(x.size(0), self.hidden_channals*64*64)  # flatten feature maps
        output = self.output_layer(x)
        return output


if __name__ == "__main__":
    torch.random.manual_seed(0)
    network = SimpleCNN1(3, 32, 3, True, 10, activation_function=nn.ELU())
    input = torch.randn(1, 3, 64, 64)
    output = network(input)
    print(
        output)  #result: tensor([[ 0.3485, -0.0793, -0.1733, -0.9075, 0.4231, -0.0460, -0.4666, -0.1664,-0.0804, -0.6130]], grad_fn=<AddmmBackward0>)


class SimpleCNN(torch.nn.Module):

    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        """Simple CNN with ``n_hidden_layers``, ``n_kernels`` and
        ``kernel_size`` as hyperparameters."""
        super().__init__()

        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)

        self.output_layer = torch.nn.Conv2d(
            in_channels=n_in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, x):
        """Apply CNN to input ``x`` of shape ``(N, n_channels, X, Y)``, where
        ``N=n_samples`` and ``X``, ``Y`` are spatial dimensions."""
        # Apply hidden layers: (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        cnn_out = self.hidden_layers(x)
        # Apply output layer: (N, n_kernels, X, Y) -> (N, 1, X, Y)
        predictions = self.output_layer(cnn_out)
        return predictions
