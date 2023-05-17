import torch.nn as nn
import torch


class SimpleCNN(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, num_hidden_layers: int,
                 use_batchnormalization: bool, num_classes: int, kernel_size: int = 3,
                 activation_function: nn.Module = nn.ReLU()):
        super().__init__()

        # Create the convolutional Layers
        hidden_layers = []
        for i in range(num_hidden_layers):
            layer = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size, padding=1, stride = 1)
            hidden_layers.append(layer)

            # Batch normalization
            if use_batchnormalization:
                bn_layer = nn.BatchNorm2d(hidden_channels)
                hidden_layers.append(bn_layer)

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
    network = SimpleCNN(3, 32, 3, True, 10, activation_function=nn.ELU())
    input = torch.randn(1, 3, 64, 64)
    output = network(input)
    print(
        output)  #result: tensor([[ 0.3485, -0.0793, -0.1733, -0.9075, 0.4231, -0.0460, -0.4666, -0.1664,-0.0804, -0.6130]], grad_fn=<AddmmBackward0>)
