import torch.nn as nn
import torch


class SimpleNetwork(nn.Module):
    def __init__(self, input_neurons: int, hidden_neurons: int,
                 output_neurons: int, activation_function: nn.Module = nn.ReLU()):
        super().__init__()
        # input layer
        self.layer_0 = nn.Linear(in_features=input_neurons, out_features=hidden_neurons)
        # the two hidden layers
        self.layer_1 = nn.Linear(in_features=hidden_neurons, out_features=hidden_neurons)
        self.layer_2 = nn.Linear(in_features=hidden_neurons, out_features=hidden_neurons)
        # the output layer
        self.layer_3 = nn.Linear(in_features=hidden_neurons, out_features=output_neurons)
        # define the activation_function
        self.activation_function = activation_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_0(x)
        x = self.activation_function(x)

        x = self.layer_1(x)
        x = self.activation_function(x)

        x = self.layer_2(x)
        x = self.activation_function(x)

        output = self.layer_3(x)
        return output


if __name__ == "__main__":
    torch.random.manual_seed(0)
    simple_network = SimpleNetwork(10, 20, 5)
    input = torch.randn(1, 10)
    output = simple_network(input)
    print(output)  # result:  tensor([[-0.2576, -0.0579, -0.1965, 0.0738, -0.0100]],grad_fn=<AddmmBackward0>)
