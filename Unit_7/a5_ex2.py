import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


def training_loop(
        network: torch.nn.Module,
        train_data: torch.utils.data.Dataset,
        eval_data: torch.utils.data.Dataset,
        num_epochs: int,
        show_progress: bool = False
) -> tuple[list, list]:

    # set the optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)  # SGD

    # Define the loss-function
    loss_function = nn.MSELoss()

    # create the DataLoader
    train_loader = DataLoader(train_data, batch_size=20, shuffle=False)
    eval_loader = DataLoader(eval_data, batch_size=20, shuffle=True)

    # Initialize the loss lists for the output
    evaluation_losses = []
    epoch_losses = []
    min_eval_loss = float('inf')

    for _ in range(num_epochs):
        errs_train = []
        sum_losses_per_epoch = 0

        # show the prozess
        if show_progress:
            train_loader = tqdm(train_loader, desc=f"training")

        # ---------------------------------
        # Training
        # ---------------------------------
        network.train()  # Set the network to training mode
        for inputs, target in train_loader:
            # set gradients back to zero
            optimizer.zero_grad()
            # calculate output of the network
            out = network(inputs)[:, 0]
            # calculate the losses
            loss = loss_function(target, out)
            # calculate gradient
            loss.backward()
            # update the weights
            optimizer.step()
            # save the losses of the minibach
            errs_train.append(loss)
            sum_losses_per_epoch += loss

        # calculate the mean loss of one training epoch
        mean_err = torch.mean(torch.tensor(errs_train)).item()
        epoch_losses.append(mean_err)

        # ---------------------------------
        # Evaluation
        # ---------------------------------
        network.eval()  # set network to evaluation mode
        errs_eval = []

        with torch.no_grad():
            for inputs, target in eval_loader:
                out = network(inputs)[:, 0]
                loss_function = nn.MSELoss()
                loss = loss_function(out, target)
                errs_eval.append(loss)

        mean_err = torch.mean(torch.tensor(errs_eval)).item()
        evaluation_losses.append(mean_err)

        # early stop
        if mean_err < min_eval_loss:
            min_eval_loss = mean_err
            num_no_change_in_min_loss = 0
        else:
            num_no_change_in_min_loss += 1
            if num_no_change_in_min_loss >= 3:
                break

    return epoch_losses, evaluation_losses


def plot_losses(train_losses: list, eval_losses: list):
    epochs = [i for i in range(len(train_losses))]
    plt.plot(epochs, train_losses, 'r', label='Train loss')
    plt.plot(epochs, eval_losses, 'g', label='Eval loss')
    plt.ylabel('Losses')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    from a4_ex1 import SimpleNetwork
    from dataset import get_dataset
    torch.random.manual_seed(0)
    train_data, eval_data = get_dataset()
    network = SimpleNetwork(32, 128, 1)
    train_losses, eval_losses = training_loop(network, train_data, eval_data, num_epochs=100, show_progress=True)
    plot_losses(train_losses, eval_losses)
