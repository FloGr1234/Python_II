import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def training_loop(
        network: torch.nn.Module,
        train_data: torch.utils.data.Dataset,
        eval_data: torch.utils.data.Dataset,
        num_epochs: int,
        show_progress: bool = False
) -> tuple[list, list]:
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001)

    # create the DataLoader
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
    eval_loader = DataLoader(eval_data, batch_size=32)

    network.train()  # Set the network to training mode

    epoch_losses = []
    eval_losses = []
    for epoch_i in range(num_epochs):
        errs_train = []
        sum_losses_per_epoch = 0

        if show_progress:
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch_i + 1}/{num_epochs}", ncols=80)

        for inputs, target in train_loader:
            # print(target.shape)
            # set gradients back to zero
            optimizer.zero_grad()

            # calculate output of the network
            out = network(inputs)[:, 0]
            # calculate the losses with mse funktion
            loss_funktion = nn.MSELoss()
            loss = nn.MSELoss()(out, target)  # loss_funktion(out[0], train_data[i + idx][1])  #torch.transpose(out, 0, 1)[0]
            # calculate gradient
            loss.backward()
            # update the weights
            optimizer.step()
            # save the losses
            errs_train.append(loss)
            sum_losses_per_epoch += loss

            # -------------------------------------------------------------------------------------------
            # print("out: ", out[0])
            # print("target: ", train_data[i][1])
            # print("loss: ", loss)
            # print("loss.back: ", loss.backward())
            # print("abs loss: ", out[0] - train_data[i][1])
            # print(len(train_data[i][0]))

        mean_err = torch.mean(torch.tensor(errs_train)).item()
        epoch_losses.append(mean_err)
        # print(mean_err)

        # print(sum_losses_per_epoch/32)

        # ---------------------------------
        # Eval
        # ---------------------------------
        network.eval()
        errs_eval = []

        with torch.no_grad():
            for inputs, target in eval_loader:
                out = network(inputs)[:, 0]
                loss_funktion = nn.MSELoss()
                loss = nn.MSELoss()(out, target)  # loss_funktion(out[0], eval_data[i + idx2][1])  #torch.abs #torch.transpose(out, 0, 1)[0]
                errs_eval.append(loss)
        mean_err = torch.mean(torch.tensor(errs_eval)).item()
        eval_losses.append(mean_err)

    return epoch_losses, eval_losses


if __name__ == "__main__":
    from a4_ex1 import SimpleNetwork
    from dataset import get_dataset

    torch.random.manual_seed(0)
    train_data, eval_data = get_dataset()
    network = SimpleNetwork(32, 128, 1)
    train_losses, eval_losses = training_loop(network, train_data, eval_data, num_epochs=20, show_progress=True)
    for epoch, (tl, el) in enumerate(zip(train_losses, eval_losses)):
        print(f"Epoch: {epoch} --- Train loss: {tl:7.2f} --- Eval loss: {el:7.2f}")


