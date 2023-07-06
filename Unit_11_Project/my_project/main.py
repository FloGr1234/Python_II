import os
import warnings

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle

from my_NN import SimpleNetwork
from my_NN import SimpleCNN
from my_NN import SimpleCNN1
from my_get_data import RandomImagePixelationDataset
# from datasets import CIFAR10, RotatedImages
from utils import plot


def evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    """Function for evaluation of a model ``model`` on the data in
    ``dataloader`` on device `device`, using the specified ``loss_fn`` loss
    function."""
    model.eval()
    # We will accumulate the mean loss
    loss = 0
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in the specified data loader
        for inputs, known_arr, targets, path in tqdm(loader, desc="Evaluating", position=0, leave=False):
            # Get a sample and move inputs and targets to device
            #inputs, known_arr, targets, _ = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get outputs of the specified model
            outputs = model(inputs)

            # Here, we could clamp the outputs to the minimum and maximum values
            # of the inputs for better performance

            # Add the current loss, which is the mean loss over all minibatch
            # samples (unless explicitly otherwise specified when creating the
            # loss function!)
            loss += loss_fn(outputs, targets).item()
    # Get final mean loss by dividing by the number of minibatch iterations
    # (which we summed up in the above loop)
    loss /= len(loader)
    model.train()
    return loss


def main(
        results_path,
        #network_config: dict,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_updates: int = 50_000,
        device: str = "cuda"
):
    """Main function that takes hyperparameters and performs training and
    evaluation of model"""
    device = torch.device(device)
    if "cuda" in device.type and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    # Set a known random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Prepare a path to plot to
    plot_path = os.path.join(results_path, "plots")
    os.makedirs(plot_path, exist_ok=True)

    # Load dataset
    # dir = r'C:\Users\flogr\OneDrive - Johannes Kepler Universität Linz\JKU_Master\Python_II\Unit_11_Project\training'
    dir = r'C:\Users\flogr\Documents\training'
    dataset = RandomImagePixelationDataset(dir, (4, 32), (4, 32), (4, 16), dtype=np.float32)

    # Split dataset into training, validation and test set

    training_set = torch.utils.data.Subset(
        dataset,
        indices=np.arange(int(len(dataset) * (4 / 5)))
    )
    validation_set = torch.utils.data.Subset(
        dataset,
        indices=np.arange(int(len(dataset) * (8 / 10)), int(len(dataset) * (9 / 10)))
    )
    test_set = torch.utils.data.Subset(
        dataset,
        indices=np.arange(int(len(dataset) * (9 / 10)), int(len(dataset) * (10/10)))
    )

    # Create data sets and data loaders with rotated targets without
    # augmentation (for evaluation)
    # training_set_eval = RotatedImages(dataset=training_set, rotation_angle=45)
    # validation_set = RotatedImages(dataset=validation_set, rotation_angle=45)
    # test_set = RotatedImages(dataset=test_set, rotation_angle=45)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=2, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    # Create augmented training data set and data loader
    # momentan noch nicht gemacht

    # Define a TensorBoard summary writer that writes to directory
    # "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))

    # Create Network
    net = SimpleCNN(1, 3, 64, 7)
    # net = SimpleNetwork(64, 128, 1)
    # net = SimpleCNN1(3, 32, 3, 10)
    net.to(device)

    # Get mse loss function
    mse = torch.nn.MSELoss()

    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    write_stats_at = 50000000  # Write status to TensorBoard every x updates
    plot_at = 1000  # Plot every x updates
    validate_at = 2000  # Evaluate model on validation set and check for new best model every x updates
    update = 0  # Current update counter
    best_validation_loss = np.inf  # Best validation loss so far
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)

    # Save initial model as "best" model (will be overwritten later)
    saved_model_file = os.path.join(results_path, "best_model.pt")
    torch.save(net, saved_model_file)

    # Train until n_updates updates have been reached
    while update < n_updates:
        for inputs, known_arr, targets, path in train_loader:
            # Get next samples
            # inputs, known_arr, targets, dir = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Get outputs of our network
            outputs = net(inputs)

            # Calculate loss, do backward pass and update weights
            #if True:  # (update < 1 / 10 * n_updates) or (update > 2 / 3 * n_updates):
            loss = mse(outputs, targets)
            #else:
             #   loss = mse(outputs[~known_arr], targets[~known_arr])
            loss.backward()
            optimizer.step()

            # Write current training status
            if (update + 1) % write_stats_at == 0:
                writer.add_scalar(tag="Loss/training", scalar_value=loss.cpu(), global_step=update)
                for i, (name, param) in enumerate(net.named_parameters()):
                    writer.add_histogram(tag=f"Parameters/[{i}] {name}", values=param.cpu(), global_step=update)
                    writer.add_histogram(tag=f"Gradients/[{i}] {name}", values=param.grad.cpu(), global_step=update)

            # Plot output
            if (update + 1) % plot_at == 0:
                plot(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
                     plot_path, update)

            # Evaluate model on validation set
            if (update + 1) % validate_at == 0:
                val_loss = evaluate_model(net, loader=val_loader, loss_fn=mse, device=device)
                writer.add_scalar(tag="Loss/validation", scalar_value=val_loss, global_step=update)
                # Save best model for early stopping
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(net, saved_model_file)

            update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progress_bar.update()

            # Increment update counter, exit if maximum number of updates is
            # reached. Here, we could apply some early stopping heuristic and
            # also exit if its stopping criterion is met
            update += 1
            if update >= n_updates:
                break

    update_progress_bar.close()
    writer.close()
    print("Finished Training!")

    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(saved_model_file)
    train_loss = evaluate_model(net, loader=train_loader, loss_fn=mse, device=device)
    val_loss = evaluate_model(net, loader=val_loader, loss_fn=mse, device=device)
    test_loss = evaluate_model(net, loader=test_loader, loss_fn=mse, device=device)
    # train_loss = 1
    # val_loss = 1
    print(f"Scores:")
    print(f"  training loss: {train_loss}")
    print(f"validation loss: {val_loss}")
    print(f"      test loss: {test_loss}")

    # Write result to file
    with open(os.path.join(results_path, "results.txt"), "w") as rf:
        print(f"Scores:", file=rf)
        print(f"  training loss: {train_loss}", file=rf)
        print(f"validation loss: {val_loss}", file=rf)
        print(f"      test loss: {test_loss}", file=rf)


if __name__ == "__main__":
    main("results", learning_rate=1e-3, weight_decay=1e-5, n_updates=10000)
