
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from utils import Trainer,Tester
from torch_lr_finder import LRFinder
import numpy as np


def save_model(model, epoch, optimizer, path):
    """Save torch model in .pt format

    Args:
        model (instace): torch instance of model to be saved
        epoch (int): epoch num
        optimizer (instance): torch optimizer
        path (str): model saving path
    """
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, path)


def train_model(
    trainer, tester, NUM_EPOCHS, use_l1=False, scheduler=None, save_best=False
):
    for epoch in range(1, NUM_EPOCHS + 1):
        trainer.train(epoch, scheduler)
        _, test_loss = tester.test()

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)

        if save_best:
            min_val_loss = np.inf
            save_path = "model.pt"
            if test_loss < min_val_loss:
                print(
                    f"Valid loss reduced from {min_val_loss:.5f} to {test_loss:.6f}. checkpoint created at...{save_path}\n"
                )
                save_model(trainer.model, epoch, trainer.optimizer, save_path)
                min_val_loss = test_loss
            else:
                print(f"Valid loss did not inprove from {min_val_loss:.5f}")

        print()

    if scheduler:
        return trainer.model, (
            trainer.train_accuracies,
            trainer.train_losses,
            tester.test_accuracies,
            tester.test_losses,
            trainer.lr_history,
        )
    else:
        return trainer.model, (
            trainer.train_accuracies,
            trainer.train_losses,
            tester.test_accuracies,
            tester.test_losses,
        )


def get_lr(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    end_lr=10,
    num_iter=200,
    step_mode="exp",
    start_lr=None,
    diverge_th=5,
):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(
        train_loader,
        end_lr=end_lr,
        num_iter=num_iter,
        step_mode=step_mode,
        start_lr=start_lr,
        diverge_th=diverge_th,
    )
    lr_finder.plot()
    min_loss = min(lr_finder.history["loss"])
    max_lr = lr_finder.history["lr"][np.argmin(lr_finder.history["loss"], axis=0)]

    print("Min Loss = {}, Max LR = {}".format(min_loss, max_lr))

    # Reset the model and optimizer to initial state
    lr_finder.reset()

    return min_loss, max_lr
