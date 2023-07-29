
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from functools import reduce
from typing import Union
from tqdm import tqdm
import numpy as np
from data import Cifar10Dataset

 

# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

class Trainer:
    def __init__(self, model, train_loader, optimizer, criterion, device) -> None:
        self.train_losses = []
        self.train_accuracies = []
        self.epoch_train_accuracies = []
        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.lr_history = []

    def train(self, epoch, scheduler=None, use_l1=False, lambda_l1=0.01):
        self.model.train()

        lr_trend = []
        correct = 0
        processed = 0
        train_loss = 0

        pbar = tqdm(self.train_loader)

        for batch_id, (inputs, targets) in enumerate(pbar):
            # transfer to device
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Initialize gradients to 0
            self.optimizer.zero_grad()

            # Prediction
            outputs = self.model(inputs)

            # Calculate loss
            loss = self.criterion(outputs, targets)

            l1 = 0
            if use_l1:
                for p in self.model.parameters():
                    l1 = l1 + p.abs().sum()
            loss = loss + lambda_l1 * l1

            self.train_losses.append(loss.item())

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            # updating LR
            if scheduler:
                if not isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step()
                    lr_trend.append(scheduler.get_last_lr()[0])

            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            processed += len(inputs)

            pbar.set_description(
                desc=f"EPOCH = {epoch} | LR = {self.optimizer.param_groups[0]['lr']} | Loss = {loss.item():3.2f} | Batch = {batch_id} | Accuracy = {100*correct/processed:0.2f}"
            )
            self.train_accuracies.append(100 * correct / processed)

        # After all the batches are done, append accuracy for epoch
        self.epoch_train_accuracies.append(100 * correct / processed)

        self.lr_history.extend(lr_trend)
        return 100 * correct / processed, train_loss / (batch_id + 1), lr_trend

class Tester:
    def __init__(self, model, test_loader, criterion, device) -> None:
        self.test_losses = []
        self.test_accuracies = []
        self.model = model.to(device)
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device

    def test(self):
        self.model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = self.model(inputs)
                loss = self.criterion(output, targets)

                test_loss += loss.item()

                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(targets.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)

        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                test_loss,
                correct,
                len(self.test_loader.dataset),
                100.0 * correct / len(self.test_loader.dataset),
            )
        )

        self.test_accuracies.append(100.0 * correct / len(self.test_loader.dataset))

        return 100.0 * correct / len(self.test_loader.dataset), test_loss

    def get_misclassified_images(self):
        self.model.eval()

        images = []
        predictions = []
        labels = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                _, preds = torch.max(output, 1)

                for i in range(len(preds)):
                    if preds[i] != target[i]:
                        images.append(data[i])
                        predictions.append(preds[i])
                        labels.append(target[i])

        return images, predictions, labels

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


def plot_incorrect_prediction(mismatch, n=10 ):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    display_images = mismatch[:n]
    index = 0
    fig = plt.figure(figsize=(10,5))
    for img in display_images:
        image = img[0].squeeze().to('cpu').numpy()
        pred = classes[img[1]]
        actual = classes[img[2]]
        ax = fig.add_subplot(2, 5, index+1)
        ax.axis('off')
        ax.set_title(f'\n Predicted Label : {pred} \n Actual Label : {actual}',fontsize=10) 
        ax.imshow(np.transpose(image, (1, 2, 0))) 
        index = index + 1
    plt.show()

def get_all_predictions(model, loader, device):
    """Get All predictions for model

    Args:
        model (Net): Trained Model 
        loader (Dataloader): instance of dataloader
        device (str): Which device to use cuda/cpu

    Returns:
        tuple: all predicted values and their targets
    """
    model.eval()
    all_preds = torch.tensor([]).to(device)
    all_targets = torch.tensor([]).to(device)
    with torch.no_grad():
        for data, target in loader:
            data, targets = data.to(device), target.to(device)
            all_targets = torch.cat(
                (all_targets, targets),
                dim=0
            )
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds = torch.cat(
                (all_preds, preds),
                dim=0
            )

    return all_preds, all_targets   

def get_incorrect_predictions(model, loader, device):
    """Get all incorrect predictions

    Args:
        model (Net): Trained model
        loader (DataLoader): instance of data loader
        device (str): Which device to use cuda/cpu

    Returns:
        list: list of all incorrect predictions and their corresponding details
    """
    model.eval()
    incorrect = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            pred = output.argmax(dim=1)
            for d, t, p, o in zip(data, target, pred, output):
                if p.eq(t.view_as(p)).item() == False:
                    incorrect.append(
                        [d.cpu(), t.cpu(), p.cpu(), o[p.item()].cpu()])

    return incorrect    

def plot_incorrect_predictions(predictions, class_map, count=10):
    """Plot Incorrect predictions

    Args:
        predictions (list): List of all incorrect predictions
        class_map (list): Lable mapping
        count (int, optional): Number of samples to print, multiple of 5. Defaults to 10.
    """
    print(f'Total Incorrect Predictions {len(predictions)}')

    if not count % 5 == 0:
        print("Count should be multiple of 10")
        return

    #classes = list(class_map.values())
    classes = class_map 
    fig = plt.figure(figsize=(10, 5))
    for i, (d, t, p, o) in enumerate(predictions):
        ax = fig.add_subplot(int(count/5), 5, i + 1, xticks=[], yticks=[])
        ax.set_title(f'{classes[t.item()]}/{classes[p.item()]}')
        plt.imshow(d.cpu().numpy().transpose(1, 2, 0))
        if i+1 == 5*(count/5):
            break  

def prepare_confusion_matrix(all_preds, all_targets, class_map):
    """Prepare Confusion matrix

    Args:
        all_preds (list): List of all predictions
        all_targets (list): List of all actule labels
        class_map (list): Class names

    Returns:
        tensor: confusion matrix for size number of classes * number of classes
    """
    stacked = torch.stack((
        all_targets, all_preds
    ),
        dim=1
    ).type(torch.int64)

    no_classes = len(class_map)

    # Create temp confusion matrix
    confusion_matrix = torch.zeros(no_classes, no_classes, dtype=torch.int64)

    # Fill up confusion matrix with actual values
    for p in stacked:
        tl, pl = p.tolist()
        confusion_matrix[tl, pl] = confusion_matrix[tl, pl] + 1

    return confusion_matrix

def get_stats(trainloader):
  """
  Args:
      trainloader (trainloader): Original data with no preprocessing
  Returns:
      mean: per channel mean
      std: per channel std
  """
  train_data = trainloader.dataset.data

  print('[Train]')
  print(' - Numpy Shape:', train_data.shape)
  print(' - Tensor Shape:', train_data.shape)
  print(' - min:', np.min(train_data))
  print(' - max:', np.max(train_data))

  train_data = train_data / 255.0

  mean = np.mean(train_data, axis=tuple(range(train_data.ndim-1)))
  std = np.std(train_data, axis=tuple(range(train_data.ndim-1)))

  print(f'\nDataset Mean - {mean}')
  print(f'Dataset Std - {std} ')

  return([mean, std])


def get_train_loader(transform=None):
  """
  Args:
      transform (transform): Albumentations transform
  Returns:
      trainloader: DataLoader Object
  """
  if transform:
    trainset = Cifar10Dataset(transform=transform)
  else:
    trainset = Cifar10Dataset(root="~/data/cifar10", train=True, 
                                    download=True)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                            shuffle=True, num_workers=2)
  return(trainloader)


def get_test_loader(transform=None):
  """
  Args:
      transform (transform): Albumentations transform
  Returns:
      testloader: DataLoader Object
  """
  if transform:
    testset = Cifar10Dataset(transform=transform, train=False)
  else:
    testset = Cifar10Dataset(train=False)
  testloader = torch.utils.data.DataLoader(testset, batch_size=512, 
                                         shuffle=False, num_workers=2)

  return(testloader)


def get_summary(model, device):
  """
  Args:
      model (torch.nn Model): Original data with no preprocessing
      device (str): cuda/CPU
  """
  print(summary(model, input_size=(3, 32, 32)))



def get_device():
  """
  Returns:
      device (str): device type
  """
  SEED = 1

  # CUDA?
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  # For reproducibility
  if cuda:
      torch.cuda.manual_seed(SEED)
  else:
    torch.manual_seed(SEED)

  return(device)

def get_layers(module: Union[torch.Tensor, nn.Module], access_string: str):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)
 
def plot_grad_cam(
    model,
    device,
    images,
    labels,
    predictions,
    target_layer,
    classes,
    use_cuda=True,
):
    """
    model = model,
    device = device,
    images = input images
    labels = correct classes for the images
    predictions = predictions for the images. If the desired gradcam is for the correct classes, pass labels here.
    target_layer = string representation of layer e.g. "layer3.1.conv2"
    classes = list of class labels
    """
    target_layers = [get_layers(model, target_layer)]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

    fig = plt.figure(figsize=(32, 32))

    plot_idx = 1
    for i in range(len(images)):
        input_tensor = images[i].unsqueeze(0).to(device)
        targets = [ClassifierOutputTarget(predictions[i])]
        rgb_img = denorm(images[i].cpu().numpy().squeeze())
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Layout = 6 images per row - 2 * (original image, gradcam and visualization)
        ax = fig.add_subplot(len(images) // 2, 6, plot_idx, xticks=[], yticks=[])
        ax.imshow(rgb_img, cmap="gray")
        ax.set_title("True class: {}".format(classes[labels[i]]))
        plot_idx += 1

        ax = fig.add_subplot(len(images) // 2, 6, plot_idx, xticks=[], yticks=[])
        ax.imshow(grayscale_cam, cmap="gray")
        ax.set_title("GradCAM Output\nPredict class: {}".format(classes[predictions[i]]))
        plot_idx += 1

        ax = fig.add_subplot(len(images) // 2, 6, plot_idx, xticks=[], yticks=[])
        ax.imshow(visualization, cmap="gray")
        ax.set_title("Visualization\nPredict class: {}".format(classes[predictions[i]]))
        plot_idx += 1

    plt.tight_layout()
    plt.show()


def denorm(img):
    channel_means = (0.4914, 0.4822, 0.4465)
    channel_stdevs = (0.2470, 0.2435, 0.2616)
    img = img.astype(dtype=np.float32)

    for i in range(img.shape[0]):
        img[i] = (img[i] * channel_stdevs[i]) + channel_means[i]

    return np.transpose(img, (1, 2, 0))
