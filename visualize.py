import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools



def print_samples(loader,class_map,count=16):
    """Print samples input images

    Args:
        loader (DataLoader): dataloader for training data
        count (int, optional): Number of samples to print. Defaults to 16.
    """
    # Print Random Samples
    
    if not count % 8 == 0:
        return
    fig = plt.figure(figsize=(15, 5))
    for imgs, labels in loader:
        for i in range(count):
            ax = fig.add_subplot(int(count/8), 8, i + 1, xticks=[], yticks=[])
            ax.set_title(f'Label: {class_map[labels[i]]}')
            plt.imshow(imgs[i].numpy().transpose(1, 2, 0))
        break


def print_class_scale(loader, class_map):
    """Print Dataset Class scale

    Args:
        loader (DataLoader): Loader instance for dataset
        class_map (dict): mapping for class names
    """
    labels_count = {k: v for k, v in zip(
        range(0, len(class_map)), [0]*len(class_map))}
    for _, labels in loader:
        for label in labels:
            labels_count[label.item()] += 1

    labels = list(class_map.keys())
    values = list(labels_count.values())

    plt.figure(figsize=(15, 5))

    # creating the bar plot
    plt.bar(labels, values, width=0.5)
    plt.legend(labels=['Samples Per Class'])
    for l in range(len(labels)):
        plt.annotate(values[l], (-0.15 + l, values[l] + 50))
    plt.xticks(rotation=45)
    plt.xlabel("Classes")
    plt.ylabel("Class Count")
    plt.title("Classes Count")
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """Plot Confusion Matrix

    Args:
        cm (tensor): Confusion Matrix
        classes (list): Class lables
        normalize (bool, optional): Enable/Disable Normalization. Defaults to False.
        title (str, optional): Title for plot. Defaults to 'Confusion matrix'.
        cmap (str, optional): Colour Map. Defaults to plt.cm.Blues.
    """
    if normalize:
        cm = cm.type(torch.float32) / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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
        ax.set_title(f'\n Actual:{classes[t.item()]} \n Predicted:{classes[p.item()]}')
        plt.imshow(d.cpu().numpy().transpose(1, 2, 0))
        if i+1 == 5*(count/5):
            break


def plot_losses_and_accuracies(trainer, tester, epochs):
    fig, ax = plt.subplots(2, 2)

    train_epoch_loss_linspace = np.linspace(0, epochs, len(trainer.train_losses))
    test_epoch_loss_linspace = np.linspace(0, epochs, len(tester.test_losses))
    train_epoch_acc_linspace = np.linspace(0, epochs, len(trainer.train_accuracies))
    test_epoch_acc_linspace = np.linspace(0, epochs, len(tester.test_accuracies))

    ax[0][0].set_xlabel("Epoch")
    ax[0][0].set_ylabel("Train Loss")
    ax[0][0].plot(train_epoch_loss_linspace, trainer.train_losses)
    ax[0][0].tick_params(axis="y", labelleft=True, labelright=True)

    ax[0][1].set_xlabel("Epoch")
    ax[0][1].set_ylabel("Test Loss")
    ax[0][1].plot(test_epoch_loss_linspace, tester.test_losses)
    ax[0][1].tick_params(axis="y", labelleft=True, labelright=True)

    ax[1][0].set_xlabel("Epoch")
    ax[1][0].set_ylabel("Train Accuracy")
    ax[1][0].plot(train_epoch_acc_linspace, trainer.train_accuracies)
    ax[1][0].tick_params(axis="y", labelleft=True, labelright=True)
    ax[1][0].yaxis.set_ticks(np.arange(0, 101, 5))

    ax[1][1].set_xlabel("Epoch")
    ax[1][1].set_ylabel("Test Accuracy")
    ax[1][1].plot(test_epoch_acc_linspace, tester.test_accuracies)
    ax[1][1].tick_params(axis="y", labelleft=True, labelright=True)
    ax[1][1].yaxis.set_ticks(np.arange(0, 101, 5))

    fig.set_size_inches(30, 10)
    plt.tight_layout()
    plt.show()
def plot_lr_history(trainer, epochs):
    fig, ax = plt.subplots()

    linspace = np.linspace(0, epochs, len(trainer.lr_history))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.plot(linspace, trainer.lr_history)
    ax.tick_params(axis="y", labelleft=True, labelright=True)

    # fig.set_size_inches(30, 10)
    plt.tight_layout()
    plt.show()    
   
