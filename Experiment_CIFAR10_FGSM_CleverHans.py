import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

from torchmetrics import Mean, Accuracy

class CNN(nn.Module):
    def __init__(self, nb_filters=64):
        super(CNN, self).__init__()
        img_size = 32
        log_resolution = int(round(math.log(img_size) / math.log(2)))
        conv_args = dict(activation=nn.functional.leaky_relu, kernel_size=3, padding=1)
        self.layers_obj = nn.ModuleList()
        for scale in range(log_resolution - 2):
            conv1 = nn.Conv2d(nb_filters << scale, nb_filters << scale, **conv_args)
            conv2 = nn.Conv2d(nb_filters << scale, nb_filters << (scale + 1), **conv_args)
            pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.layers_obj.extend([conv1, conv2, pool])
        conv = nn.Conv2d(nb_filters << (log_resolution - 2), 10, **conv_args)
        self.layers_obj.append(conv)

    def forward(self, x):
        for layer in self.layers_obj:
            x = layer(x)
        return torch.mean(x, [2, 3])


def ld_cifar10():
    """Load training and test data."""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=128, shuffle=False, num_workers=2)

    return train_loader, test_loader


def main():
    # Load training and test data
    train_loader, test_loader = ld_cifar10()
    model = CNN()
    loss_object = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Metrics to track the different accuracies.
    train_loss = Mean()
    test_acc_clean = Accuracy()
    test_acc_fgsm = Accuracy()
    test_acc_pgd = Accuracy()

    def train_step(x, y):
        model.train()
        optimizer.zero_grad()
        predictions = model(x)
        loss = loss_object(predictions, y)
        loss.backward()
        optimizer.step()
        train_loss(loss.item())

    # Train model with adversarial training
    for epoch in range(FLAGS.nb_epochs):
        progress_bar_train = tqdm(train_loader)
        for batch_idx, (x, y) in enumerate(progress_bar_train):
            x, y = x.cuda(), y.cuda()
            if FLAGS.adv_train:
                # Replace clean example with adversarial example for adversarial training
                x_adv = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, np.inf)
                x = x_adv
            train_step(x, y)
            progress_bar_train.set_description(f"Epoch {epoch+1}/{FLAGS.nb_epochs} - Loss: {train_loss.result():.4f}")

    # Evaluate on clean and adversarial data
    progress_bar_test = tqdm(test_loader)
    model.eval()
    with torch.no_grad():
        for x, y in progress_bar_test:
            x, y = x.cuda(), y.cuda()

            y_pred = model(x)
            test_acc_clean(y_pred, y)

            x_fgm = fast_gradient_method(model, x, FLAGS.eps, np.inf)
            y_pred_fgm = model(x_fgm)
            test_acc_fgsm(y_pred_fgm, y)

            x_pgd = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, np.inf)
            y_pred_pgd = model(x_pgd)
            test_acc_pgd(y_pred_pgd, y)

    print("Test acc on clean examples (%): {:.3f}".format(test_acc_clean.compute() * 100))
    print("Test acc on FGM adversarial examples (%): {:.3f}".format(test_acc_fgsm.compute() * 100))
    print("Test acc on PGD adversarial examples (%): {:.3f}".format(test_acc_pgd.compute() * 100))


if __name__ == "__main__":
    FLAGS = EasyDict({
        "nb_epochs": 8,
        "eps": 0.05,
        "adv_train": False
    })
    main()
