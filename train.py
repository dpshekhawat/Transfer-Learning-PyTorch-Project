# Basic usage:
#     python train.py data_directory
# Options:
#     Set directory to save checkpoints:
#       python train.py data_dir --save_dir save_directory
#     Choose architecture:
#       python train.py data_dir --arch "vgg13"
#     Set hyperparameters:
#       python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#     Use GPU for training:
#       python train.py data_dir --gpu


import os
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse


def input_parser():
    parser = argparse.ArgumentParser(description="this is a cli NN training script")

    parser.add_argument("data_dir")
    parser.add_argument("--save_dir", default=".")
    parser.add_argument(
        "--arch", default="vgg16", choices=["vgg11", "vgg13", "vgg16", "vgg19"]
    )
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--hidden_units", default=512, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--gpu", default=True, action="store_true")

    results = parser.parse_args()

    return results


def data_transform(data_dir):
    if not os.path.exists(data_dir):
        print(f"The directory {data_dir} doesn't exist")

    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"

    # Transforms for the training, validation sets
    batch_size = 16
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        ),
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        "train_data": datasets.ImageFolder(
            train_dir, transform=data_transforms["train"]
        ),
        "valid_data": datasets.ImageFolder(
            valid_dir, transform=data_transforms["valid"]
        ),
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        "trainloader": torch.utils.data.DataLoader(
            image_datasets["train_data"], batch_size, shuffle=True
        ),
        "validloader": torch.utils.data.DataLoader(
            image_datasets["valid_data"], batch_size
        ),
    }

    return (
        dataloaders["trainloader"],
        dataloaders["validloader"],
        image_datasets["train_data"].class_to_idx,
    )


# training and validation
def train_model(args, trainloader, validloader, class_to_idx):
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = (args.hidden_units,)
    epochs = (args.epochs,)
    gpu = args.gpu

    if arch == "vgg11":
        model = models.vgg11(pretrained=True)
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "vgg19":
        model = models.vgg19(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[0].in_features
    out_features = 4096
    train_classes = 102

    model.classifier = nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(out_features, train_classes),
        nn.LogSoftmax(dim=1),
    )

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(args.epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            model.train()
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            train_loss = criterion(logps, labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item()

        else:
            valid_loss = 0
            accuracy = 0

            model.eval()
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                valid_loss += criterion(logps, labels)

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

            print(
                f"Epoch: {epoch+1}/{epochs} \n"
                f"Train loss: {running_loss/len(trainloader):.3f}\n"
                f"Valid loss: {valid_loss/len(validloader):.3f}\n"
                f"Valid Accuracy: {accuracy/len(validloader):.3f}\n"
            )

    model.class_to_idx = class_to_idx

    checkpoint = {
        "network": args.arch,
        "input_size": in_features,
        "output_size": train_classes,
        "learning_rate": learning_rate,
        "batch_size": 16,
        "classifier": model.classifier,
        "epochs": epochs,
        "optimizer": optimizer.state_dict(),
        "state_dict": model.state_dict(),
        "class_to_idx": model.class_to_idx,
    }

    path = save_dir + "/" + "checkpoint.pth"

    torch.save(checkpoint, path)


if __name__ == "__main__":
    # training model
    # data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu = input_parser()

    args = input_parser()

    trainloader, validloader, class_to_idx = data_transform(args.data_dir)

    train_model(args, trainloader, validloader, class_to_idx)

    print("\nFinished Training")
