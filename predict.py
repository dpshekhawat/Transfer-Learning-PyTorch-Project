# Basic usage:
#     python predict.py /path/to/image checkpoint
# Options:
#     Return top K most likely classes:
#         python predict.py input checkpoint --top_k 3
#     Use a mapping of categories to real names:
#         python predict.py input checkpoint --category_names cat_to_name.json
#     Use GPU for inference:
#         python predict.py input checkpoint --gpu

import json
import numpy as np
import torch
from torchvision import transforms, models
import argparse
from PIL import Image


def input_parser():
    """
    Parse input arguments

    Arguments:
        None

    Returns:
        input : image file path
        checkpoint : trained model checkpoint path
        top_k : top k predicted classes
        category_names : category label to name path
        gpu : to use 'cuda' or not
    """
    parser = argparse.ArgumentParser(description="cli NN prediction script")

    parser.add_argument("input", default="flowers/test/1/image_06743.jpg")
    parser.add_argument("checkpoint", default="checkpoint.pth")
    parser.add_argument("--category_names", default="cat_to_name.json")
    parser.add_argument("--top_k", default=5, dest="top_k", type=int)
    parser.add_argument("--gpu", default=True, action="store_true")

    results = parser.parse_args()

    return (
        results.input,
        results.checkpoint,
        results.category_names,
        results.top_k,
        results.gpu,
    )


def load_checkpoint(filepath):
    """Load model checkpoint from filepath and returns a model rebuild using checkpoint"""
    checkpoint = torch.load(filepath)

    if checkpoint["network"] == "vgg11":
        model = models.vgg11(pretrained=True)
    elif checkpoint["network"] == "vgg13":
        model = models.vgg13(pretrained=True)
    elif checkpoint["network"] == "vgg16":
        model = models.vgg16(pretrained=True)
    elif checkpoint["network"] == "vgg19":
        model = models.vgg19(pretrained=True)

    model.classifier = checkpoint["classifier"]
    model.optimizer = checkpoint["optimizer"]
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    return model


def process_image(image):
    """Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array"""
    img_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )

    pil_image = Image.open(image)
    pil_image = img_transform(pil_image)
    # pil_image = Image.resize((255,255))

    np_image = np.array(pil_image)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = (np.transpose(np_image, (1, 2, 0)) - mean) / std
    np_image = np.transpose(np_image, (2, 0, 1))

    return np_image


def predict(image_path, checkpoint, topk, device, cat_to_name):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    model = load_checkpoint(checkpoint)
    model.to(device)

    np_image = process_image(image_path)
    tensor_image = torch.from_numpy(np_image)

    inputs = tensor_image.type(torch.FloatTensor)
    if torch.cuda.is_available():
        inputs = tensor_image.type(torch.cuda.FloatTensor)

    inputs = inputs.unsqueeze(dim=0)
    inputs.to(device)

    model.eval()
    with torch.no_grad():
        ps = torch.exp(model(inputs))

    top_ps, top_class = ps.topk(topk, dim=1)

    class_to_idx_inverted = {model.class_to_idx[cs]: cs for cs in model.class_to_idx}

    top_classes = []
    for label in top_class.cpu().detach().numpy()[0]:
        top_classes.append(class_to_idx_inverted[label])

    probs, classes = top_ps.cpu().detach().numpy()[0], top_classes

    return probs, [cat_to_name[c] for c in classes]


if __name__ == "__main__":
    input_image, checkpoint, category_names, top_k, gpu = input_parser()

    with open(category_names, "r") as f:
        cat_to_name = json.load(f)

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    top_ps, top_classes = predict(input_image, checkpoint, top_k, device, cat_to_name)

    print(f"\n Top {top_k} predictions:\n")
    for i in range(top_k):
        print(f"{i+1} - {top_classes[i]} ( probability: {(top_ps[i]):.3f} )")
