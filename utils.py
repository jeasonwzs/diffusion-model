import os
import paddle
import paddle.vision as V
from paddle.io import DataLoader
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


def save_images(sampled_images, save_path, cols=4):
    for i in range(8):
        img = sampled_images[i].transpose([1, 2, 0])
        img = np.array(img).astype("uint8")
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
    plt.savefig(f"{save_path}.png")


def get_data(args):
    transforms = V.transforms.Compose([
        V.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        V.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        V.transforms.ToTensor(),
        V.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = V.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
