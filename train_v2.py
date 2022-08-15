
from __future__ import print_function
from cProfile import label
import os
import argparse
from typing import List
from PIL import Image
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn.functional as F
from torchvision import transforms as T
from models import *
import torchvision
import torch
import numpy as np
import random
import time
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set True to be faster
    print(f'Setting all seeds to be {seed} to reproduce...')


seed_everything(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model: nn.Module, accuracy: float, loss: float, epoch: int, save_path: str):
    torch.save({"weights": model.state_dict(),
                "accuracy": accuracy,
                "loss": loss,
                "epoch": epoch}, save_path)
    print("Save model with accuracy: {}, loss {} at epoch: {}".format(accuracy, loss, epoch))


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, default=r"D:\Face_Datasets\img_aligned_celeba_train_val_1\train")
    parser.add_argument("--val_dir", type=str, default=r"D:\Face_Datasets\img_aligned_celeba_train_val_1\val")
    parser.add_argument("--model_dir", type=str, default=r"D:\Face_Datasets\CelebA_Models")
    parser.add_argument("--checkpoint_pattern", type=str, default=r"checkpoint")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-4)

    args = parser.parse_args()

    return args


def enumerate_images(images_dir: str) -> List[str]:
    images_list: List[str] = []

    for dirs, _, files in os.walk(images_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png", ".ifjf")):
                images_list.append(os.path.join(dirs, file))

    return images_list


class FaceDataset(Dataset):

    def __init__(self, images_dir: str, phase='train'):
        super().__init__()
        self.imgs= enumerate_images(images_dir=images_dir)
        self.phase = phase
        normalize = T.Normalize(mean=[0], std=[1])

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        image = self.imgs[index]
        label = os.path.normpath(image).split(os.sep)[-2]
        data = Image.open(image)
        data = data.convert('RGB')
        data = self.transforms(data)
        label = np.int32(label)
        return data.float(), label

    def __len__(self):
        return len(self.imgs)


class FaceModel(nn.Module):

    def __init__(self, num_classes=10177):
        super().__init__()

        self.backbone = resnet_face18(use_se=False)
        self.metric_fc = ArcMarginProduct(512, num_classes, s=30, m=0.5, easy_margin=False)

    def forward(self, inputs, labels):
        feature = self.backbone(inputs)
        output = self.metric_fc(feature, labels)

        return output

    def extract_features(self, inputs):
        return self.backbone(inputs)


def train_epoch(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer: optim.Optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        pred = model(images, labels)
        loss = loss_fn(pred, labels)

        loss.backward()

        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(images)
            print("Loss: {}, [{}/{}]".format(loss, current, size))


def val_loop(dataloader: DataLoader, model: nn.Module, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).long()

            pred = model(images, labels)

            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
        
    test_loss /= num_batch
    correct /= size
    print("Accuracy: {}%, Avg loss: {}".format(correct * 100, test_loss))
    return correct, test_loss


if __name__ == "__main__":
    args = get_args()

    train_dir = args.train_dir
    val_dir = args.val_dir
    model_dir = args.model_dir
    checkpoint_pattern = args.checkpoint_pattern
    pretrained = args.pretrained
    num_epochs  = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    train_dataset = FaceDataset(images_dir=train_dir)
    val_dataset = FaceDataset(images_dir=val_dir)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = FaceModel(num_classes=10177)

    if pretrained:
        model.load_state_dict(torch.load(pretrained, map_location=torch.device("cpu"))["weights"])
    model.to(device=device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    max_accuracy = -np.inf
    save_path = os.path.join(model_dir, checkpoint_pattern + ".pth")
    
    for t in range(num_epochs):
        print("Epoch {}\n-------------------------------------------------".format(t + 1))
        train_epoch(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
        val_acc, val_loss = val_loop(dataloader=val_dataloader, model=model, loss_fn=loss_fn)
        if val_acc > max_accuracy:
            max_accuracy = val_acc
            save_model(model=model, accuracy=val_acc, loss=val_loss, epoch=t+1, save_path=save_path)
    print("Done")