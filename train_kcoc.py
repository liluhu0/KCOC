"""Clean, portable implementation of the two-stage KCOC training procedure."""
import argparse
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm


NUM_CLASSES = 5


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ManifestDataset(Dataset):
    def __init__(self, csv_path, transform, two_views=False):
        self.csv_path = Path(csv_path)
        self.frame = pd.read_csv(self.csv_path)
        if not {"path", "label"}.issubset(self.frame.columns):
            raise ValueError(f"{csv_path} must contain path,label columns")
        self.labels = self.frame.label.astype(int).tolist()
        if any(label < 0 or label >= NUM_CLASSES for label in self.labels):
            raise ValueError("labels must be integers in [0, 4]")
        self.transform = transform
        self.two_views = two_views

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        path = Path(str(self.frame.iloc[index].path))
        if not path.is_absolute():
            path = self.csv_path.parent / path
        with Image.open(path) as source:
            image = source.convert("RGB")
        label = self.labels[index]
        if self.two_views:
            return self.transform(image), self.transform(image), label
        return self.transform(image), label


def image_transforms(size, train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomChoice([
                transforms.RandomRotation((0, 0)),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((270, 270)),
            ]),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])


def create_backbone(name, output_dim, imagenet=False):
    weights = "DEFAULT" if imagenet else None
    if name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
    elif name == "densenet121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, output_dim)
    else:
        raise ValueError(f"unsupported backbone: {name}")
    return model


def classifier_layer(model):
    return model.fc if hasattr(model, "fc") else model.classifier


def replace_classifier(model, layer):
    if hasattr(model, "fc"):
        model.fc = layer
    else:
        model.classifier = layer


class KPositiveMoCo(nn.Module):
    def __init__(self, backbone, dim, queue_size, momentum, temperature, num_positive, imagenet):
        super().__init__()
        self.encoder_q = create_backbone(backbone, dim, imagenet)
        self.encoder_k = deepcopy(self.encoder_q)
        for parameter in self.encoder_k.parameters():
            parameter.requires_grad = False
        self.momentum = momentum
        self.temperature = temperature
        self.num_positive = num_positive
        self.queue_size = queue_size
        self.register_buffer("queue", F.normalize(torch.randn(dim, queue_size), dim=0))
        self.register_buffer("queue_labels", torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_key_encoder(self):
        for query, key in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            key.data.mul_(self.momentum).add_(query.data, alpha=1 - self.momentum)

    @torch.no_grad()
    def enqueue(self, keys, labels):
        count = keys.shape[0]
        if count > self.queue_size:
            keys, labels, count = keys[-self.queue_size:], labels[-self.queue_size:], self.queue_size
        pointer = int(self.queue_ptr.item())
        first = min(count, self.queue_size - pointer)
        self.queue[:, pointer:pointer + first] = keys[:first].T
        self.queue_labels[pointer:pointer + first] = labels[:first]
        remaining = count - first
        if remaining:
            self.queue[:, :remaining] = keys[first:].T
            self.queue_labels[:remaining] = labels[first:]
        self.queue_ptr[0] = (pointer + count) % self.queue_size

    def forward(self, query_image, key_image, labels):
        query = F.normalize(self.encoder_q(query_image), dim=1)
        with torch.no_grad():
            self.update_key_encoder()
            key = F.normalize(self.encoder_k(key_image), dim=1)
        logits = torch.cat([
            torch.einsum("nc,nc->n", query, key).unsqueeze(1),
            torch.einsum("nc,ck->nk", query, self.queue.detach()),
        ], dim=1) / self.temperature

        positive_mask = torch.zeros_like(logits, dtype=torch.bool)
        positive_mask[:, 0] = True
        queue_matches = labels[:, None].eq(self.queue_labels[None, :])
        for row in range(len(labels)):
            candidates = torch.where(queue_matches[row])[0]
            if len(candidates):
                take = min(self.num_positive, len(candidates))
                chosen = candidates[torch.randperm(len(candidates), device=candidates.device)[:take]]
                positive_mask[row, chosen + 1] = True
        log_probability = F.log_softmax(logits, dim=1)
        loss = -(log_probability * positive_mask).sum(1) / positive_mask.sum(1)
        self.enqueue(key, labels)
        return loss.mean()


class OrdinalCrossEntropy(nn.Module):
    def __init__(self, beta=4.0, classes=NUM_CLASSES):
        super().__init__()
        indices = torch.arange(classes)
        distance = 2 * (indices[:, None] - indices[None, :]).abs().float() / classes
        self.register_buffer("margin", beta * distance)

    def forward(self, logits, targets):
        return F.cross_entropy(logits + self.margin[targets], targets)


def balanced_sampler(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights, len(labels), replacement=True)


def loader(csv_path, size, batch_size, train, two_views=False, balanced=False, workers=4):
    dataset = ManifestDataset(csv_path, image_transforms(size, train), two_views)
    sampler = balanced_sampler(dataset.labels) if balanced else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=train and sampler is None,
                      sampler=sampler, num_workers=workers, pin_memory=torch.cuda.is_available(),
                      drop_last=train)


def save_checkpoint(path, model, **metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **metadata}, path)


def load_state(path, device):
    return torch.load(path, map_location=device)


def pretrain(args):
    device = torch.device(args.device)
    data = loader(args.train_csv, args.image_size, args.batch_size, True, True,
                  args.balanced_sampling, args.workers)
    model = KPositiveMoCo(args.backbone, args.dim, args.queue_size, args.momentum,
                         args.temperature, args.num_positive, args.imagenet).to(device)
    optimizer = torch.optim.SGD(model.encoder_q.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best = math.inf
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for first, second, labels in tqdm(data, desc=f"pretrain {epoch + 1}/{args.epochs}"):
            first, second, labels = first.to(device), second.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(first, second, labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        scheduler.step()
        average = total / len(data.dataset)
        save_checkpoint(Path(args.output) / "pretrain_last.pth", model, backbone=args.backbone,
                        dim=args.dim, epoch=epoch, loss=average)
        if average < best:
            best = average
            save_checkpoint(Path(args.output) / "pretrain_best.pth", model,
                            backbone=args.backbone, dim=args.dim, epoch=epoch, loss=average)
        print(json.dumps({"epoch": epoch + 1, "loss": average}))


def encoder_state(checkpoint):
    prefix = "encoder_q."
    return {key[len(prefix):]: value for key, value in checkpoint["state_dict"].items()
            if key.startswith(prefix)}


@torch.no_grad()
def predictions(model, data, device):
    model.eval()
    labels, probabilities = [], []
    for images, target in data:
        probabilities.append(F.softmax(model(images.to(device)), dim=1).cpu())
        labels.append(target)
    return torch.cat(labels).numpy(), torch.cat(probabilities).numpy()


def metrics(labels, probabilities):
    predicted = probabilities.argmax(1)
    result = {
        "accuracy": accuracy_score(labels, predicted),
        "balanced_accuracy": balanced_accuracy_score(labels, predicted),
        "weighted_kappa": cohen_kappa_score(labels, predicted, weights="quadratic"),
        "macro_f1": f1_score(labels, predicted, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predicted, average="weighted", zero_division=0),
        "macro_recall": recall_score(labels, predicted, average="macro", zero_division=0),
        "macro_precision": precision_score(labels, predicted, average="macro", zero_division=0),
    }
    try:
        result["auc"] = roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
    except ValueError:
        result["auc"] = None
    return result


def finetune(args):
    device = torch.device(args.device)
    train_data = loader(args.train_csv, args.image_size, args.batch_size, True, False,
                        args.balanced_sampling, args.workers)
    val_data = loader(args.val_csv, args.image_size, args.batch_size, False,
                      workers=args.workers)
    checkpoint = load_state(args.pretrained, device)
    dim = checkpoint.get("dim", args.dim)
    model = create_backbone(args.backbone, dim)
    model.load_state_dict(encoder_state(checkpoint), strict=True)
    old = classifier_layer(model)
    replace_classifier(model, nn.Linear(old.in_features, NUM_CLASSES))
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in classifier_layer(model).parameters():
        parameter.requires_grad = True
    model.to(device)
    criterion = OrdinalCrossEntropy(args.beta).to(device)
    optimizer = torch.optim.SGD(classifier_layer(model).parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best = -math.inf
    for epoch in range(args.epochs):
        # Keep frozen encoder BatchNorm statistics fixed; only the classifier trains.
        model.eval()
        classifier_layer(model).train()
        for images, labels in tqdm(train_data, desc=f"finetune {epoch + 1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        labels, probabilities = predictions(model, val_data, device)
        result = metrics(labels, probabilities)
        save_checkpoint(Path(args.output) / "finetune_last.pth", model,
                        backbone=args.backbone, beta=args.beta, metrics=result)
        if result["weighted_kappa"] > best:
            best = result["weighted_kappa"]
            save_checkpoint(Path(args.output) / "finetune_best.pth", model,
                            backbone=args.backbone, beta=args.beta, metrics=result)
        print(json.dumps({"epoch": epoch + 1, **result}))


def evaluate(args):
    device = torch.device(args.device)
    data = loader(args.data_csv, args.image_size, args.batch_size, False,
                  workers=args.workers)
    checkpoint = load_state(args.checkpoint, device)
    model = create_backbone(args.backbone, NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    labels, probabilities = predictions(model, data, device)
    print(json.dumps(metrics(labels, probabilities), indent=2))


def common(parser):
    parser.add_argument("--backbone", choices=("resnet50", "densenet121"), default="resnet50")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("pretrain")
    common(first)
    first.add_argument("--train-csv", required=True)
    first.add_argument("--output", required=True)
    first.add_argument("--epochs", type=int, default=200)
    first.add_argument("--lr", type=float, default=0.016)
    first.add_argument("--dim", type=int, default=128)
    first.add_argument("--queue-size", type=int, default=1024)
    first.add_argument("--num-positive", type=int, default=64)
    first.add_argument("--momentum", type=float, default=0.999)
    first.add_argument("--temperature", type=float, default=0.07)
    first.add_argument("--imagenet", action="store_true")
    first.add_argument("--balanced-sampling", action="store_true")

    second = sub.add_parser("finetune")
    common(second)
    second.add_argument("--train-csv", required=True)
    second.add_argument("--val-csv", required=True)
    second.add_argument("--pretrained", required=True)
    second.add_argument("--output", required=True)
    second.add_argument("--epochs", type=int, default=30)
    second.add_argument("--lr", type=float, default=0.1)
    second.add_argument("--beta", type=float, default=4.0)
    second.add_argument("--dim", type=int, default=128)
    second.add_argument("--balanced-sampling", action="store_true")

    test = sub.add_parser("evaluate")
    common(test)
    test.add_argument("--data-csv", required=True)
    test.add_argument("--checkpoint", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    seed_everything(591884092)
    {"pretrain": pretrain, "finetune": finetune, "evaluate": evaluate}[arguments.command](arguments)
