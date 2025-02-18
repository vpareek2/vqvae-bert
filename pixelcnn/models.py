import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import time
import os
import sys
from pathlib import Path

# Add VQVAE and PixelCNN directories to path
current_path = Path.cwd()
sys.path.append(str(current_path))
sys.path.append(str(current_path / 'pixelcnn'))

from pixelcnn.models import GatedPixelCNN
import utils

import argparse
# Parse arguments
parser = argparse.ArgumentParser()

# Training parameters
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("-save", action="store_true")
parser.add_argument("-gen_samples", action="store_true")

# Model parameters
parser.add_argument("--dataset", type=str, default='LATENT_BLOCK',
    help='accepts CIFAR10 | MNIST | FashionMNIST | LATENT_BLOCK')
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--img_dim", type=int, default=8)
parser.add_argument("--input_dim", type=int, default=1,
    help='1 for grayscale 3 for rgb')
parser.add_argument("--n_embeddings", type=int, default=512,
    help='number of embeddings from VQ VAE')
parser.add_argument("--n_layers", type=int, default=15)
parser.add_argument("--learning_rate", type=float, default=3e-4)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loaders
if args.dataset == 'LATENT_BLOCK':
    _, _, train_loader, test_loader, _ = utils.load_data_and_data_loaders('LATENT_BLOCK', args.batch_size)
else:
    train_loader = torch.utils.data.DataLoader(
        eval('datasets.'+args.dataset)(
            '../data/{}/'.format(args.dataset), train=True, download=True,
            transform=transforms.ToTensor(),
        ), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        eval('datasets.'+args.dataset)(
            '../data/{}/'.format(args.dataset), train=False,
            transform=transforms.ToTensor(),
        ), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

model = GatedPixelCNN(args.n_embeddings, args.img_dim**2, args.n_layers).to(device)
criterion = nn.CrossEntropyLoss().to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

def train():
    train_loss = []
    for batch_idx, (x, label) in enumerate(train_loader):
        start_time = time.time()

        if args.dataset == 'LATENT_BLOCK':
            x = x[:, 0].to(device)
        else:
            x = (x[:, 0] * (args.n_embeddings-1)).long().to(device)
        label = label.to(device)

        # Train PixelCNN with images
        logits = model(x, label)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        loss = criterion(
            logits.view(-1, args.n_embeddings),
            x.view(-1)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss.append(loss.item())

        if (batch_idx + 1) % args.log_interval == 0:
            print(f'\tIter: [{batch_idx * len(x)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: '
                  f'{np.mean(train_loss[-args.log_interval:]):.6f} '
                  f'Time: {time.time() - start_time:.3f}s')

def test():
    start_time = time.time()
    val_loss = []
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            if args.dataset == 'LATENT_BLOCK':
                x = x[:, 0].to(device)
            else:
                x = (x[:, 0] * (args.n_embeddings-1)).long().to(device)
            label = label.to(device)

            logits = model(x, label)
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss = criterion(
                logits.view(-1, args.n_embeddings),
                x.view(-1)
            )

            val_loss.append(loss.item())

    mean_loss = np.mean(val_loss)
    print(f'Validation Completed!\tLoss: {mean_loss:.6f} Time: {time.time() - start_time:.3f}s')
    return mean_loss

def generate_samples(epoch):
    label = torch.arange(10).expand(10, 10).contiguous().view(-1)
    label = label.to(device)

    x_tilde = model.generate(label, shape=(args.img_dim, args.img_dim), batch_size=100)
    print(x_tilde[0])

# Training loop
BEST_LOSS = float('inf')
LAST_SAVED = -1

for epoch in range(1, args.epochs + 1):
    print(f"\nEpoch {epoch}:")
    train()
    cur_loss = test()

    if args.save or cur_loss <= BEST_LOSS:
        BEST_LOSS = cur_loss
        LAST_SAVED = epoch
        print("Saving model!")
        torch.save(model.state_dict(), f'results/{args.dataset}_pixelcnn.pt')
    else:
        print(f"Not saving model! Last saved: {LAST_SAVED}")

    if args.gen_samples:
        generate_samples(epoch)
