import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import timm
import numpy as np
import argparse
import time
import json
import os
from tqdm import tqdm
from models import get_model
from utils import AverageMeter, accuracy, save_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='ViT vs CNN Comparison')
    parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--model', default='resnet50', 
                       choices=['resnet50', 'efficientnet_b0', 'vit_base_patch16_224', 
                               'deit_base_patch16_224', 'convnext_base'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-proportion', type=float, default=1.0)
    return parser.parse_args()

def get_dataloaders(dataset_name, batch_size, data_proportion=1.0):
    """Get CIFAR dataloaders with data augmentation"""
    if dataset_name == 'cifar10':
        num_classes = 10
        dataset_class = torchvision.datasets.CIFAR10
    else:  # cifar100
        num_classes = 100
        dataset_class = torchvision.datasets.CIFAR100
    
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load datasets
    train_dataset = dataset_class(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = dataset_class(root='./data', train=False, download=True, transform=test_transform)
    
    # Apply data proportion if needed
    if data_proportion < 1.0:
        num_train = len(train_dataset)
        indices = torch.randperm(num_train)[:int(num_train * data_proportion)]
        train_dataset = Subset(train_dataset, indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, num_classes

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc='Training')
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        
        pbar.set_postfix({'Loss': f'{losses.avg:.4f}', 'Acc': f'{top1.avg:.2f}%'})
    
    return losses.avg, top1.avg

def validate(model, test_loader, criterion, device):
    """Validate the model"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc='Validation'):
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
    
    return losses.avg, top1.avg

def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data
    train_loader, test_loader, num_classes = get_dataloaders(
        args.dataset, args.batch_size, args.data_proportion
    )
    
    # Model
    model = get_model(args.model, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"Training {args.model} on {args.dataset} for {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, True, f'checkpoints/{args.model}_{args.dataset}')
    
    print(f'Best validation accuracy: {best_acc:.2f}%')
    
    # Save training history
    with open(f'results/{args.model}_{args.dataset}_history.json', 'w') as f:
        json.dump(history, f)

if __name__ == '__main__':
    main()
