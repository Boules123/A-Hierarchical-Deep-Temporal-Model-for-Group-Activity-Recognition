import argparse
import os
import sys
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard.writer import SummaryWriter

# files for train model
from data import GroupActivityDataset, ACTIVITIES_LABELS
from models import SceneClassifier_B1
from utils import (load_config, load_checkpoint,
                     save_checkpoint, setup_logging)
from eval import get_f1_score

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(scaler, writer, logger, model, loader, criterion, optimizer, device, epoch):
    """Train the model for one epoch."""
    
    model.train()
    total_loss = 0
    total_samples = 0
    total_correct = 0
    
    for idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        with autocast(dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_samples += targets.size(0)
        
        outputs = outputs.argmax(dim=1)
        target = targets.argmax(dim=1) if targets.ndim > 1 else targets

        total_correct += outputs.eq(target).sum().item()
        
        if idx % 50 == 0:
            current_acc = 100. * total_correct / total_samples
            logger.info(f'Epoch: {epoch+1} | Batch: {idx}/{len(loader)} | Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%')

            step = epoch * len(loader) + idx
            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("Accuracy/train", current_acc, step)

        epoch_loss = total_loss / len(loader)
        writer.add_scalar("Loss/train/epoch", epoch_loss, epoch)
        
        epoch_acc = 100. * total_correct / total_samples
        writer.add_scalar("Accuracy/train/epoch", epoch_acc, epoch)

    logger.info(f'Epoch {epoch+1} completed. Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return epoch_acc, epoch_loss


def val_one_epoch(writer, logger, model, val_loader, criterion, device, epoch, class_names):
    """Validate the model for one epoch."""
    model.eval()
    total_loss = 0
    total_samples = 0
    total_correct = 0

    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast(dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            total_loss += loss.item()
            total_samples += targets.size(0)

            outputs = outputs.argmax(dim=1)
            target = targets.argmax(dim=1) if targets.ndim > 1 else targets

            total_correct += outputs.eq(target).sum().item()

            y_true.extend(target.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
            
    val_loss = total_loss / len(val_loader)
    val_acc = 100. * total_correct / total_samples
    f1 = get_f1_score(y_pred, y_true, average="weighted")

    writer.add_scalar("Loss/val/epoch", val_loss, epoch)
    writer.add_scalar("Accuracy/val/epoch", val_acc, epoch)
    writer.add_scalar("F1/val/epoch", f1, epoch)

    logger.info(f'Validation Epoch {epoch+1} completed. Average Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%, F1 Score: {f1:.4f}')

    return val_acc, val_loss


def fit(config_path, resume_train=None):
    """
    Main function to orchestrate the model training process.
    """
    config = load_config(config_path)
    set_seed(config.system.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_train:
        # Resuming from a checkpoint (resume train)
        model = SceneClassifier_B1()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        model, optimizer, loaded_config, exp_dir, start_epoch, best_val_acc = load_checkpoint(resume_train, model, optimizer, device)
        config = loaded_config 
        logger = setup_logging(exp_dir)
        logger.info(f"Resumed training from checkpoint: {resume_train}")
        logger.info(f"Starting from epoch {start_epoch + 1}. Best validation accuracy: {best_val_acc:.2f}%")
    else:
        # Starting a new experiment (start train)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{config.experiment.name}_V{config.experiment.version}_{timestamp}"
        exp_dir = os.path.join('/kaggle/working/', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        logger = setup_logging(exp_dir)
        logger.info(f"Starting new experiment: {exp_name}")
        
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tensorboard'))
    logger.info(f"Using device: {device}. Seed: {config.system.seed}")

    if not resume_train: 
        model = SceneClassifier_B1()
        model = model.to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.group_activity.weight_decay
        )
    
    train_transforms = A.Compose([
        A.Resize(224, 224),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.ColorJitter(brightness=0.2),
            A.RandomBrightnessContrast(),
            A.GaussNoise()
        ], p=0.5),
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ], p=0.05),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    val_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
 
    train_dataset = GroupActivityDataset(
        videos_path=config.data.videos_path,
        annot_path=config.data.annot_path,
        split=config.data.video_splits.train, 
        labels=ACTIVITIES_LABELS['group'],
        transform=train_transforms
    )
    
    val_dataset = GroupActivityDataset(
        videos_path=config.data.videos_path,
        annot_path=config.data.annot_path,
        split=config.data.video_splits.validation, 
        labels=ACTIVITIES_LABELS['group'],
        transform=val_transforms
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.group_activity.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4, 
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.group_activity.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    logger.info("Starting training process")
    for epoch in range(start_epoch, config.training.epochs):
        logger.info(f"\n--- Epoch {epoch+1}/{config.training.epochs} ---")

        train_acc, train_loss = train_one_epoch(scaler, writer, logger, model, train_loader, criterion, optimizer, device, epoch)

        val_acc, val_loss = val_one_epoch(writer, logger, model, val_loader, criterion, device, epoch, config.dataset.label_classes.group_activity)

        scheduler.step(val_loss)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info(f"New best Val loss: {best_val_loss:.4f}! | Val Acc: {val_acc:.4f}% Saving model...")

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'exp_dir':exp_dir,
            'config': config,
        }, is_best)

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
        logger.info(f"Current learning rate: {current_lr}")
        
    writer.close()
    logger.info("Training completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Group Activity Script")

    parser.add_argument('--config_path', type=str, required=True, help="Path to the config file")
    parser.add_argument('--resume_train_path', type=str, default=None, help="Path to the checkpoint file to load the pretrained person model")

    args = parser.parse_args()

    # run all in fit fun
    fit(args.config_path, resume_train=args.resume_train_path)


