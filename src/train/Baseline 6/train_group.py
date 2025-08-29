import argparse
import os
import sys
import yaml
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

# Files for train model
from data import GroupActivityDataset, ACTIVITIES_LABELS
from collate_fn import group_collate_fn
from models import PersonClassifier, SceneTempClassifier_B6
from utils import (load_config, load_checkpoint,
                     save_checkpoint, setup_logging)
from eval import get_f1_score, plot_confusion_matrix


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, writer, logger):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0
    correct_preds = 0
    total_samples = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # use mixed precision to speed up training and reduce memory usage
        with autocast(dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()

        # Calculate accuracy
        predicted = outputs.argmax(dim=1)
        target_class = targets.argmax(dim=1)
        total_samples += targets.size(0)
        correct_preds += predicted.eq(target_class).sum().item()
        
        if batch_idx % 50 == 0:
            current_acc = 100. * correct_preds / total_samples
            logger.info(f'Epoch: {epoch+1} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%')
            
            # Log batch-level metrics to TensorBoard
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Training/Batch_Loss', loss.item(), step)
            writer.add_scalar('Training/Batch_Accuracy', current_acc, step)
            
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100. * correct_preds / total_samples
    
    # Log epoch-level metrics to TensorBoard
    writer.add_scalar('Training/Epoch_Loss', epoch_loss, epoch)
    writer.add_scalar('Training/Epoch_Accuracy', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc



def validate_model(model, val_loader, criterion, device, epoch, writer, logger, class_names):
    """Validates the model on the validation set."""
    model.eval()
    total_loss = 0
    correct_preds = 0
    total_samples = 0
    
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast(dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            total_loss += loss.item()

            # For Metrics
            predicted = outputs.argmax(dim=1)
            target_class = targets.argmax(dim=1)
            total_samples += targets.size(0)
            correct_preds += predicted.eq(target_class).sum().item()
            
            y_true.extend(target_class.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct_preds / total_samples
    f1 = get_f1_score(y_true, y_pred, average="weighted")
    
    logger.info(f"Epoch {epoch+1} | Valid Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | F1 Score: {f1:.4f}")
    writer.add_scalar('Validation/Loss', avg_loss, epoch)
    writer.add_scalar('Validation/Accuracy', accuracy, epoch)
    writer.add_scalar('Validation/F1_Score', f1, epoch)
    
    fig = plot_confusion_matrix(y_true, y_pred, class_names, save_path=None)
    writer.add_figure('Validation/Confusion_Matrix', fig, epoch)
    
    return avg_loss, accuracy


def fit(config_path, checkpoint_path=None, resume_train=None):
    """
    Main function to orchestrate the model training process.    
    """
    config = load_config(config_path)
    set_seed(config.system.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    person_model = PersonClassifier().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    person_model.load_state_dict(checkpoint["model_state_dict"])

    start_epoch = 0
    best_val_acc = 0
    if resume_train:
        # Resuming from a checkpoint (resume train)
        model = SceneTempClassifier_B6(person_model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001) 
        model, optimizer, loaded_config, exp_dir, start_epoch, best_val_acc = load_checkpoint(resume_train, model, optimizer, device)
        config = loaded_config 
        logger = setup_logging(exp_dir)
        logger.info(f"Resumed training from checkpoint: {resume_train}")
        logger.info(f"Starting from epoch {start_epoch + 1}. Best validation accuracy: {best_val_acc:.2f}%")
    else:
        # Starting a new experiment (start train)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{config.experiment.name_group}_V{config.experiment.version}_{timestamp}"
        exp_dir = os.path.join('/kaggle/working/', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        logger = setup_logging(exp_dir)
        logger.info(f"Starting new experiment: {exp_name}")
        
    writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tensorboard'))
    logger.info(f"Using device: {device}. Seed: {config.experiment.seed}")
    
    config_save_path = os.path.join(exp_dir, 'config.yml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
        
    if not resume_train: # Initialize if not loaded from checkpoint
        model = SceneTempClassifier_B6(person_model)
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
        crops=True,
        seq=True,
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
        collate_fn=group_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.group_activity.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        collate_fn=group_collate_fn
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    logger.info("Starting training process")
    for epoch in range(start_epoch, config.training.epochs):
        logger.info(f"\n--- Epoch {epoch+1}/{config.training.epochs} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, writer, logger)
        logger.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        val_loss, val_acc = validate_model(model, val_loader, criterion, device, epoch, writer, logger, config.dataset.label_classes.group_activity)

        scheduler.step(val_loss)
        
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            logger.info(f"New best validation accuracy: {best_val_acc:.2f}%! Saving model...")

        # Save checkpoint
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
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to the checkpoint file to load the pretrained person model")
    parser.add_argument('--resume_train', type=str, default=None, help="Path to the checkpoint file to resume training")
    args = parser.parse_args()

    # run all in fit fun 
    fit(args.config_path, checkpoint_path=args.checkpoint_path, resume_train=args.resume_train)




