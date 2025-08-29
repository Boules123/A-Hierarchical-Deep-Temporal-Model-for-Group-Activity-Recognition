import argparse

import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import load_config, load_checkpoint
from models import SceneClassifier_B1
from data import GroupActivityDataset, ACTIVITIES_LABELS
from eval import evaluate_model


def test_model(args):
    """Test the model."""
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Best model...")
    model = SceneClassifier_B1().to(device)
    model = load_checkpoint(
        checkpoint_path=args.best_model_path, 
        model=model,
        optimizer=None, 
        device=device
    )
    
    print("Preparing test dataloader...")
    val_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    test_dataset = GroupActivityDataset(
        videos_path=config.data.videos_path,
        annot_path=config.data.annot_path,
        split=config.data.video_splits.test, 
        labels=ACTIVITIES_LABELS['group'],
        transform=val_transforms
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.training.group_activity.batch_size,
        shuffle=False,
        num_workers=config.system.num_workers
    )
    
    print("Starting evaluation...")
    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate_model(
        model=model, 
        data_loader=test_loader, 
        device=device,
        criterion=criterion,
        class_names=config.dataset.label_classes.group_activity,
        output_path=None,
        baseline="B1"
    )
    
    print("--- Test Results ---\n")
    print(metrics.get('report_text', 'Not available.'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a Scene Classifier model.")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to the configuration file (e.g., 'config.yaml')."
    )
    parser.add_argument(
        "--best_model_path", 
        type=str, 
        required=True,
        help="Path to the model checkpoint file (e.g., 'model.pth.tar')."
    )
    
    args = parser.parse_args()
    test_model(args)




"""
--- Test Results ---
Accuracy : 72.79%
Average Loss: 1.3911
F1 Score (Weighted): 0.7284

Classification Report:
              precision    recall  f1-score   support

       r_set       0.70      0.65      0.68      1728
     r_spike       0.77      0.78      0.77      1557
      r-pass       0.60      0.69      0.64      1890
  r_winpoint       0.79      0.83      0.81       783
  l_winpoint       0.94      0.83      0.88       918
      l-pass       0.67      0.64      0.66      2034
     l-spike       0.81      0.85      0.83      1611
       l_set       0.72      0.68      0.70      1512

    accuracy                           0.73     12033
   macro avg       0.75      0.74      0.75     12033
weighted avg       0.73      0.73      0.73     12033
"""