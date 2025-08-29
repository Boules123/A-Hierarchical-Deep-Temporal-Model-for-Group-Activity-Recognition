
import argparse

import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import load_config, load_checkpoint
from models import SceneTempClassifier_B4
from data import GroupActivityDataset, ACTIVITIES_LABELS
from eval import evaluate_model

def collate_fn_B4(batch):
    clips, labels = zip(*batch) 
    clips = torch.stack(clips, dim=0) 
    labels = torch.stack(labels, dim=0)  
    labels = labels[:, -1, :]  # use the label of last frame
    return clips, labels

def log_results(metrics):
    print("\n--- Test Results ---")
    print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
    print(f"Average Loss: {metrics.get('avg_loss', 'N/A'):.4f}")
    print(f"F1 Score (Weighted): {metrics.get('f1_score', 'N/A'):.4f}")
    print("\n--- Classification Report ---")
    print(metrics.get('report_dict', 'Not available.'))


def test_model(args):
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SceneTempClassifier_B4().to(device)
    model = load_checkpoint(
        checkpoint_path=args.checkpoint, 
        model=model,
        optimizer=None, 
        device=device
    )
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    test_dataset = GroupActivityDataset(
        seq=True,
        videos_path=config.data.videos_path,
        annot_path=config.data.annot_path,
        split=config.data.video_splits.test,
        labels=ACTIVITIES_LABELS['group'],
        transform=val_transform
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.system.num_workers, 
        collate_fn=collate_fn_B4
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate_model(
        model=model, 
        data_loader=test_loader, 
        device=device,
        criterion=criterion,
        class_names=config.dataset.label_classes,
        output_path=None,
        baseline="B4"
    )
    
    log_results(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a Scene Classifier model.")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to the configuration file (e.g., 'config.yaml')."
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to the model checkpoint file (e.g., 'model.pth.tar')."
    )
    
    args = parser.parse_args()
    test_model(args)



"""
--- Test Results ---
Accuracy : 77.56%
Average Loss: 1.0055
F1 Score (Weighted): 0.7739

Classification Report:
              precision    recall  f1-score   support

       r_set       0.76      0.67      0.71       192
     r_spike       0.77      0.86      0.81       173
      r-pass       0.68      0.74      0.71       210
  r_winpoint       0.88      0.84      0.86        87
  l_winpoint       0.85      0.92      0.88       102
      l-pass       0.79      0.66      0.72       226
     l-spike       0.84      0.90      0.87       179
       l_set       0.74      0.76      0.75       168

    accuracy                           0.78      1337
   macro avg       0.79      0.79      0.79      1337
weighted avg       0.78      0.78      0.77      1337
"""