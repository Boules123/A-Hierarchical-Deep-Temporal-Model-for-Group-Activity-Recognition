
import argparse

import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import load_config, load_checkpoint
from models import SceneClassifier_B3, PersonClassifier
from data import GroupActivityDataset, ACTIVITIES_LABELS
from eval import evaluate_model

def collate_fn(batch):
    """
    collate to keep the same size of  12 player per frame.
    """
    clips, labels = zip(*batch)
    
    padded_clips = []
    padded_labels = []
    
    for clip, label in zip(clips, labels):
        # Pad clip to 12 player if needed
        if clip.size(0) < 12:
            padding_size = 12 - clip.size(0)
            padding = torch.zeros(padding_size, *clip.shape[1:])
            clip = torch.cat([clip, padding])
        
        padded_clips.append(clip)
        padded_labels.append(label)
    
    return torch.stack(padded_clips), torch.stack(padded_labels)


def test_model(args):
    """Test the model."""
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the pre-trained person classifier
    model_person = PersonClassifier().to(device)
    checkpoint_person = torch.load(args.checkpoint_person, map_location=device, weights_only=False)
    model_person.load_state_dict(checkpoint_person['model_state_dict'])
    
    model = SceneClassifier_B3(model_person).to(device)
    model = load_checkpoint(
        checkpoint_path=args.best_model_path, 
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
        crops=True,
        videos_path=config.data.videos_path,
        annot_path=config.data.annot_path,
        split=config.data.video_splits.validation,
        labels=ACTIVITIES_LABELS['group'],
        transform=val_transform
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.training.group_activity.batch_size,
        shuffle=False,
        num_workers=config.system.num_workers, 
        collate_fn=collate_fn
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate_model(
        model=model, 
        data_loader=test_loader, 
        device=device,
        criterion=criterion,
        class_names=config.dataset.label_classes.group_activity,
        output_path=None,
        baseline="B3"
    )
    
    print("--- Test Results ---\n")
    print(metrics.get('report_text', 'Not available.'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a Scene Classifier model.")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        required=True,
        help="Path to the configuration file (e.g., 'config.yaml')."
    )
    parser.add_argument(
        "--best_model_path", 
        type=str, 
        default=None,
        required=True,
        help="Path to the model checkpoint file (e.g., 'model.pth.tar')."
    )
    
    parser.add_argument(
        "--checkpoint_person", 
        type=str, 
        default=None,
        required=True,
        help="Path to the person classifier checkpoint file (e.g., 'person_model.pth.tar')."
    )
    
    args = parser.parse_args()
    test_model(args)



"""
--- Test Results ---
Accuracy : 81.11%
Average Loss: 0.5976
F1 Score (Weighted): 0.8099

Classification Report:
              precision    recall  f1-score   support

       r_set       0.86      0.78      0.82      1728
     r_spike       0.90      0.89      0.89      1557
      r-pass       0.79      0.80      0.79      1890
  r_winpoint       0.60      0.47      0.53       783
  l_winpoint       0.62      0.72      0.67       918
      l-pass       0.81      0.87      0.84      2034
     l-spike       0.89      0.90      0.90      1611
       l_set       0.83      0.84      0.84      1512

    accuracy                           0.81     12033
   macro avg       0.79      0.78      0.78     12033
weighted avg       0.81      0.81      0.81     12033
"""