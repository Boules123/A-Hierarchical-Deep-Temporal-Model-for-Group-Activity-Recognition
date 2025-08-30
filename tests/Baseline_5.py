
import argparse

import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import load_config, load_checkpoint
from models import SceneClassifier_B5, PersonTempClassifier
from data import GroupActivityDataset, ACTIVITIES_LABELS
from eval import evaluate_model
import torch


def group_collate_fn(batch):
    """
    Collate function to pad players to 12 per frame
    """
    clips, labels = zip(*batch)
    
    padded_clips = []

    for clip in clips:
        if clip.size(0) < 12:
            padding_size = 12 - clip.size(0)
            clip_padding = torch.zeros(padding_size, *clip.shape[1:])
            clip = torch.cat([clip, clip_padding])
        
        padded_clips.append(clip)
    
    padded_clips = torch.stack(padded_clips)
    labels = torch.stack(labels)
    
    # Use the label of last frame
    labels = labels[:, -1, :]
    
    return padded_clips, labels


def test_model(args):
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the pre-trained person classifier
    model_person = PersonTempClassifier().to(device)
    checkpoint_person = torch.load(args.checkpoint_person, map_location=device, weights_only=False)
    model_person.load_state_dict(checkpoint_person['model_state_dict'])
    
    model = SceneClassifier_B5(model_person).to(device)
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
        crops=True,
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
        num_workers=4, 
        collate_fn=group_collate_fn
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate_model(
        model=model, 
        data_loader=test_loader, 
        device=device,
        criterion=criterion,
        class_names=config.dataset.label_classes.group_activity,
        output_path=None,
        baseline="B5"
    )
    
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
        "--checkpoint", 
        type=str, 
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
Accuracy : 78.61%
Average Loss: 0.6390
F1 Score (Weighted): 0.7870

Classification Report:
              precision    recall  f1-score   support

       r_set       0.88      0.74      0.80       192
     r_spike       0.89      0.88      0.89       173
      r-pass       0.72      0.82      0.77       210
  r_winpoint       0.53      0.51      0.52        87
  l_winpoint       0.60      0.68      0.64       102
      l-pass       0.80      0.78      0.79       226
     l-spike       0.88      0.87      0.87       179
       l_set       0.83      0.83      0.83       168

    accuracy                           0.79      1337
   macro avg       0.77      0.76      0.76      1337
weighted avg       0.79      0.79      0.79      1337
"""