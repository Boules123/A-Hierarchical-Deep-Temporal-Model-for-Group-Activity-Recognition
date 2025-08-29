
import argparse

import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import load_config, load_checkpoint
from models import SceneTempClassifier_B8, PersonTempClassifier
from data import GroupActivityDataset, ACTIVITIES_LABELS
from eval import evaluate_model


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
    
    # Load the pre-trained person classifier
    model_person = PersonTempClassifier().to(device)
    checkpoint_person = torch.load(args.checkpoint_person, map_location=device, weights_only=False)
    model_person.load_state_dict(checkpoint_person['model_state_dict'])
    
    model = SceneTempClassifier_B8(model_person).to(device)
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
        sort=True,
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
        collate_fn=group_collate_fn
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate_model(
        model=model, 
        data_loader=test_loader, 
        device=device,
        criterion=criterion,
        class_names=config.dataset.label_classes,
        output_path=None,
        baseline='B8'
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
Accuracy : 90.50%
Average Loss: 0.3449
F1 Score (Weighted): 0.9048

Classification Report:
              precision    recall  f1-score   support

       r_set       0.93      0.81      0.86       192
     r_spike       0.94      0.93      0.93       173
      r-pass       0.83      0.92      0.87       210
  r_winpoint       0.96      0.91      0.93        87
  l_winpoint       0.91      0.98      0.94       102
      l-pass       0.89      0.93      0.91       226
     l-spike       0.94      0.93      0.94       179
       l_set       0.91      0.87      0.89       168

    accuracy                           0.91      1337
   macro avg       0.91      0.91      0.91      1337
weighted avg       0.91      0.91      0.90      1337

"""