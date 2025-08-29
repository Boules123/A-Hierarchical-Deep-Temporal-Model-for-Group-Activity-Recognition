from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score

import os

import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def get_f1_score(y_true, y_pred, average="weighted"):
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def get_classification_report(y_true, y_pred, class_names):
    return classification_report(
        y_true, y_pred,
        target_names=class_names if class_names else None,
    )


def plot_confusion_matrix(y_true,y_pred,class_names,save_path, display=False):
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        fig, ax = plt.subplots(figsize=(0.6 * len(class_names) + 3, 0.6 * len(class_names) + 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names, rotation=0)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path)
            
        if display:
            fig.show()
        
        return fig


def evaluate_model(
    model,
    data_loader,
    device,
    criterion = None,
    class_names = None,
    output_path = None,
    baseline = None
    
) :
    """
    Evaluates the model.
    """
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            
            pred = outputs.argmax(dim=1)
            target = targets.argmax(dim=1)
            
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            
            if criterion:
                loss = criterion(outputs, targets)
                total_loss += loss.item()

    avg_loss = total_loss / len(data_loader) if criterion and len(data_loader) > 0 else None
    accuracy = accuracy_score(y_true, y_pred) * 100
    weighted_f1 = get_f1_score(y_true, y_pred, average='weighted')
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    print(f"{'Accuracy':<20}: {accuracy:.2f}%")
    print(f"{'Average Loss':<20}: {avg_loss:.4f}")
    print(f"{'F1 Score (Weighted)':<20}: {weighted_f1:.4f}")
    print("\nClassification Report:")

    if class_names and output_path:
        os.makedirs(output_path, exist_ok=True) 
        save_path = os.path.join(output_path, f"confusion_matrix_{baseline}.png")
        plot_confusion_matrix(y_true, y_pred, class_names=class_names, save_path=save_path, display=True)
        print(f"\nConfusion matrix saved to: {save_path}")

    metrics = {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "f1_score": weighted_f1,
        "report_dict": report_dict,
    }
    return metrics