import os
import torch

def save_checkpoint(checkpoint, is_best=False):
    """
    Save the best model and checkpoint model 
    """

    checkpoint_path = os.path.join(checkpoint['exp_dir'], f"checkpoint_epoch_{checkpoint['epoch']}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
 
    if is_best:
        best_model_path = os.path.join(checkpoint['exp_dir'], 'best_model.pth')
        torch.save(checkpoint, best_model_path)
        print(f"Best model saved to {best_model_path}")

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """
    Load model for resume training and test model 
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer == None: 
        return model
    
    start_epoch = checkpoint["epoch"] + 1
    config = checkpoint["config"]
    exp_dir = checkpoint["exp_dir"]
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, config, exp_dir, start_epoch
