import torch

def collate_fn_B4(batch):
    clips, labels = zip(*batch) 
    clips = torch.stack(clips, dim=0) 
    labels = torch.stack(labels, dim=0)  
    labels = labels[:, -1, :]  # utile the label of last frame
    return clips, labels
