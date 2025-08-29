import torch


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
