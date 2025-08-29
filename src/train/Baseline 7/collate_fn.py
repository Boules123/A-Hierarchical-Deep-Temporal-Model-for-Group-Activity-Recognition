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
