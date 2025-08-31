import torch


def collate_fn(batch):
    """
    Collate function to ensure each frame has 12 players.
    Pads with zeros if fewer than 12 are present.
    """
    clips, labels = zip(*batch)
    
    max_players = 12
    padded_clips = []
    padded_labels = []

    for clip, label in zip(clips, labels):
        if clip.size(0) < max_players:
            padding = torch.zeros(max_players - clip.size(0), *clip.shape[1:], dtype=clip.dtype)
            clip = torch.cat([clip, padding], dim=0)
        padded_clips.append(clip)
        padded_labels.append(label)
    
    return torch.stack(padded_clips), torch.stack(padded_labels)
