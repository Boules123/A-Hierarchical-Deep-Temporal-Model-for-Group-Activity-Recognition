import torch


def person_collate_fn(batch):
    """
    Collate function to pad players to 12.
    """
    clips, labels = zip(*batch)
    
    padded_clips = []
    padded_labels = []

    for clip, label in zip(clips, labels):
        if clip.size(0) < 12:
            padding_size = 12 - clip.size(0)
            clip_padding = torch.zeros(padding_size, *clip.shape[1:])
            clip = torch.cat([clip, clip_padding])
            
            label_padding = torch.zeros(padding_size, *label.shape[1:])
            label = torch.cat([label, label_padding])

        padded_clips.append(clip)
        padded_labels.append(label)

    padded_clips = torch.stack(padded_clips)
    padded_labels = torch.stack(padded_labels)  #[b, 12, f, num_classes]

    padded_labels = padded_labels[:, :, -1, :]  #[b, 12, num_classes]

    b, p, num_class = padded_labels.shape #[b * 12, num_classes]
    padded_labels = padded_labels.view(b * p, num_class)

    return padded_clips, padded_labels


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