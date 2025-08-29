import cv2
import pickle
import torch
import numpy as np
import albumentations as A
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Union

from box_info import BoxInfo

PERSON_ACTIVITY_CLASSES = ["Waiting", "Setting", "Digging", "Falling", "Spiking", "Blocking", "Jumping", "Moving", "Standing"]
GROUP_ACTIVITY_CLASSES = ["r_set", "r_spike", "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]

ACTIVITIES_LABELS = {
    "person": {class_name.lower(): i for i, class_name in enumerate(PERSON_ACTIVITY_CLASSES)},
    "group": {class_name: i for i, class_name in enumerate(GROUP_ACTIVITY_CLASSES)}
}



# Base class for common functions 
class BaseDataset(Dataset):
    def __init__(self, videos_path: str, annot_path: str, split: List[int],
                 labels: Dict[str, int], transform: A.Compose = None):
        """
        Args:
            videos_path (str): Path to the directory containing video frames.
            annot_path (str): Path to the pickled annotation file.
            split (List[int]): List of clip IDs to include in the dataset.
            labels (Dict[str, int]): Dictionary mapping class names to integer labels.
            transform (A.Compose, optional): Albumentations transform pipeline.
        """
        self.videos_path = Path(videos_path)
        self.split = split
        self.labels = labels
        self.transform = transform
        
        with open(annot_path, 'rb') as f:
            self.videos_annot = pickle.load(f)
            
        self.data_samples = self._prepare_data()

    def _prepare_data(self) -> List[Dict[str, Any]]:
        """
        Prepares the list of data samples for the dataset.
        """
        raise NotImplementedError("Child classes must implement the _prepare_data method.")

    def __len__(self) -> int:
        return len(self.data_samples)

    def _load_frame(self, video_id: int, clip_dir: int, frame_id: int) -> np.ndarray:
        """Loads a single image frame from disk."""
        frame_path = self.videos_path / str(video_id) / str(clip_dir) / f"{frame_id}.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(f"Failed to load frame: {frame_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB

    def _create_one_hot_label(self, category: str) -> torch.Tensor:
        """Creates a one-hot encoded tensor for a given category."""
        label = torch.zeros(len(self.labels))
        label[self.labels[category]] = 1
        return label

    def __getitem__(self, idx: int):
        raise NotImplementedError("Child classes must implement the __getitem__ method.")


# Person dataset class
class PersonActionDataset(BaseDataset):
    """
    Dataset for individual person action recognition.
    """
    def __init__(self, videos_path: str, annot_path: str, split: List[int],
                 labels: Dict[str, int], seq: bool = False, only_tar: bool = False,
                 transform: A.Compose = None):
        """
        Args:
            seq (bool): If True, returns data as sequences of frames per clip.
            only_tar (bool): If True, only uses the target frame of each clip.
        """
        self.seq = seq
        self.only_tar = only_tar
        super().__init__(videos_path, annot_path, split, labels, transform)

    def _prepare_data(self) -> List[Dict[str, Any]]:
        """
        Prepares data indices.
        """
        indices = []
        for video_id in self.split:
            clip_data = self.videos_annot[str(video_id)]
            for clip_dir, clip_info in clip_data.items():
                frames_data = clip_info['frame_boxes_dct']
                
                if self.seq:
                    indices.append({
                        'video_id': video_id,
                        'clip_dir': clip_dir,
                        'frames_data': frames_data
                    })
                
                else:
                    for frame_id, boxes in frames_data.items():
                        
                        if self.only_tar and str(frame_id) != str(clip_dir):
                            continue
                        
                        for box in boxes:
                            indices.append({
                                'video_id': video_id,
                                'clip_dir': clip_dir,
                                'frame_id': frame_id,
                                'box': box,
                            })
        return indices

    def _get_person_crop(self, frame: np.ndarray, box: BoxInfo) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts and transforms a single person crop and its label."""
        x_min, y_min, x_max, y_max = box.box
        person_crop = frame[y_min:y_max, x_min:x_max]
        
        if self.transform:
            person_crop = self.transform(image=person_crop)['image']
        
        label = self._create_one_hot_label(box.category)
        return torch.from_numpy(person_crop).permute(2, 0, 1), label

    def __getitem__(self, idx: int):
        sample = self.data_samples[idx]

        if self.seq: # [n,t, C, H, W], [n, t ,num_classes]
            all_frame_crops = []
            all_frame_labels = []
            
            frames_data = sample['frames_data']
            for frame_id, boxes in frames_data.items():
                frame = self._load_frame(sample['video_id'], sample['clip_dir'], frame_id)
                frame_crops, frame_labels = [], []
                
                for box in boxes:
                    crop, label = self._get_person_crop(frame, box)
                    frame_crops.append(crop)
                    frame_labels.append(label)
                
                all_frame_crops.append(torch.stack(frame_crops))
                all_frame_labels.append(torch.stack(frame_labels))

            crops_tensor = torch.stack(all_frame_crops).permute(1, 0, 2, 3, 4) #[n, t, c, h, w]
            labels_tensor = torch.stack(all_frame_labels).permute(1, 0, 2)
            return crops_tensor, labels_tensor

        else:
            frame = self._load_frame(sample['video_id'], sample['clip_dir'], sample['frame_id'])
            crop, label = self._get_person_crop(frame, sample['box'])
            return crop, label

            

#Group activity class
class GroupActivityDataset(BaseDataset):
    """
    Dataset for group activity recognition.
    """
    
    def __init__(self, videos_path: str, annot_path: str, split: List[int],
                 labels: Dict[str, int], seq: bool = False, crops: bool = False,
                 sort: bool = False, only_tar: bool = False, transform: A.Compose = None):
        """
        Args:
            seq (bool): If True, returns sequences (clips). Otherwise, single frames.
            crops (bool): If True, returns cropped person images. Otherwise, full frames.
            sort (bool): If True and crops=True, sorts person crops by their x-coordinate.
            only_tar (bool): If True, only uses the target frame of each clip.
        """
        self.seq = seq
        self.crops = crops
        self.sort = sort
        self.only_tar = only_tar
        super().__init__(videos_path, annot_path, split, labels, transform)

    def _prepare_data(self) -> List[Dict[str, Any]]:
        """Prepares data samples based on seq and crops flags."""
        samples = []
        for video_id in self.split:
            clip_data = self.videos_annot[str(video_id)]
            for clip_dir, clip_info in clip_data.items():
                category = clip_info['category']
                frames_data = list(clip_info['frame_boxes_dct'].items())
                
                if self.only_tar:
                    frames_data = [(fid, b) for fid, b in frames_data if str(fid) == str(clip_dir)]

                if not self.seq: 
                    for frame_id, boxes in frames_data:
                        sample = {'video_id': video_id, 'clip_dir': clip_dir, 'frame_id': frame_id, 'category': category}
                        if self.crops:
                            sample['boxes'] = boxes
                        samples.append(sample)
                
                else: 
                    sample = {'video_id': video_id, 'clip_dir': clip_dir, 'frames_data': frames_data, 'category': category}
                    samples.append(sample)
        return samples
    
    def _extract_person_crops(self, frame: np.ndarray, boxes: List[Any]) -> List[torch.Tensor]:
        """Extracts and transforms all person crops from a single frame."""
        person_crops = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box.box
            crop = frame[y_min:y_max, x_min:x_max]
            
            if self.transform:
                # The transform handles conversion to a tensor
                crop_tensor = self.transform(image=crop)['image']
            else:
                # Manually convert if no transform is applied
                crop_tensor = torch.from_numpy(crop).permute(2, 0, 1)
            
            person_crops.append(crop_tensor)
        return person_crops

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data_samples[idx]
        group_label = self._create_one_hot_label(sample['category'])

        # Full frame for B1 [c, h, w]
        if not self.crops and not self.seq:
            frame = self._load_frame(sample['video_id'], sample['clip_dir'], sample['frame_id'])
            if self.transform:
                # The transform already returns a tensor in the correct shape
                frame_tensor = self.transform(image=frame)['image']
            else:
                # Manually convert if no transform is present
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
            return frame_tensor, group_label

        # seq of frames for B4 [t, c, h, w]
        elif not self.crops and self.seq: 
            clip_frames = []
            
            for frame_id, _ in sample['frames_data']:
                frame = self._load_frame(sample['video_id'], sample['clip_dir'], frame_id)
                
                if self.transform:
                    frame_tensor = self.transform(image=frame)['image']
                else:
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
                
                clip_frames.append(frame_tensor)
            
            return torch.stack(clip_frames), group_label.expand(len(clip_frames), -1)

        # crops players from frames for B3 [n, c, h, w]
        elif self.crops and not self.seq:
            frame = self._load_frame(sample['video_id'], sample['clip_dir'], sample['frame_id'])
            crops = self._extract_person_crops(frame, sample['boxes'])
            return torch.stack(crops), group_label

        else: # players crops and seq of players for B5, B7, B8, B6 [n, t, c, h, w]
            clip_person_crops = []
            for frame_id, boxes in sample['frames_data']:
                frame = self._load_frame(sample['video_id'], sample['clip_dir'], frame_id)
                frame_crops = self._extract_person_crops(frame, boxes)
                
                if self.sort:
                    centers = [(b.box[0] + b.box[2]) / 2 for b in boxes]
                    sorted_crops = [crop for _, crop in sorted(zip(centers, frame_crops))]
                    frame_crops = sorted_crops
                
                clip_person_crops.append(torch.stack(frame_crops))
            
            clips_tensor = torch.stack(clip_person_crops).permute(1, 0, 2, 3, 4)
            return clips_tensor, group_label.expand(len(sample['frames_data']), -1)