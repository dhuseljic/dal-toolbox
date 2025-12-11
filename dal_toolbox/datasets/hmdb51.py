from torch.utils.data import Dataset
from datasets import load_dataset
from .base import BaseData, BaseTransforms
from .transforms import CustomTransforms


class HMDB51(BaseData):

    def __init__(self,
                 dataset_path: str,
                 transforms: BaseTransforms = None,
                 val_split: float = 0.0,
                 seed: int = None) -> None:
        self.transforms =  CustomTransforms(lambda x: x, lambda x: x) if transforms is None else transforms
        self.train_transform = self.transforms.train_transform
        self.eval_transform = self.transforms.eval_transform
        self.query_transform = self.transforms.query_transform

        super().__init__(dataset_path, val_split, seed)

    @property
    def num_classes(self):
        return 51

    def download_datasets(self):
        _HMDB51(self.dataset_path)

    @property
    def full_train_dataset(self):
        return _HMDB51(self.dataset_path, split="train", transform=self.train_transform)

    @property
    def full_train_dataset_eval_transforms(self):
        return _HMDB51(self.dataset_path, split="train", transform=self.eval_transform)

    @property
    def full_train_dataset_query_transforms(self):
        return _HMDB51(self.dataset_path, split="train", transform=self.query_transform)

    @property
    def test_dataset(self):
        return _HMDB51(self.dataset_path, split="test", transform=self.eval_transform)


class _HMDB51(Dataset):
    def __init__(self, root, split='train', transform=None, target_frames=32, stride=2):
        super().__init__()
        self.transform = transform
        self.target_frames = target_frames
        self.stride = stride
        self.split = split
        if self.split not in ['train', 'test']:
            raise ValueError(f"Split must be 'train' or 'test', got {split}")

        # See: https://huggingface.co/datasets/jili5044/hmdb51
        full_ds = load_dataset("jili5044/hmdb51", split="train", cache_dir=root)
        splitted_ds = full_ds.train_test_split(test_size=0.2, seed=42)
        self.ds = splitted_ds[self.split]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        item = self.ds[index]

        decoder = item['video']
        label = item['label']
        frames = decoder[::self.stride]

        # Ensure Fixed Length (32 frames). If too short, we loop; if too long, we clip.
        if len(frames) < self.target_frames:
            # Padding logic (repeat frames) could go here,
            # but for simplicity we just repeat the video to fill space
            repeats = (self.target_frames // len(frames)) + 1
            frames = frames.repeat(repeats, 1, 1, 1)

        frames = frames[:self.target_frames]

        # 5. Prepare for Transforms
        # Permute to (Channels, Time, Height, Width) for 3D Transforms
        frames = frames.float() / 255.0
        frames = frames.permute(1, 0, 2, 3)

        if self.transform:
            frames = self.transform(frames)

        return frames, label
