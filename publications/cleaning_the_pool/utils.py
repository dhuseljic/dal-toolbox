import os
import torch
import torch.nn as nn

from transformers import AutoProcessor, CLIPVisionModel
from transformers import AutoImageProcessor, AutoModel

import torchaudio
import soundfile as sf
import torchvision.transforms as vision_transforms
from datasets import load_dataset, config

from dal_toolbox.models.laplace import LaplaceLinear, LaplaceModel
from dal_toolbox import datasets as dal_datasets
from dal_toolbox.datasets.utils import FeatureDataset
from dal_toolbox.datasets.transforms import CustomTransforms


image_datasets = ['cifar10', 'stl10', 'snacks', 'dopanim', 'dtd', 'cifar100', 'food101', 'flowers102',
                  'caltech101', 'stanford_dogs', 'tiny_imagenet', 'imagenet', 'esc']
text_datasets = ['agnews', 'dbpedia', 'banking77', 'clinc']


def build_backbone(args):
    if args.model.backbone == 'dinov2':
        # backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    elif args.model.backbone == 'clip':
        backbone = CLIP()
    elif args.model.backbone == 'dinov3':
        backbone = DINOv3()
    elif args.model.backbone == 'eat':
        backbone = EAT()
    else:
        raise NotImplementedError(f'Backbone {args.model.backbone} not implemented.')
    return backbone


def build_datasets(args):
    backbone = build_backbone(args)

    if args.dataset.name in image_datasets:
        data = build_image_data(args)
        if args.dataset.cache_features:
            train_ds = FeatureDataset(backbone, data.train_dataset, cache=True, cache_dir=args.dataset.path)
            test_ds = FeatureDataset(backbone, data.test_dataset, cache=True, cache_dir=args.dataset.path)
        else:
            train_ds = data.train_dataset
            test_ds = data.test_dataset
        num_classes = data.num_classes
    return train_ds, test_ds, num_classes


def build_image_data(args):
    if args.model.backbone == 'dinov2':
        transforms = DinoTransforms(size=(256, 256))
    elif args.model.backbone == 'dinov3':
        dinov3_transform = DINOv3Transform()
        transforms = CustomTransforms(dinov3_transform, dinov3_transform)
    elif args.model.backbone == 'clip':
        clip_transform = CLIPTransform()
        transforms = CustomTransforms(clip_transform, clip_transform)

    if args.dataset.name == 'cifar10':
        data = dal_datasets.CIFAR10(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'stl10':
        data = dal_datasets.STL10(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'dopanim':
        data = dal_datasets.Dopanim(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'snacks':
        data = dal_datasets.Snacks(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'dtd':
        data = dal_datasets.DTD(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'cifar100':
        data = dal_datasets.CIFAR100(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'food101':
        data = dal_datasets.Food101(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'flowers102':
        data = dal_datasets.Flowers102(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'stanford_dogs':
        data = dal_datasets.StanfordDogs(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'tiny_imagenet':
        data = dal_datasets.TinyImageNet(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'imagenet':
        data = dal_datasets.ImageNet(args.dataset.path, transforms=transforms)
    elif args.dataset.name == 'esc':
        transform = EATTransform()
        train_ds = ESC(args.dataset.path, split='train', transform=transform)
        test_ds = ESC(args.dataset.path, split='test', transform=transform)
        data = transform
        data.train_dataset = train_ds 
        data.test_dataset = test_ds
        data.num_classes = 50
    else:
        raise NotImplementedError()
    return data


def build_text_data(args):
    from datasets import load_dataset  # Huggingface import
    if args.dataset_name == "agnews":
        data = load_dataset("ag_news")
        num_classes = 4
    elif args.dataset_name == "dbpedia":
        data = load_dataset("dbpedia_14")
        data = data.rename_column("content", "text")
        num_classes = 14
    elif args.dataset_name == "banking77":
        data = load_dataset("banking77")
        num_classes = 77
        # data = data.rename_column("coarse_label", "label")
    elif args.dataset_name == "clinc":
        data = load_dataset("clinc_oos", "plus")
        data = data.rename_column("intent", "label")
        num_classes = 151
    else:
        raise NotImplementedError()
    return data, num_classes


def build_model(args, num_features, num_classes):
    laplace_kwargs = dict(mean_field_factor=args.model.mean_field_factor,
                          mc_samples=args.model.mc_samples, bias=True)
    if args.model.name == 'linear':
        model = LinearModel(num_features, num_classes, **laplace_kwargs)
    elif args.model.name == 'mlp':
        model = MLP(num_features, num_classes, **laplace_kwargs)
    elif args.model.name == 'all':
        raise NotImplementedError(f"Training of {args.model.name} not implemented.")
    else:
        raise NotImplementedError(f"Training of {args.model.name} not implemented.")

    params = [{'params': [p for n, p in model.named_parameters()]}]

    if args.optimizer.name == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.optimizer.lr, nesterov=args.optimizer.nesterov,
                                    momentum=args.optimizer.momentum, weight_decay=args.optimizer.weight_decay)
    elif args.optimizer.name == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=args.optimizer.lr,
                                      weight_decay=args.optimizer.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {args.model.name} not implemented.")
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.model.num_epochs)
    # lr_scheduler = partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=args.model.num_epochs)

    model = LaplaceModel(model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    return model


class LinearModel(nn.Module):
    def __init__(self, in_features, out_features, **laplace_kwargs):
        super().__init__()
        self.layer = LaplaceLinear(in_features, out_features, **laplace_kwargs)

    def forward_features(self, x):
        return x

    def forward_head(self, x, mean_field=False):
        if mean_field:
            out = self.layer.forward_mean_field(x)
        else:
            out = self.layer(x)
        return out

    def forward(self, x, mean_field=False):
        features = self.forward_features(x)
        logits = self.forward_head(features, mean_field=mean_field)
        return logits


class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_hidden=512, **laplace_kwargs):
        super().__init__()
        self.layer1 = nn.Linear(in_features, num_hidden)
        self.layer2 = LaplaceLinear(num_hidden, out_features, **laplace_kwargs)
        self.act = nn.ReLU()

    def forward_features(self, x):
        out = self.layer1(x)
        out = self.act(out)
        return out

    def forward_head(self, x, mean_field=False):
        if mean_field:
            out = self.layer2.forward_mean_field(x)
        else:
            out = self.layer2(x)
        return out

    def forward(self, x):
        features = self.forward_features(x)
        logits = self.forward_head(features)
        return logits

    def forward_mean_field(self, x):
        features = self.forward_features(x)
        mean_field_logits = self.forward_head(features, mean_field=True)
        return mean_field_logits


def flatten_cfg(cfg, parent_key='', sep='.'):
    from omegaconf import DictConfig
    items = []
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, (dict, DictConfig)):
            items.extend(flatten_cfg(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class DinoTransforms():
    def __init__(self, size=None, center_crop_size=224):
        if size:
            # https://github.com/facebookresearch/dino/blob/main/eval_linear.py#L65-L70
            dino_mean = (0.485, 0.456, 0.406)
            dino_std = (0.229, 0.224, 0.225)
            self.transform = vision_transforms.Compose([
                vision_transforms.Resize(size, interpolation=3),
                vision_transforms.CenterCrop(center_crop_size),
                vision_transforms.ToTensor(),
                vision_transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) != 3 else x),
                vision_transforms.Normalize(dino_mean, dino_std),
            ])

        else:
            self.transform = vision_transforms.Compose([vision_transforms.ToTensor()])

    @property
    def train_transform(self):
        return self.transform

    @property
    def query_transform(self):
        return self.transform

    @property
    def eval_transform(self):
        return self.transform


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, inputs):
        outputs = self.backbone(inputs)
        return outputs.pooler_output


class CLIPTransform():
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __call__(self, x):
        output = self.processor(images=x, return_tensors='pt')
        return output['pixel_values'].squeeze()


class DINOv3(nn.Module):
    def __init__(self):
        super().__init__()
        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", None)
        pretrained_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        self.backbone = AutoModel.from_pretrained(pretrained_model_name, token=hf_token)

    def forward(self, inputs):
        outputs = self.backbone(inputs)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return cls_token


class DINOv3Transform():
    def __init__(self):
        pretrained_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)

    def __call__(self, x):
        output = self.processor(images=x, return_tensors='pt')
        return output['pixel_values'].squeeze()


class EAT(nn.Module):
    def __init__(self):
        super().__init__()
        model_id = 'worstchan/EAT-base_epoch30_pretrain'
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()

    def forward(self, x):
        features = self.model.extract_features(x) #normalization etc. is done in the model
        cls_token = features[:,0]
        return cls_token


class ESC():
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        config.DOWNLOADED_DATASETS_PATH = root
        self.ds = load_dataset('ashraq/esc50', download_mode="reuse_dataset_if_exists", split='train')
        self.ds = self.ds.train_test_split(train_size=0.8)
        self.ds = self.ds['train'] if split == 'train' else self.ds['test']

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        data = self.ds[index]
        wav = torch.from_numpy(data['audio']['array']).float()
        lbl = data['target']

        if self.transform:
            wav = self.transform(wav, orig_sample_rate=44_100)
        if self.target_transform:
            lbl = self.target_transform(lbl)
        return wav, lbl


class EATTransform():
    def __init__(self, target_length=1024, sample_rate=16000, norm_mean=-4.268, norm_std=4.569, augment=False):
        """
        Audio transforms for EAT (Efficient Audio Transformer) models

        Args:
            target_length (int): Target number of frames for mel-spectrogram (1024 for 10s audio)
            sample_rate (int): Target sample rate (16000 for EAT)
            norm_mean (float): Normalization mean for mel-spectrogram
            norm_std (float): Normalization std for mel-spectrogram
        """
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.augment = augment

    def __call__(self, audio_input, **kwargs):
        if isinstance(audio_input, str):
            # File path - load audio
            wav, sr = sf.read(audio_input)
            waveform = torch.tensor(wav).float()
        elif isinstance(audio_input, tuple):
            # Pre-loaded audio data as (waveform, sample_rate)
            waveform, sr = audio_input
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform).float()
        else:
            # If already a tensor
            waveform = audio_input
            sr = kwargs['orig_sample_rate']

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Normalize waveform
        waveform = waveform - waveform.mean()

        # Convert to mel-spectrogram
        mel = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0),
            htk_compat=True,
            sample_frequency=self.sample_rate,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10
        ).unsqueeze(0)

        # Pad or truncate to target length
        n_frames = mel.shape[1]
        if n_frames < self.target_length:
            mel = torch.nn.ZeroPad2d((0, 0, 0, self.target_length - n_frames))(mel)
        else:
            mel = mel[:, :self.target_length, :]

        # Normalize mel-spectrogram
        mel = (mel - self.norm_mean) / (self.norm_std * 2)  # 1, T, F
        # mel = mel.unsqueeze(0)  # shape: [1, 1, T, F]

        return mel
