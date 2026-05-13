from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class UprightBottleImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            use_state=True,
            ):
        super().__init__()
        keys = ['image1', 'image2', 'state', 'action'] if use_state \
            else ['image1', 'image2', 'action']
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=keys)
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_state = use_state

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
        }
        if self.use_state:
            data['state'] = self.replay_buffer['state']
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image1'] = get_image_range_normalizer()
        normalizer['image2'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        image1 = np.moveaxis(sample['image1'], -1, 1).astype(np.float32) / 255.0
        image2 = np.moveaxis(sample['image2'], -1, 1).astype(np.float32) / 255.0
        action = sample['action'].astype(np.float32)
        obs = {
            'image1': image1,  # T, 3, H, W
            'image2': image2,  # T, 3, H, W
        }
        if self.use_state:
            obs['state'] = sample['state'].astype(np.float32)  # T, 7
        return {
            'obs': obs,
            'action': action,    # T, 7
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)
