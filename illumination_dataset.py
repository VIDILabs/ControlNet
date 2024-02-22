import json
import cv2
import random
import numpy as np

from torch.utils.data import Dataset


class IlluminationDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/illum/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/illum/' + source_filename)
        target = cv2.imread('./training/illum/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if (source.shape[0] > target.shape[0]):
            source = cv2.resize(source, (target.shape[0], target.shape[1]))
        if (target.shape[0] > source.shape[0]):
            target = cv2.resize(target, (source.shape[0], source.shape[1]))

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class IlluminationGridDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/illum_grid/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/illum_grid/' + source_filename)
        target = cv2.imread('./training/illum_grid/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if (source.shape[0] > target.shape[0]):
            source = cv2.resize(source, (target.shape[0], target.shape[1]))
        if (target.shape[0] > source.shape[0]):
            target = cv2.resize(target, (source.shape[0], source.shape[1]))

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class IlluminationProceduralGridDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/illum/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sources = []
        targets = []
        prompt = ''
        for i in range(4):
            offset = random.randint(0, 150)
            # offset = 30
            item = self.data[idx]

            source_filename = item['source']
            source_filename_no = int(source_filename.split('/')[-1]
                                     .split('.')[0])
            source_filename = f"{''.join(source_filename.split('/')[:-1])}/{(source_filename_no + offset)%151:04d}.{source_filename.split('.')[-1]}"
            target_filename = item['target']
            target_filename_no = int(target_filename.split('/')[-1]
                                     .split('.')[0])
            target_filename = f"{''.join(target_filename.split('/')[:-1])}/{(target_filename_no + offset)%151:04d}.{target_filename.split('.')[-1]}"
            prompt = item['prompt']

            source = cv2.imread('./training/illum/' + source_filename)
            source = cv2.resize(source, (source.shape[0] // 2, source.shape[1] // 2))
            target = cv2.imread('./training/illum/' + target_filename)
            target = cv2.resize(target, (target.shape[0] // 2, target.shape[1] // 2))

            # Do not forget that OpenCV read images in BGR order.
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

            # Normalize source images to [0, 1].
            source = source.astype(np.float32) / 255.0

            # Normalize target images to [-1, 1].
            target = (target.astype(np.float32) / 127.5) - 1.0

            # Append to lists
            sources.append(source)
            targets.append(target)

        source0 = np.concatenate(sources[:2], axis=0)
        source1 = np.concatenate(sources[2:], axis=0)
        source = np.concatenate([source0, source1], axis=1)
        target0 = np.concatenate(targets[:2], axis=0)
        target1 = np.concatenate(targets[2:], axis=0)
        target = np.concatenate([target0, target1], axis=1)

        return dict(jpg=target, txt=prompt, hint=source)
