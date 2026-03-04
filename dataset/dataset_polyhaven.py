# Dataset loader for polyhaven_lvsm format
# Metadata JSON with fxfycxcy intrinsics and w2c in OpenCV convention

import os
import json

import torch
import numpy as np

from render import util

from dataset import Dataset


def _load_img(path):
    img = util.load_image_raw(path)
    if img.dtype != np.float32:  # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img


# OpenCV (x-right, y-down, z-forward) -> OpenGL (x-right, y-up, z-backward)
OPENCV_TO_OPENGL = torch.tensor([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
], dtype=torch.float32)


class DatasetPolyhaven(Dataset):
    def __init__(self, metadata_path, FLAGS, examples=None, validate=False, val_skip=8):
        self.FLAGS = FLAGS
        self.examples = examples

        with open(metadata_path, 'r') as f:
            self.cfg = json.load(f)

        self.base_dir = os.path.dirname(os.path.dirname(metadata_path))
        self.scene_name = self.cfg.get('scene_name', os.path.splitext(os.path.basename(metadata_path))[0])

        all_frames = self.cfg['frames']

        val_indices = set(range(0, len(all_frames), val_skip))
        if validate:
            self.frames = [all_frames[i] for i in range(len(all_frames)) if i in val_indices]
        else:
            self.frames = [all_frames[i] for i in range(len(all_frames)) if i not in val_indices]

        self.n_images = len(self.frames)

        first_img_path = self._resolve_image_path(self.frames[0])
        first_img = _load_img(first_img_path)
        self.resolution = first_img.shape[0:2]
        self.aspect = self.resolution[1] / self.resolution[0]

        fx, fy, cx, cy = self.frames[0]['fxfycxcy']
        H = self.resolution[0]
        self.fovy = 2.0 * np.arctan(0.5 * H / fy)

        print("DatasetPolyhaven [%s]: %d images (%s), resolution [%d, %d], fovy=%.4f rad" % (
            self.scene_name, self.n_images, "val" if validate else "train",
            self.resolution[0], self.resolution[1], self.fovy))

        if self.FLAGS.pre_load:
            self.preloaded_data = []
            for i in range(self.n_images):
                self.preloaded_data.append(self._parse_frame(i))

    def _resolve_image_path(self, frame):
        path = frame['image_path']
        if os.path.isfile(path):
            return path
        # Fallback: reconstruct from base_dir/images/scene_name/filename
        filename = os.path.basename(path)
        alt_path = os.path.join(self.base_dir, 'images', self.scene_name, filename)
        if os.path.isfile(alt_path):
            return alt_path
        raise FileNotFoundError(
            "Image not found at '%s' or '%s'" % (path, alt_path))

    def _parse_frame(self, idx):
        frame = self.frames[idx]

        img_path = self._resolve_image_path(frame)
        img = _load_img(img_path)

        fx, fy, cx, cy = frame['fxfycxcy']
        H = self.resolution[0]
        fovy = 2.0 * np.arctan(0.5 * H / fy)
        proj = util.perspective(fovy, self.aspect,
                                self.FLAGS.cam_near_far[0],
                                self.FLAGS.cam_near_far[1])

        w2c = torch.tensor(frame['w2c'], dtype=torch.float32)
        mv = OPENCV_TO_OPENGL @ w2c

        campos = torch.linalg.inv(mv)[:3, 3]
        mvp = proj @ mv

        return img[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...]

    def getMesh(self):
        return None

    def __len__(self):
        return self.n_images if self.examples is None else self.examples

    def __getitem__(self, itr):
        if self.FLAGS.pre_load:
            img, mv, mvp, campos = self.preloaded_data[itr % self.n_images]
        else:
            img, mv, mvp, campos = self._parse_frame(itr % self.n_images)

        return {
            'mv': mv,
            'mvp': mvp,
            'campos': campos,
            'resolution': self.FLAGS.train_res,
            'spp': self.FLAGS.spp,
            'img': img,
        }
