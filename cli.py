import os
from glob import glob
import numpy as np
from clize import run
from openpifpaf import datasets, decoder, show, transforms
from openpifpaf.network import nets
import torch
import torch.nn as nn
from PIL import Image
import skvideo.io
import torchvision
import json
from collections import defaultdict
from joblib import dump, load
from scoreml.detection import Thing, Box
from torch.optim import Adam
from iou_tracker import iou_tracker


class Args(object):
    pass


class ImageList(torch.utils.data.Dataset):
    def __init__(self, images, image_transform=None):
        self.images = images
        self.image_transform = image_transform or transforms.image_transform

    def __getitem__(self, index):
        image = self.images[index]
        image = Image.fromarray(image)
        original_image = torchvision.transforms.functional.to_tensor(image)
        image = self.image_transform(image)
        return "", original_image, image

    def __len__(self):
        return len(self.images)


def predict_pose(video_path, *, out="out", device="cuda"):
    args = Args()
    args.checkpoint = None
    args.basenet = None
    args.dilation = None
    args.dilation_end = None
    args.keypoint_threshold = None
    args.force_complete_pose = True
    args.debug_pif_indices = []
    args.debug_paf_indices = []
    args.seed_threshold = 0.2
    args.connection_method = "max"
    args.fixed_b = None
    args.pif_fixed_scale = None
    args.profile_decoder = False
    args.instance_threshold = 0.0
    model, _ = nets.factory(args)
    args.device = device
    args.show = False
    args.figure_width = 10
    args.dpi_factor = 1.0
    
    model = model.to(args.device)
    processors = decoder.factory(args, model)

    vid = skvideo.io.vread(video_path)

    dataset = ImageList(vid)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=2
    )
    show.KeypointPainter(show_box=False)
    skeleton_painter = show.KeypointPainter(
        show_box=False, color_connections=True, markersize=1, linewidth=6
    )
    i = 0
    name = os.path.basename(video_path)
    class_name = os.path.basename(os.path.dirname(video_path))
    dest = os.path.join(out, class_name)
    out = os.path.join(dest, name+".pkl")
    if os.path.exists(out):
        return
    frames = []
    if not os.path.exists(dest):
        os.makedirs(dest)
    print(len(processors))
    for image_i, (image_paths, image_tensors, processed_images_cpu) in enumerate(data_loader):
        images = image_tensors.permute(0, 2, 3, 1)
        processed_images = processed_images_cpu.to(args.device, non_blocking=True)
        fields_batch = processors[0].fields(processed_images)
        for image_path, image, processed_image_cpu, fields in zip(image_paths, images, processed_images_cpu, fields_batch):
            processors[0].set_cpu_image(image, processed_image_cpu)
            keypoint_sets, scores = processors[0].keypoint_sets(fields)
            frames.append((keypoint_sets, scores))
            # with show.image_canvas(
                # image,
                # output_path + ".keypoints.png",
                # show=args.show,
                # fig_width=args.figure_width,
                # dpi_factor=args.dpi_factor,
            # ) as ax:
                # # show.white_screen(ax, alpha=0.5)
                # keypoint_painter.keypoints(ax, keypoint_sets)
            # with show.image_canvas(
                # image,
                # output_path + ".skeleton.png",
                # show=args.show,
                # fig_width=args.figure_width,
                # dpi_factor=args.dpi_factor,
            # ) as ax:
                # skeleton_painter.keypoints(ax, keypoint_sets, scores=scores)
    dump((vid.shape, frames), out)

def predict_pose_videos(pattern):
    for video_path in glob(pattern):
        predict_pose(video_path)


class KeypointDataset:

    def __init__(self, folder, max_len=100):
        self.folder = folder
        self.videos = glob(os.path.join(folder, "**", "*.pkl"))
        self.videos = self.videos[0:100]
        self.tracks = []
        self.classes = []
        self.max_len = max_len
        self._prepare()
    
    def _prepare(self):
        for vid in self.videos:
            shape, frames = load(vid)
            height, width = shape[1], shape[2]
            objs_frames = []
            for keypoint_sets, scores in frames:
                objs = []
                for kp, score in zip(keypoint_sets, scores):
                    kp = kp.copy()
                    kp[:, 0] /= width
                    kp[:, 1] /= height
                    xmin = kp[:, 0].min()
                    ymin = kp[:, 1].min()
                    xmax = kp[:, 0].max()
                    ymax = kp[:, 1].max()
                    box = Box(xmin, ymin, xmax - xmin, ymax - ymin)
                    thing = Thing(box, "person", confidence=score)
                    thing.kp = kp
                    objs.append(thing)
                objs_frames.append(objs)
            tracks = iou_tracker(
                objs_frames,
                obj_confidence_threshold=0.1,
                track_confidence_threshold=0.5,
                iou_threshold=0.5,
                track_length_min=10

            )
            class_name = os.path.basename(os.path.dirname(vid))
            for track in tracks:
                track = sorted(track, key=lambda obj:obj.frame_id)
                self.tracks.append(track)
                self.classes.append(class_name)
        self.classes_unique =  sorted(list(set(self.classes)))


    def __getitem__(self, idx):
        track = self.tracks[idx]
        class_ = self.classes[idx]
        kps = []
        for obj in track:
            kps.append(obj.kp)
        kps = np.array(kps)
        L = kps.shape[0]
        z = np.zeros((max(self.max_len - kps.shape[0], 0), kps.shape[1], kps.shape[2]))
        kps = np.concatenate((kps, z), axis=0)
        kps = kps[0:self.max_len]
        kps = torch.from_numpy(kps).float()
        mask = torch.zeros_like(kps).float()
        mask[0:L] = 1
        return kps, mask, self.classes_unique.index(class_)
    
    def __len__(self):
        return len(self.tracks)

class Model(nn.Module):

    def __init__(self, input_size=17*3, hidden_size=128, output_size=17*3):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size,
            hidden_size, 
            batch_first=True, 
            num_layers=1
        )
        self.out = nn.Linear(hidden_size, output_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
    
    def forward(self, x):
        b = x.shape[0]
        t = x.shape[1]

        h, _ = self.rnn(x)
        h = h.contiguous()
        h = h.view(b*t, self.hidden_size)
        o = self.out(h)
        o = o.view(b, t, self.output_size)
        return o


def train(folder, device="cpu", epochs=1000):
    dataset = KeypointDataset(folder, max_len=50)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True)
    model = Model()
    model = model.to(device)
    opt = Adam(model.parameters(), lr=1e-3)
    n_iter = 0
    keypoint_painter = show.KeypointPainter(show_box=True)
    for epoch in range(epochs):
        for X, M, Y in dataloader:
            X = X.to(device)
            X = X.view(X.size(0), X.size(1), -1)

            M = M.view(M.size(0), M.size(1), -1)
            M = M.to(device)

            I = X[:, 0:-1, :]
            O = X[:, 1:, :]
            P = model(I)
            opt.zero_grad()
            loss = (((P - O) ** 2) * M[:, 1:]).mean()
            loss.backward()
            opt.step()
            print(n_iter, loss.item())
            if n_iter % 100 == 0:
                P = P.detach().cpu().numpy()
                P = P.reshape((P.shape[0], P.shape[1], 17, 3))
                P = np.clip(P, 0, 1)
                P = P * 500
                p = P[0]
                for i in range(len(p)):
                    image = np.zeros((500, 500, 3))
                    with show.image_canvas(image, f"log/{i:05d}.png") as ax:
                        keypoint_painter.keypoints(ax, p[i:i+1])
            n_iter += 1


def viz(folder):
    dataset = KeypointDataset(folder)
    for i in range(len(dataset)):
        if len(dataset.tracks[i]) > 100:
            break
    X, Y = dataset[i]
    X = X.numpy() * 500
    keypoint_painter = show.KeypointPainter(show_box=True)
    for i in range(len(X)):
        image = np.zeros((500, 500, 3))
        with show.image_canvas(image, f"{i:05d}.png") as ax:
            keypoint_painter.keypoints(ax, X[i:i+1])


if __name__ == "__main__":
    run([train, predict_pose, predict_pose_videos, viz])
