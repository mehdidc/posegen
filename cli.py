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
from torch.distributions import Normal
import cv2


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


def predict_pose(video_path, *, out="data", device="cuda"):
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
    out = os.path.join(dest, name + ".pkl")
    if os.path.exists(out):
        return
    frames = []
    if not os.path.exists(dest):
        os.makedirs(dest)
    print(len(processors))
    for image_i, (image_paths, image_tensors, processed_images_cpu) in enumerate(
        data_loader
    ):
        images = image_tensors.permute(0, 2, 3, 1)
        processed_images = processed_images_cpu.to(args.device, non_blocking=True)
        fields_batch = processors[0].fields(processed_images)
        for image_path, image, processed_image_cpu, fields in zip(
            image_paths, images, processed_images_cpu, fields_batch
        ):
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
    def __init__(self, folder, max_len=100, max_examples=None):
        self.folder = folder
        self.videos = glob(os.path.join(folder, "**", "*.pkl"))
        self.tracks = []
        self.classes = []
        self.max_len = max_len
        self.max_examples = max_examples
        self._prepare()
        self.classes_unique = sorted(list(set(self.classes)))

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
                track_length_min=self.max_len,
            )
            class_name = os.path.basename(os.path.dirname(vid))
            for track in tracks:
                track = sorted(track, key=lambda obj: obj.frame_id)
                self.tracks.append(track)
                self.classes.append(class_name)
                if self.max_examples and len(self.tracks) == self.max_examples:
                    return

    def __getitem__(self, idx):
        track = self.tracks[idx]
        class_ = self.classes[idx]
        kps = []
        for obj in track:
            kps.append(obj.kp)
        kps = np.array(kps)
        kps = kps[:, :, 0:2]
        mode = "sample"
        if mode == "pad":
            L = kps.shape[0]
            first = np.zeros((1, kps.shape[1], kps.shape[2]))
            last = np.zeros(
                (max(self.max_len - kps.shape[0] - 1, 0), kps.shape[1], kps.shape[2])
            )
            kps = np.concatenate((first, kps, last), axis=0)
            kps = kps[0 : self.max_len]
            kps -= kps.min()
            kps /= kps.max()
            kps = torch.from_numpy(kps).float()
            mask = torch.zeros_like(kps).float()
            mask[0 : L + 1] = 1
            kps[L + 1 :] = kps[L : L + 1]
        elif mode == "sample":
            assert len(kps) >= self.max_len
            start = np.random.randint(0, len(kps) - self.max_len + 1)
            end = start + self.max_len - 1
            kps = kps[start:end]
            first = np.zeros((1, kps.shape[1], kps.shape[2]))
            kps = np.concatenate((first, kps), axis=0)

            # kps -= kps.min()
            # kps /= kps.max()

            kps = torch.from_numpy(kps).float()
            mask = torch.ones_like(kps).float()
        return kps, mask, self.classes_unique.index(class_)

    def __len__(self):
        return len(self.tracks)


class Model(nn.Module):
    def __init__(
        self, input_size=17 * 3, hidden_size=128, output_size=17 * 3, num_layers=1
    ):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SELU(),
            nn.Linear(hidden_size, output_size),
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, x, h=None):
        b = x.shape[0]
        t = x.shape[1]
        o, h = self.rnn(x, h)
        o = o.contiguous()
        o = o.view(b * t, self.hidden_size)
        o = self.out(o)
        o = o.view(b, t, self.output_size)
        return o, h


def mdn_loss_function(output, target, nb_components=1, nb_outputs=17 * 3):
    B = output.size(0)
    T = output.size(1)
    M = nb_components
    F = nb_outputs
    EPSILON = 1e-7
    o = output
    t = target
    o = o.view(B, T, M, F * 2 + 1)
    out_mu = o[:, :, :, 0:F]

    out_sigma = o[:, :, :, F : 2 * F]
    out_sigma = torch.exp(out_sigma)

    out_pi = o[:, :, :, 2 * F : 2 * F + 1]
    out_pi = nn.Softmax(dim=2)(out_pi)

    result = Normal(loc=out_mu, scale=out_sigma)
    target = target.view(B, T, 1, -1)
    result = torch.exp(result.log_prob(target))
    result = torch.sum(result * out_pi, dim=2)
    result = -torch.log(EPSILON + result)
    return result


def sample_pi(pi):
    # shape (T, nb_components, 1)
    T = pi.shape[0]
    M = pi.shape[1]
    pi = pi.view(T, M)
    pi_sample = torch.multinomial(pi, 1)
    pi[:] = 0
    pi.scatter_(1, pi_sample, 1)
    pi = pi.view(T, M, 1)
    return pi


def sample_pi_batch(pi):
    # shape (B, T, nb_components, 1)
    B = pi.shape[0]
    T = pi.shape[1]
    M = pi.shape[2]
    pi = pi.view(B * T, M)
    pi_sample = torch.multinomial(pi, 1)
    pi[:] = 0
    pi.scatter_(1, pi_sample, 1)
    pi = pi.view(B, T, M, 1)
    return pi


def train(*, data="data", device="cpu", epochs=1000):
    dataset = KeypointDataset(data, max_len=50, max_examples=1000)
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    nb_components = 10
    nb_keypoints = 17 * 2
    model = Model(
        input_size=nb_keypoints,
        hidden_size=128,
        output_size=nb_components * (2 * nb_keypoints + 1),
        num_layers=1,
    )
    model = model.to(device)
    opt = Adam(model.parameters(), lr=1e-3)
    n_iter = 0
    keypoint_painter = show.KeypointPainter(show_box=False)
    for epoch in range(epochs):
        for X, M, Y in dataloader:
            X = X.to(device)
            X = X.view(X.size(0), X.size(1), -1)
            M = M.view(M.size(0), M.size(1), -1)
            M = M[:, 0:-1, :]
            M = M.to(device)

            I = X[:, 0:-1, :]
            T = X[:, 1:, :]

            O, _ = model(I)
            opt.zero_grad()
            mdn_loss = mdn_loss_function(
                O, T, nb_components=nb_components, nb_outputs=nb_keypoints
            )
            # loss = (mdn_loss * M).mean()
            loss = mdn_loss.mean()
            loss.backward()
            opt.step()
            if n_iter % 1000 == 0:
                print(n_iter, -loss.item())
            if n_iter % 10000 == 0:
                sig_mult = 1 / 10
                # Pred
                O = O.detach().cpu()
                B = O.shape[0]
                O = O.reshape(
                    (O.shape[0], O.shape[1], nb_components, (2 * nb_keypoints + 1))
                )
                mu = O[:, :, :, 0:nb_keypoints]
                sig = O[:, :, :, nb_keypoints : nb_keypoints * 2]
                sig = torch.exp(sig) * sig_mult
                pi = O[:, :, :, nb_keypoints * 2 : nb_keypoints * 2 + 1]
                pi = nn.Softmax(dim=2)(pi)
                # pi = sample_pi_batch(pi)
                O = (torch.normal(mu, sig) * pi).sum(dim=2)
                O = O.numpy()
                O = np.clip(O, 0, 1)
                O = O * 500
                nb_steps = O.shape[1]
                O = O.reshape((B, nb_steps, 17, 2))
                image = np.zeros((500, 500, 3))
                for i in range(B):
                    folder = f"log/pred/{i:05d}"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    for t in range(nb_steps):
                        kp = O[i, t]
                        image[:] = 0
                        draw(kp, image)
                        cv2.imwrite(f"{folder}/{t:05d}.png", image)
                # Gen
                B = 20
                O = torch.zeros(B, nb_steps, nb_keypoints)
                o = torch.zeros(B, 1, nb_keypoints)
                o = o.to(device)
                x = o
                for t in range(nb_steps):
                    o, h = model(x)
                    o = o.view(B, nb_components, (2 * nb_keypoints + 1))
                    mu = o[:, :, 0:nb_keypoints]
                    sig = o[:, :, nb_keypoints : nb_keypoints * 2]
                    sig = torch.exp(sig) * sig_mult
                    pi = o[:, :, nb_keypoints * 2 : nb_keypoints * 2 + 1]
                    pi = nn.Softmax(dim=1)(pi)
                    pi = sample_pi(pi)
                    o = (torch.normal(mu, sig) * pi).sum(dim=1)
                    o = o.view(B, 1, nb_keypoints)
                    x = o.detach()
                    O[:, t : t + 1] = x.cpu()
                O = O.cpu().numpy()
                O = np.clip(O, 0, 1)
                O = O * 500
                O = O.reshape((B, nb_steps, 17, 2))
                image = np.zeros((500, 500, 3))
                for i in range(B):
                    folder = f"log/gen/{i:05d}"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    for t in range(nb_steps):
                        kp = O[i, t]
                        image[:] = 0
                        draw(kp, image)
                        cv2.imwrite(f"{folder}/{t:05d}.png", image)
            n_iter += 1


def draw(kp, image):
    COCO_PERSON_SKELETON = [
        [16, 14],
        [14, 12],
        [17, 15],
        [15, 13],
        [12, 13],
        [6, 12],
        [7, 13],
        [6, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [9, 11],
        [2, 3],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 7],
    ]
    col = (255, 255, 255)
    min_x = kp[:, 0].min()
    min_y = kp[:, 1].min()
    max_x = kp[:, 0].max()
    max_y = kp[:, 1].max()
    area = (max_x - min_x) * (max_y - min_y)
    for a, b in COCO_PERSON_SKELETON:
        x1 = kp[a - 1, 0]
        y1 = kp[a - 1, 1]
        x2 = kp[b - 1, 0]
        y2 = kp[b - 1, 1]
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.line(image, (x1, y1), (x2, y2), color=col, thickness=2)
        if a == 2 and b == 3:
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)
            cv2.circle(
                image, (cx, cy), radius=int(area * 0.002), color=col, thickness=-1
            )


if __name__ == "__main__":
    run([train, predict_pose, predict_pose_videos])
