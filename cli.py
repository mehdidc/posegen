import os
from glob import glob
import numpy as np
from clize import run
from openpifpaf import datasets, decoder, show, transforms
from openpifpaf.network import nets
import torch
from PIL import Image
import skvideo.io
import torchvision
import json
from collections import defaultdict
from joblib import dump

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
        dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2
    )
    show.KeypointPainter(show_box=False)
    skeleton_painter = show.KeypointPainter(
        show_box=False, color_connections=True, markersize=1, linewidth=6
    )
    i = 0
    name = os.path.basename(video_path)
    class_name = os.path.basename(os.path.dirname(video_path))
    dest = os.path.join(out, class_name)
    frames = []
    if not os.path.exists(dest):
        os.makedirs(dest)
    for image_i, (image_paths, image_tensors, processed_images_cpu) in enumerate(data_loader):
        images = image_tensors.permute(0, 2, 3, 1)
        processed_images = processed_images_cpu.to(args.device, non_blocking=True)
        fields_batch = processors[0].fields(processed_images)
        for image_path, image, processed_image_cpu, fields in zip(image_paths, images, processed_images_cpu, fields_batch):
            processors[0].set_cpu_image(image, processed_image_cpu)
            for processor in processors:
                keypoint_sets, scores = processor.keypoint_sets(fields)
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
    out = os.path.join(dest, name+".pkl")
    dump(frames, out)

def predict_pose_videos(pattern):
    for video_path in glob(pattern):
        predict_pose(video_path)

if __name__ == "__main__":
    run([predict_pose, predict_pose_videos])
