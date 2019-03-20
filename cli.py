from clize import run
from openpifpaf import datasets, decoder, show
from openpifpaf.network import nets

class Args(object):
    pass

def predict_pose(video_path):
    args = Args()
    args.checkpoint = None
    args.basenet = None
    args.dilation = None
    args.dilation_end  = None
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
    device = "cpu"
    model = model.to(device)
    processors = decoder.factory(args, model)


if __name__ == "__main__":
    run([predict_pose])
