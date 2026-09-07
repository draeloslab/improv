"""
Minimal JARVIS-MoCap EfficientTrack inference helpers for the live Processor
actor (see processor_jarvis.py). Trimmed from
hand-tracking-notebooks-jake-2026-08-17/jarvis/jarvis_utils.py down to just
the pieces needed at inference time -- no dataset conversion or self-labeling
code, which don't belong in the live camera pipeline.

Requires the `jarvis-mocap` package (JARVIS-HybridNet, installed editable)
importable in whatever conda env runs this Processor actor. In the notebook
project that lives in its own `jarvis-test` env, separate from the
`deeplabcut`/torch env processor.py's DLC pipeline runs in -- running
ProcessorJarvis live means installing jarvis-mocap (and its deps: yacs,
imgaug, streamlit) into that same env, or giving this actor its own env and
process. See the note at the top of processor_jarvis.py.
"""
import torch
from torchvision import transforms
from yacs.config import CfgNode as CN

from jarvis.efficienttrack.efficienttrack import EfficientTrack

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_cfg(model_size="medium", center_image_size=320, bbox_size=512, num_joints=11):
    cfg = CN()
    cfg.DATASET = CN()
    cfg.DATASET.MEAN = [0.485, 0.456, 0.406]
    cfg.DATASET.STD = [0.229, 0.224, 0.225]
    cfg.CENTERDETECT = CN()
    cfg.CENTERDETECT.IMAGE_SIZE = center_image_size
    cfg.CENTERDETECT.MODEL_SIZE = model_size
    cfg.CENTERDETECT.NUM_JOINTS = 1
    cfg.KEYPOINTDETECT = CN()
    cfg.KEYPOINTDETECT.MODEL_SIZE = model_size
    cfg.KEYPOINTDETECT.NUM_JOINTS = num_joints
    cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE = bbox_size
    return cfg


def load_predictor(cfg, weights_center, weights_keypoint):
    """Loads CenterDetect + KeypointDetect models in inference mode."""
    centerDetect = EfficientTrack("CenterDetectInference", cfg, weights_center).model
    keypointDetect = EfficientTrack("KeypointDetectInference", cfg, weights_keypoint).model
    return centerDetect, keypointDetect


def run_inference(centerDetect, keypointDetect, cfg, img_bgr, center_threshold=40):
    """
    Runs the 2-stage EfficientTrack pipeline on a single BGR frame.
    Returns (points2D [num_joints,2], confidences [num_joints], center_xy) or
    (None, None, None) if CenterDetect found nothing above center_threshold.
    """
    transform_mean = torch.tensor(cfg.DATASET.MEAN, device=DEVICE).view(3, 1, 1)
    transform_std = torch.tensor(cfg.DATASET.STD, device=DEVICE).view(3, 1, 1)
    bbox_hw = int(cfg.KEYPOINTDETECT.BOUNDING_BOX_SIZE / 2)
    center_img_size = int(cfg.CENTERDETECT.IMAGE_SIZE)

    img = torch.from_numpy(img_bgr).to(DEVICE).float().permute(2, 0, 1)[[2, 1, 0]] / 255.
    img = img.unsqueeze(0)
    img_size = torch.tensor([img.shape[3], img.shape[2]], device=DEVICE)
    ds = torch.tensor([img_size[0] / float(center_img_size),
                        img_size[1] / float(center_img_size)], device=DEVICE).float()
    imgr = transforms.functional.resize(img, [center_img_size, center_img_size])
    imgr = (imgr - transform_mean) / transform_std

    with torch.no_grad():
        outputs = centerDetect(imgr)
        hm = outputs[1].view(outputs[1].shape[0], outputs[1].shape[1], -1)
        m = hm.argmax(2).view(hm.shape[0], hm.shape[1], 1)
        maxval = hm.gather(2, m).squeeze()
        if maxval <= center_threshold:
            return None, None, None

        c = torch.cat((m % outputs[1].shape[2], m // outputs[1].shape[3]), dim=2).squeeze() * ds * 2
        c = c.int()
        c[0] = torch.clamp(c[0], bbox_hw, img_size[0] - bbox_hw - 1)
        c[1] = torch.clamp(c[1], bbox_hw, img_size[1] - bbox_hw - 1)

        crop = img[:, :, c[1]-bbox_hw:c[1]+bbox_hw, c[0]-bbox_hw:c[0]+bbox_hw]
        crop = (crop - transform_mean) / transform_std

        outkp = keypointDetect(crop)
        hm2 = outkp[1].view(outkp[1].shape[0], outkp[1].shape[1], -1)
        m2 = hm2.argmax(2).view(hm2.shape[0], hm2.shape[1], 1)
        pts = torch.cat((m2 % outkp[1].shape[2], m2 // outkp[1].shape[3]), dim=2).squeeze() * 2
        conf = hm2.gather(2, m2).squeeze()
        conf = torch.clamp(conf, max=255.) / 255.
        pts = pts + c - bbox_hw

        return pts.cpu().numpy(), conf.cpu().numpy(), c.cpu().numpy()
