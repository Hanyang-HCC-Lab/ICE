import colorsys
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from torch.nn.functional import interpolate
from tqdm import tqdm

from utils import get_voc_dataset, get_model, parse_args
from timm.models import create_model
from models_ICE import VisionTransformer#_1mlp
from functools import partial

import torch
import torch.nn as nn
from functools import partial


from timm.models.vision_transformer import  _cfg,Attention
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_

def get_attention_masks(args, image, model):
    # make the image divisible by the patch size
    w, h = image.shape[2] - image.shape[2] % args.patch_size, image.shape[3] - image.shape[3] % args.patch_size
    image = image[:, :w, :h]
    w_featmap = image.shape[-2] // args.patch_size
    h_featmap = image.shape[-1] // args.patch_size

    attentions = model.forward_selfattention(image.cuda())
    nh = attentions.shape[1]

    # we keep only the output patch attention
    if args.is_dist:
        if args.use_shape:
            attentions = attentions[0, :, 1, 2:].reshape(nh, -1)  # use distillation token attention
        else:
            attentions = attentions[0, :, 0, 2:].reshape(nh, -1)  # use class token attention
    else:
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cum_val = torch.cumsum(val, dim=1)
    th_attn = cum_val > (1 - args.threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0]

    return th_attn


def get_per_sample_jaccard(pred, target):
    jac = 0
    object_count = 0
    for mask_idx in torch.unique(target):
        if mask_idx in [0, 255]:  # ignore index
            continue
        cur_mask = target == mask_idx
        intersection = (cur_mask * pred) * (cur_mask != 255)  # handle void labels
        intersection = torch.sum(intersection, dim=[1, 2])  # handle void labels
        union = ((cur_mask + pred) > 0) * (cur_mask != 255)
        union = torch.sum(union, dim=[1, 2])
        jac_all = intersection / union
        jac += jac_all.max().item()
        object_count += 1
    return jac / object_count


def run_eval(args, data_loader, model, device):
    model.to(device)
    model.eval()
    total_jac = 0
    image_count = 0
    for idx, (sample, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        sample, target = sample.to(device), target.to(device)
        _, attention_mask ,_ = model(sample)
        attention_mask = torch.argmax(attention_mask.view(32,32,1001),axis = 2)
        attention_mask = attention_mask.masked_fill(attention_mask != 1000,1) # class patch
        attention_mask = attention_mask.masked_fill(attention_mask == 1000,0) # background patch
        attention_mask = attention_mask.repeat_interleave(16,dim=0)
        attention_mask = attention_mask.repeat_interleave(16,dim=1)
        
        jac_val = get_per_sample_jaccard(attention_mask, target)
        total_jac += jac_val
        image_count += 1
    return total_jac / image_count


def apply_mask_last(image, mask, color=(0.0, 0.0, 1.0), alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    image = image.permute(1, 2, 0).cpu().numpy()
    mask = mask.cpu().numpy()

    plt.ioff()
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]

    # Generate random colors

    def random_colors(N, bright=True):
        """
        Generate random colors.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        return colors

    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = (image * 255).astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            pass
            # _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask_last(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    plt.close(fig)


def generate_images_per_model(args, model, device):

    model.to(device)
    model.eval()

    samples = []
    for im_name in tqdm(os.listdir(args.test_dir)):
        im_path = f"{args.test_dir}/{im_name}"
        img = Image.open(f"{im_path}").resize((512, 512))
        img = torchvision.transforms.functional.to_tensor(img)
        if img.shape[0] == 1:
            img = torch.cat([img, img, img], dim=0)
        samples.append(img)
    samples = torch.stack(samples, 0).to(device)

    attention_masks = []
    for sample in samples:
        attention_masks.append(get_attention_masks(args, sample.unsqueeze(0), model))

    os.makedirs(f"{args.save_path}", exist_ok=True)
    os.makedirs(f"{args.save_path}/{args.model_name}_{args.threshold}", exist_ok=True)
    for idx, (sample, mask) in enumerate(zip(samples, attention_masks)):
        for head_idx, mask_h in enumerate(mask):
            f_name = f"{args.save_path}/{args.model_name}_{args.threshold}/im_{idx:03d}_{head_idx}.png"
            display_instances(sample, mask_h, fname=f_name)


if __name__ == '__main__':
    opt = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_dataset, test_data_loader = get_voc_dataset()

    opt.is_dist = "dist" in opt.model_name
    if opt.use_shape:
        assert opt.is_dist, "shape token only present in distilled models"

    if opt.rand_init:
        dino_model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.0,img_size=224)
    else:
        dino_model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.0,img_size=224)
        msg = dino_model.load_state_dict(torch.load(opt.checkpoint, map_location='cpu')['model'], strict=False) 
        

        print(msg)

    model_accuracy = run_eval(opt, test_data_loader, dino_model, device)
    print(f"Jaccard index for {opt.model_name}: {model_accuracy}")
