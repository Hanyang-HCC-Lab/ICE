import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from numpy import *
import argparse
from PIL import Image
import imageio
import os
from tqdm import tqdm
from utils.metrices import *

from utils import render
from utils.saver import Saver
from utils.iou import IoU

from data.Imagenet import Imagenet_Segmentation

from ViT_explanation_generator import Baselines, LRP
from ViT_new import vit_base_patch16_224
from ViT_LRP import vit_base_patch16_224 as vit_LRP
from ViT_orig_LRP import vit_base_patch16_224 as vit_orig_LRP

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

import torch.nn.functional as F


from timm.models import create_model
from models_ICE import VisionTransformer
from functools import partial

import torch
import torch.nn as nn
from functools import partial

#from timm.models.vision_transformer import VisionTransformer, _cfg, Block
from timm.models.vision_transformer import  _cfg,Attention
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

#from timm.models.vision_transformer import *
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_


plt.switch_backend('agg')


# hyperparameters
num_workers = 0
batch_size = 1

cls = ['airplane',
       'bicycle',
       'bird',
       'boat',
       'bottle',
       'bus',
       'car',
       'cat',
       'chair',
       'cow',
       'dining table',
       'dog',
       'horse',
       'motobike',
       'person',
       'potted plant',
       'sheep',
       'sofa',
       'train',
       'tv'
       ]

# Args
parser = argparse.ArgumentParser(description='Training multi-class classifier')
parser.add_argument('--arc', type=str, default='vgg', metavar='N',
                    help='Model architecture')
parser.add_argument('--train_dataset', type=str, default='imagenet', metavar='N',
                    help='Testing Dataset')
parser.add_argument('--method', type=str,
                    default='grad_rollout',
                    choices=[ 'rollout', 'lrp','transformer_attribution', 'full_lrp', 'lrp_last_layer',
                              'attn_last_layer', 'attn_gradcam','test'],
                    help='')
parser.add_argument('--thr', type=float, default=0.,
                    help='threshold')
parser.add_argument('--K', type=int, default=1,
                    help='new - top K results')
parser.add_argument('--save-img', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-ia', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-fx', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-fgx', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-m', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-reg', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--is-ablation', type=bool,
                    default=False,
                    help='')
parser.add_argument('--imagenet-seg-path', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True)

args = parser.parse_args()

args.checkname = args.method + '_' + args.arc

alpha = 2

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Define Saver
saver = Saver(args)
saver.results_dir = os.path.join(saver.experiment_dir, 'results')
if not os.path.exists(saver.results_dir):
    os.makedirs(saver.results_dir)
if not os.path.exists(os.path.join(saver.results_dir, 'input')):
    os.makedirs(os.path.join(saver.results_dir, 'input'))
if not os.path.exists(os.path.join(saver.results_dir, 'explain')):
    os.makedirs(os.path.join(saver.results_dir, 'explain'))

args.exp_img_path = os.path.join(saver.results_dir, 'explain/img')
if not os.path.exists(args.exp_img_path):
    os.makedirs(args.exp_img_path)
args.exp_np_path = os.path.join(saver.results_dir, 'explain/np')
if not os.path.exists(args.exp_np_path):
    os.makedirs(args.exp_np_path)

# Data
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
test_img_trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])
test_lbl_trans = transforms.Compose([
    transforms.Resize((224, 224), Image.NEAREST),
])

ds = Imagenet_Segmentation(args.imagenet_seg_path,
                           transform=test_img_trans, target_transform=test_lbl_trans)
dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

# Model

model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.0,img_size=224)



#model.load_state_dict(torch.load("../ICE.npz", map_location='cpu'), strict=False)
model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'], strict=False)


model = model.cuda()

metric = IoU(2, ignore_index=-1)

iterator = tqdm(dl)

model.eval()


def compute_pred(output):
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    # pred[0, 0] = 282
    # print('Pred cls : ' + str(pred))
    T = pred.squeeze().cpu().numpy()
    T = np.expand_dims(T, 0)
    T = (T[:, np.newaxis] == np.arange(1000)) * 1.0
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = T.cuda()

    return Tt


def eval_batch(image, labels, evaluator, index):
    evaluator.zero_grad()
    # Save input image
    if args.save_img:
        img = image[0].permute(1, 2, 0).data.cpu().numpy()
        img = 255 * (img - img.min()) / (img.max() - img.min())
        img = img.astype('uint8')
        Image.fromarray(img, 'RGB').save(os.path.join(saver.results_dir, 'input/{}_input.png'.format(index)))
        Image.fromarray((labels.repeat(3, 1, 1).permute(1, 2, 0).data.cpu().numpy() * 255).astype('uint8'), 'RGB').save(
            os.path.join(saver.results_dir, 'input/{}_mask.png'.format(index)))

    image.requires_grad = True

    image = image.requires_grad_()
    predictions = evaluator(image)
    
    
    _, attention_mask ,_ = model(image.cuda())
    attention_mask = torch.argmax(attention_mask.view(14,14,1001),axis = 2)
    attention_mask = attention_mask.masked_fill(attention_mask != 1000,1) 
    attention_mask = attention_mask.masked_fill(attention_mask == 1000,0) 
    attention_mask = attention_mask.repeat_interleave(16,dim=0)
    attention_mask = attention_mask.repeat_interleave(16,dim=1)
    Res = attention_mask.unsqueeze(0).unsqueeze(0)

    # threshold between FG and BG is the mean    
    Res = (Res - Res.min()) / (Res.max() - Res.min())

    ret = Res.mean()

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())

    Res_1_AP = Res
    Res_0_AP = 1-Res

    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0


    # TEST

    target = labels.view(-1).data.cpu().numpy()


    output = torch.cat((Res_0, Res_1), 1)


    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
    batch_ap, batch_f1 = 0, 0

    # Segmentation resutls
    correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
    inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union


    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, 0, target


total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
total_ap, total_f1 = [], []

predictions, targets = [], []
for batch_idx, (image, labels) in enumerate(iterator):

    if args.method == "blur":
        images = (image[0].cuda(), image[1].cuda())
    else:
        images = image.cuda()
    labels = labels.cuda()


    correct, labeled, inter, union, ap, f1, pred, target = eval_batch(images, labels, model, batch_idx)

    predictions.append(pred)
    targets.append(target)

    total_correct += correct.astype('int64')
    total_label += labeled.astype('int64')
    total_inter += inter.astype('int64')
    total_union += union.astype('int64')
    total_ap += [ap]
    total_f1 += [f1]
    pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
    IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
    mIoU = IoU.mean()
    mAp = np.mean(total_ap)
    mF1 = np.mean(total_f1)
    iterator.set_description('pixAcc: %.4f, mIoU: %.4f, mAP: %.4f, mF1: %.4f' % (pixAcc, mIoU, mAp, mF1))



plt.figure()


txtfile = os.path.join(saver.experiment_dir, 'result_mIoU_%.4f.txt' % mIoU)

fh = open(txtfile, 'w')
print("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
print("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
print("Mean AP over %d classes: %.4f\n" % (2, mAp))
print("Mean F1 over %d classes: %.4f\n" % (2, mF1))

fh.write("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
fh.write("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
fh.write("Mean AP over %d classes: %.4f\n" % (2, mAp))
fh.write("Mean F1 over %d classes: %.4f\n" % (2, mF1))
fh.close()