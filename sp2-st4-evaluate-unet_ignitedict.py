# -*- coding: utf-8 -*-
# ## evaluation 코드 테스트
# * `st4-ignite_ex-segmentation-unet_training_dict.ipynb`으로 학습한 모델을 바탕으로 evaluation
# * 내부 코드를 잘 보면 inference로 바로 쓸수도 있을 듯하다.
#   * pred_img를 post-transform으로 invert하여 nifti로 저장하면 됨.

import logging
import os
import sys
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch
from ignite.engine import Engine    # training에선 없었음.
from torch.utils.data import DataLoader

import monai
from monai.data import create_test_image_3d, list_data_collate, decollate_batch
from monai.handlers import CheckpointLoader, MeanDice, StatsHandler
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    AsChannelFirstd,
    LoadImaged,
    ScaleIntensityd,
    EnsureTyped
    
    # 아래는 dictform 아님(ignite의 post_ransform을 위해서임)
    Activations, 
    AsDiscrete, 
    Compose,  
    SaveImage,   # 이게 특이함.
    EnsureType

import matplotlib.pyplot as plt

# ### validation file 5개 생성(for inference)

# +
tempdir = './data/unet_ignite_infer'
monai.config.print_config()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print(f"generating synthetic data to {tempdir} (this may take a while)")

for i in range(5):
    im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)

    n = nib.Nifti1Image(im, np.eye(4))
    nib.save(n, os.path.join(tempdir, f"im{i:d}.nii.gz"))

    n = nib.Nifti1Image(seg, np.eye(4))
    nib.save(n, os.path.join(tempdir, f"seg{i:d}.nii.gz"))

images = sorted(glob(os.path.join(tempdir, "im*.nii.gz")))
segs = sorted(glob(os.path.join(tempdir, "seg*.nii.gz")))
val_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]
# -

# ### Transform, Dataset, DataLoader
# * transform: 완전 똑같으나 RandCropByPosNegLabeld()만 빠짐. 즉, 학습시 썼던 val_transforms과 같음

# +
# define transforms for image and segmentation
val_transforms = Compose(
    [
        LoadImaged(keys=["img", "seg"]),
        AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
        ScaleIntensityd(keys="img"),
        EnsureTyped(keys=["img", "seg"]),
    ]
)
val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(
    val_ds, batch_size=1, 
    num_workers=4, collate_fn=list_data_collate, 
    pin_memory=torch.cuda.is_available()
)

check_data = monai.utils.misc.first(val_loader)

# +
# 이미지로 그려보자.

a = 60
plt.subplots(1, 2, figsize=(8, 8))
plt.subplot(1, 2, 1)
img = check_data['img'].numpy()
seg = check_data['seg'].numpy()
print(type(img), img.shape)
img = img[0, 0, :, :, a]
seg = seg[0, 0, :, :, a]
plt.xlabel('img')
print(img.shape)
plt.imshow(img)
# plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.subplot(1, 2, 2)
plt.xlabel('seg')
print(seg.shape)
plt.imshow(seg)
plt.show()
# -

# ### post-transform 정의
# * sp1과 달리 Invertd가 없음. 없어도 상관없는지 확인 필요.

## sp1과 달리 list form이다.
## 아래 보면 inference를 위해서 어쩔 수 없는 듯.
post_trans = Compose([EnsureType(), 
                      Activations(sigmoid=True), 
                      AsDiscrete(threshold=0.5)])
save_image = SaveImage(output_dir="./data/unet_ignite_infer/out", 
                       output_ext=".nii.gz", 
                       output_postfix="seg")

# ## Do Main inference
# * pytorch 버전에선 load model을 모델을 정의하고 바로했으나 ignite 에선 evaluator를 정의하고 거기에 붙인다.
# * evaluator를 engine에 붙여서 생성가능하며, 아래 코드는 img만 따로 처리하기 때문에 transform-dict를 이용하지 않고 list form을 이용한다.

# +
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

# pytorch 버전 코딩을 쓰려면 여기서 load_model을 시작해야 한다.
# 그러나 Ignite 버전은 evaluator를 먼저 정의한뒤 evaluator에 model을 loading시켜야 한다.

# define sliding window size and batch size for windows inference
roi_size = (96, 96, 96)
sw_batch_size = 4

# +
## train(ignite ver)에선 Ignite api-create_supervised_trainer, evaluator를 썼으나,
## Ignite api - Engine을 로드하면 직접 정의해서 쓸수있다.
## train코드의 prepare_batch처럼 디폴트로 batch를 받아서 활용가능하다.
"""참고 : https://github.com/pytorch/ignite/blob/master/examples/notebooks/TextCNN.ipynb

# 마치 pytorch에서 직접 정의하는 것과 같이 쓸 수 있다.

def process_function(engine, batch):
    model.train()
    optimizer.zero_grad()
    y, x = batch
    x = x.to(device)
    y = y.to(device)
    y_pred = model(x)
    loss = criterion(y_pred, y.float())
    loss.backward()
    optimizer.step()
    return loss.item()

def eval_function(engine, batch):
    model.eval()
    with torch.no_grad():
        y, x = batch
        y = y.to(device)
        x = x.to(device)
        y = y.float()
        y_pred = model(x)
        return y_pred, y

trainer = Engine(process_function)
train_evaluator = Engine(eval_function)
validation_evaluator = Engine(eval_function)
    
"""
def _sliding_window_processor(engine, batch):
    net.eval()
    with torch.no_grad():
        val_images, val_labels = batch["img"].to(device), batch["seg"].to(device)
        seg_probs = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
        # decollate는 batch를 각각의 리스트로 쪼개주는 역할
        seg_probs = [post_trans(i) for i in decollate_batch(seg_probs)]

        # decollate는 batch를 각각의 리스트로 쪼개주는 역할
        val_data = decollate_batch(batch["img_meta_dict"])
        for seg_prob, data in zip(seg_probs, val_data):
            save_image(seg_prob, data)
        return seg_probs, val_labels   # ignite에선 return output을 (pred, label로 줘야하는듯?)

evaluator = Engine(_sliding_window_processor)

# add evaluation metric to the evaluator engine
MeanDice().attach(evaluator, "Mean_Dice")

# StatsHandler prints loss at every iteration and print metrics at every epoch,
# we don't need to print loss for evaluator, so just print metrics, user can also customize print functions
val_stats_handler = StatsHandler(
    name="evaluator",
    output_transform=lambda x: None,  # no need to print loss value, so disable per iteration output
)
val_stats_handler.attach(evaluator)

# the model was trained by "unet_training_dict" example
CheckpointLoader(load_path="./runs_dict/net_checkpoint_40.pt", load_dict={"net": net}).attach(evaluator)

# sliding window inference for one image at every iteration
state = evaluator.run(val_loader)
print(state)
# -











# ----------
#
# ## 아래는 원본 script

def main(tempdir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    print(f"generating synthetic data to {tempdir} (this may take a while)")
    for i in range(5):
        im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)

        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"im{i:d}.nii.gz"))

        n = nib.Nifti1Image(seg, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"seg{i:d}.nii.gz"))

    images = sorted(glob(os.path.join(tempdir, "im*.nii.gz")))
    segs = sorted(glob(os.path.join(tempdir, "seg*.nii.gz")))
    val_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]

    # define transforms for image and segmentation
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            ScaleIntensityd(keys="img"),
            EnsureTyped(keys=["img", "seg"]),
        ]
    )
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # define sliding window size and batch size for windows inference
    roi_size = (96, 96, 96)
    sw_batch_size = 4

    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    save_image = SaveImage(output_dir="tempdir", output_ext=".nii.gz", output_postfix="seg")

    def _sliding_window_processor(engine, batch):
        net.eval()
        with torch.no_grad():
            val_images, val_labels = batch["img"].to(device), batch["seg"].to(device)
            seg_probs = sliding_window_inference(val_images, roi_size, sw_batch_size, net)
            seg_probs = [post_trans(i) for i in decollate_batch(seg_probs)]
            
            # decollate는 batch를 각각의 리스트로 쪼개주는 역할
            val_data = decollate_batch(batch["img_meta_dict"])
            for seg_prob, data in zip(seg_probs, val_data):
                save_image(seg_prob, data)
            return seg_probs, val_labels

    evaluator = Engine(_sliding_window_processor)

    # add evaluation metric to the evaluator engine
    ## !!! MONAI handler MeanDice ##
    MeanDice().attach(evaluator, "Mean_Dice")

    # StatsHandler prints loss at every iteration and print metrics at every epoch,
    # we don't need to print loss for evaluator, so just print metrics, user can also customize print functions
    val_stats_handler = StatsHandler(
        name="evaluator",
        output_transform=lambda x: None,  # no need to print loss value, so disable per iteration output
    )
    val_stats_handler.attach(evaluator)

    # the model was trained by "unet_training_dict" example
    CheckpointLoader(load_path="./runs_dict/net_checkpoint_50.pt", load_dict={"net": net}).attach(evaluator)

    # sliding window inference for one image at every iteration
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available()
    )
    state = evaluator.run(val_loader)
    print(state)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)
