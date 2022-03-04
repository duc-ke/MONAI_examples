# -*- coding: utf-8 -*-
# ## Inference 코드 테스트
# * `st2-torch_ex-segmentation-unet_training_dict.ipynb` 으로 학습한 모델을 바탕으로 inference
#

import logging
import os
import sys
import tempfile          # ?
from glob import glob

import nibabel as nib    # ?
import numpy as np
import torch

# +
from monai.config import print_config
from monai.data import Dataset, DataLoader, create_test_image_3d, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    Resized,
    SaveImaged,
    ScaleIntensityd,
    EnsureTyped,
)

import monai
import matplotlib.pyplot as plt
# -

# ### test file 5개 생성(for inference)

# +
tempdir = './data/unet_infer'

print_config()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

print(f"generating synthetic data to {tempdir} (this may take a while)")

for i in range(5):
    im, _ = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
    n = nib.Nifti1Image(im, np.eye(4))
    nib.save(n, os.path.join(tempdir, f"im{i:d}.nii.gz"))
    
images = sorted(glob(os.path.join(tempdir, "im*.nii.gz")))
files = [{"img": img} for img in images]
# -

# ### Transform, Dataset, DataLoader
#
# * transform : 순서는 변했을 지라도 필수적인것은 똑같이 들어가나 resize 새롭게 추가.
#   * 같음 : load, 채널앞부분 삽입, scaling, ensuretype
#   * 달라짐 :
#     * orienation : 어짜피 MRI 방향만 바꿔주는거니까 상관 없을 듯
#     * Resize : training시 `RandCropByPosNegLabeld`를 썻기 때문에 crop size(96, 96, 96)으로 맞춰주기 위해서 삽입됨. 이러면 크롭하여 prediction하는게 아니라 전체 이미지가 줄어드는건 아닌지 우려가 되긴 하네.
#  

# +
# define pre transforms
pre_transforms = Compose([
    LoadImaged(keys="img"),
    EnsureChannelFirstd(keys="img"),    # (128, 128, 128, 1)를 (1, 128, 128, 128)로 맞춰줌 + torch 변환
    Orientationd(keys="img", axcodes="RAS"),
#     Resized(keys="img", spatial_size=(96, 96, 96), mode="trilinear", align_corners=True),
    ScaleIntensityd(keys="img"),
    EnsureTyped(keys="img"),
])
test_no_resize_transforms = Compose([
    LoadImaged(keys="img"),
    EnsureChannelFirstd(keys="img"), 
    Orientationd(keys="img", axcodes="RAS"),
    ScaleIntensityd(keys="img"),
    EnsureTyped(keys="img"),
])
# define dataset and dataloader
dataset = Dataset(data=files, transform=pre_transforms)
dataloader = DataLoader(dataset, batch_size=2, num_workers=4)
check_data = monai.utils.misc.first(dataloader)
print(check_data["img"].shape)


# resize 안하면?
dataset2 = Dataset(data=files, transform=test_no_resize_transforms)
dataloader2 = DataLoader(dataset2, batch_size=2, num_workers=4)
check_data2 = monai.utils.misc.first(dataloader2)
# -

# #### ex) Resize 테스트
# 단순히 resize하여 inference를 하는거면 crop하는게 아니고 Resize를 한다면.. 처음 학습 의도와는 다르게 그림자체가 줄어든 상태로 prediction을 할텐데.. 실제 resize는 어떻게 작동하는지 확인해본다.
#
# 결과 : 여기선 내가 우려하는 방식으로 resize는 전체적인 픽셀을 줄여주는 방식이였음. 이방법을 사용하면 안될 것 같음.

# +
# Resize 했을 때 테스트

a = int(96/2)-1
a2 = int(128/2)-1
plt.subplots(1, 2, figsize=(8, 8))
plt.subplot(1, 2, 1)
img = check_data['img'].numpy()
img2 = check_data2['img'].numpy()
print(type(img), img.shape)
img = img[0, 0, :, :, a]
img2 = img2[0, 0, :, :, a2]
plt.xlabel('img')
print(img.shape)
plt.imshow(img)
# plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.subplot(1, 2, 2)
plt.xlabel('img2')
print(img2.shape)
plt.imshow(img2)
plt.show()

# for each in check_data:
#     img = each['img'].numpy()
#     print(type(img), img.shape)

#     img = img[0, 0, 0, :, :]
#     seg = seg[0, 0, 0, :, :]
#     print(img.shape)
# #     plt.imshow(img, cmap="gray", vmin=0, vmax=255)
# #     plt.imshow(img, cmap="gray")
#     plt.subplot(1, 2, 1)
#     plt.xlabel('img')
#     plt.imshow(img)
#     plt.subplot(1, 2, 2)
#     plt.xlabel('seg')
#     plt.imshow(seg)
# #     raise AssertionError("!!!")
# plt.tight_layout()
# plt.show()
# -

# ### post-transform 정의
# * binary classes seg를 위하여 sigmoid를 달아주고 원본으로 복원하기 위해 Invertd.
# * Inverted : pre-trainsform에서 사용했던 이미지 방향(Orientation), Resize, Scaling을 복원
# * post-transform에서 saveImaged로 결과 저장

post_transforms = Compose([
    EnsureTyped(keys="pred"),
    Activationsd(keys="pred", sigmoid=True),
    Invertd(
        keys="pred",  # invert the `pred` data field, also support multiple fields
        transform=pre_transforms,
        # 이미지에 적용되었던 pre_transform 정보를 가져와서 이 정보를 기반으로 pred를 반전
        orig_keys="img",  # get the previously applied pre_transforms information on the `img` data field,
                          # then invert `pred` based on this information. we can use same info
                          # for multiple fields, also support different orig_keys for different fields
        # 반전된 pred의 데이터를 저장하기 위한 변수설정
        meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
        
        # 반전시 'spacingd'변환에선 'affine'이 필요 할 수 있음.
        orig_meta_keys="img_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
                                         # for example, may need the `affine` to invert `Spacingd` transform,
                                         # multiple fields can use the same meta data to invert
        
        # `meta_keys`를 지정안하면 {keys}_{meta_key_postfix} 이것을 디폴트로 알아먹음
        meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
                                       # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
                                       # otherwise, no need this arg during inverting
        
        # 부드러운 출력을 만들기 위한다면 반전시 보간 모드를 nearest를 쓰지말고 AsDiscreted transform을 실행해야 한다. (지금 예제가 그러함)
        nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                               # to ensure a smooth output, then execute `AsDiscreted` transform
        to_tensor=True,  # convert to PyTorch Tensor after inverting
    ),
    AsDiscreted(keys="pred", threshold=0.5),   # 0과 1로 나눔
    SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="data/unet_infer/out", output_postfix="seg", resample=False),
])

len(dataloader)

# ## Do Main inference
# * 모델과 초기값은 학습과 동일하게 해야함.
# * 모델 load는 pytorch, monai와 같다.
# * pytorch 코딩처럼 eval모드를 키고 하면 된다
#
# Q?: 어짜피 resize를 진행해서 그림 사이즈를 줄였는데 sliding window inference를 할 필요가 있는가?

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
net.load_state_dict(torch.load("models/best_metric_model_segmentation3d_dict.pth"))

net.eval()
with torch.no_grad():
    for d in dataloader:
        images = d["img"].to(device)
        # define sliding window size and batch size for windows inference
        # sliding window를 할 필요가 있는가..?
        d["pred"] = sliding_window_inference(inputs=images, roi_size=(96, 96, 96), sw_batch_size=4, predictor=net)
        print(d["pred"].shape)
        # decollate the batch data into a list of dictionaries, then execute postprocessing transforms
        # decollate는 batch를 각각의 리스트로 쪼개주는 역할
        print(type(decollate_batch(d)), decollate_batch(d)[0]['img'].shape)
        d = [post_transforms(i) for i in decollate_batch(d)]


# -

# ## 아래는 원본 script

# +
def main(tempdir):
    print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    print(f"generating synthetic data to {tempdir} (this may take a while)")
    for i in range(5):
        im, _ = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"im{i:d}.nii.gz"))

    images = sorted(glob(os.path.join(tempdir, "im*.nii.gz")))
    files = [{"img": img} for img in images]

    # define pre transforms
    pre_transforms = Compose([
        LoadImaged(keys="img"),
        EnsureChannelFirstd(keys="img"),
        Orientationd(keys="img", axcodes="RAS"),
        Resized(keys="img", spatial_size=(96, 96, 96), mode="trilinear", align_corners=True),
        ScaleIntensityd(keys="img"),
        EnsureTyped(keys="img"),
    ])
    # define dataset and dataloader
    dataset = Dataset(data=files, transform=pre_transforms)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=4)
    # define post transforms
    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", sigmoid=True),
        Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=pre_transforms,
            orig_keys="img",  # get the previously applied pre_transforms information on the `img` data field,
                              # then invert `pred` based on this information. we can use same info
                              # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="img_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
                                             # for example, may need the `affine` to invert `Spacingd` transform,
                                             # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
                                           # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
                                           # otherwise, no need this arg during inverting
            nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                                   # to ensure a smooth output, then execute `AsDiscreted` transform
            to_tensor=True,  # convert to PyTorch Tensor after inverting
        ),
        AsDiscreted(keys="pred", threshold=0.5),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="data/unet_infer/out", output_postfix="seg", resample=False),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    net.load_state_dict(torch.load("best_metric_model_segmentation3d_dict.pth"))

    net.eval()
    with torch.no_grad():
        for d in dataloader:
            images = d["img"].to(device)
            # define sliding window size and batch size for windows inference
            d["pred"] = sliding_window_inference(inputs=images, roi_size=(96, 96, 96), sw_batch_size=4, predictor=net)
            # decollate the batch data into a list of dictionaries, then execute postprocessing transforms
            d = [post_transforms(i) for i in decollate_batch(d)]

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
        main(tempdir)
