import sys
import os
sys.path.append(f"./diffusers_jittor/src")
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import numpy as np
import argparse
import warnings
import json, os, tqdm, torch

import jittor as jt
import jittor.nn as nn

from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline
from PIL import Image
from pathlib import Path
from PIL.ImageOps import exif_transpose
from jittor import transform
from jittor.compatibility.utils.data import Dataset

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

def set_seed_fc(args, style_id, items_fc):
    seed_fc_my = int(time.time())
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/seed.txt", mode="a") as file_obj:
        file_obj.write(f"{items_fc}, {style_id}: {seed_fc_my}\n")
    # 设置 NumPy 的随机种子
    np.random.seed(seed_fc_my)
    
    # 设置 Jittor 的随机种子
    jt.set_global_seed(seed_fc_my)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_result_8_20_add_negative_4535_700_13_3_ip_text_addce_0",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--seed_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args = parser.parse_args()
    return args

args = parse_args()         # args.output_dir 是生成图像的保存路径，在parse_args(input_args=None)函数中修改
start_num = 0               # 开始测试的风格对应的数字
end_num = start_num + 28    
dataset_root = "./B"        # 风格数据集的路径
root_path_weight = f"./style_8_20_text_45_35_addce_ip_text_13_001_700_3"    # 权重路径


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt=None,
        tokenizer=None,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transform.Compose(
            [
                transform.Resize(size),
                transform.CenterCrop(size) if center_crop else transform.RandomCrop(size),        # 感觉会影响生成图像的全局性
                transform.ToTensor(),       # 自带规一化操作
                transform.ImageNormalize([0.5], [0.5]),         # 这个好像不能删除，会影响生成图像的色彩
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image_name = os.path.basename(self.instance_images_path[index % self.num_instance_images]).split(".")[0]
        if "_" in instance_image_name:
            instance_image_name = instance_image_name.replace("_", " ")
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        
        return example


class AddcE(nn.Module):
    """不同层级语义特征加权求和"""
    def __init__(self, input_num=2):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.input_num = input_num
        self.fl_weight = nn.Parameter(jt.zeros(self.input_num - 1))     # 第一个参数乘以风格特征
        # self.add_parameters("fl_weight", jt.zeros(self.input_num - 1))

    def execute(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        tem = x[0]*self.fl_weight + x[1]*(1 - self.fl_weight)       # 加权的权重值和为1，是为保证输出值的取值范围和输入值的取值范围接近（主要是考虑到原始的SD代码中对vae_encoder之后的特征有一个尺寸的缩放，说明尺度的范围还是比较重要的）
                
        return tem

class AddcE2(nn.Module):
    """不同层级语义特征加权求和"""
    def __init__(self, input_num=2):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.input_num = input_num
        self.fl_weight = nn.Parameter(jt.zeros(self.input_num))     # 第一个参数乘以风格特征
        # self.add_parameters("fl_weight", jt.zeros(self.input_num - 1))

    def execute(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        tem = x[0]*self.fl_weight[0] + x[1]*self.fl_weight[1]       # 加权的权重值和为1，是为保证输出值的取值范围和输入值的取值范围接近（主要是考虑到原始的SD代码中对vae_encoder之后的特征有一个尺寸的缩放，说明尺度的范围还是比较重要的）
                
        return tem

class AddcE3(nn.Module):
    """不同层级语义特征加权求和, 先cat后liner, linear通过1*1的卷积完成"""

    def __init__(self, in_chenannel=8, out_channel=4):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.liner_proj = nn.Linear(in_chenannel, out_channel)

        # 将线性层的偏置设置为0；将权重的前一半设置为1，后一半设置为0        # 权重的初始化设置
        self.liner_proj.bias = jt.zeros(self.liner_proj.bias.shape)
        half_weight = self.liner_proj.weight.shape[1]//2
        self.liner_proj.weight[:, :half_weight] = jt.zeros((out_channel, half_weight))     # 通道的前一半对应的是风格噪声        
        self.liner_proj.weight[:, half_weight:] = jt.ones((out_channel, half_weight))     # 通道的后一半对应的是真实特征        


        # self.fl_weight = nn.Parameter(jt.zeros(self.input_num))     # 第一个参数乘加权风格特征
        # self.fl_weight[1] = 1     # 第2个参数乘加权原本特征
        
        # self.add_parameters("fl_weight", jt.zeros(self.input_num - 1))

    def execute(self, x):  # x为list, 第一个元素为风格特征, 第二个元素为真实噪声
        x_cat = jt.cat(x, dim=1).permute(0, 2, 3, 1)        # NCHW ---> NHWC
        tem = self.liner_proj(x_cat)

        return tem.permute(0, 3, 1, 2)      # NHWC ----> NCHW
    
class AddcE4(nn.Module):
    """使用softmax 将特征的加权值之和限制为1"""

    def __init__(self, in_channel=8, out_channel=4):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        
        weight_tem = jt.zeros(out_channel, in_channel)
        # 将线性层的偏置设置为0；将权重的前一半设置为1，后一半设置为0        # 权重的初始化设置        
        half_weight = in_channel//2
        weight_tem[:, :half_weight] = jt.zeros((out_channel, half_weight))     # 通道的前一半对应的是风格噪声        
        weight_tem[:, half_weight:] = jt.ones((out_channel, half_weight))     # 通道的后一半对应的是真实特征        
        
        self.weight = nn.Parameter(weight_tem)
        self.bias = None
        

        # self.fl_weight = nn.Parameter(jt.zeros(self.input_num))     # 第一个参数乘加权风格特征
        # self.fl_weight[1] = 1     # 第2个参数乘加权原本特征
        
        # self.add_parameters("fl_weight", jt.zeros(self.input_num - 1))

    def execute(self, x):  # x为list, 第一个元素为风格特征, 第二个元素为真实噪声
        x_cat = jt.cat(x, dim=1).permute(0, 2, 3, 1)        # NCHW ---> NHWC
        
        tem = nn.linear(x_cat, nn.softmax(self.weight, dim=1))

        return tem.permute(0, 3, 1, 2)      # NHWC ----> NCHW



class vae_img_feature_change(nn.Module):
    """将VAE特征维度变换到文本向量的特征维度"""    

    def __init__(self, in_channel=8, out_channel=77):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.activate_fn1 = nn.SiLU()

        self.conv2 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.activate_fn2 = nn.SiLU()

        # self.linear_layer = nn.Linear(8*64*64, 77*1024)
        # self.linear_layer_weight = nn.Parameter(jt.zeros((77*1024, 8*64*64)))
        # self.ln_layer = nn.LayerNorm(77*1024)

        # 将权重初始化为0
        self.conv1.weight = jt.zeros(self.conv1.weight.shape)
        # 如果需要，也可以初始化偏置为 0
        if self.conv1.bias is not None:
            self.conv1.bias = jt.zeros(self.conv1.bias.shape)
        # with jt.init_scope():
        # nn.init.constant(self.linear_layer.weight, "float32", 0.0)

        # # 如果你也想将偏置初始化为0，可以这样设置
        # if self.linear_layer.bias is not None:
        #     # with jt.init_scope():
        #     nn.init.constant(self.linear_layer.bias, "float32", 0.0)
        # self.miu_weight = nn.Parameter(jt.zeros((1, num_img)))
        # self.sigma_weight = nn.Parameter(jt.zeros((1, num_img)))

        
    def execute(self, x_std, x_miu):  # x为list, 第一个元素为风格特征, 第二个元素为真实噪声
        x_img = jt.cat([x_std, x_miu], dim=1)       # 将vae encode的结果进行拼接、展平
        

        tem_x = self.activate_fn1(self.bn1(self.conv1(x_img)))
        tem_x = self.activate_fn2(self.bn2(self.conv2(tem_x)))
        # tem_result = self.ln_layer(nn.linear(x_img, self.linear_layer_weight)).view(1, 77, -1)
        # x_std_cat = jt.cat(x_std, dim=0).permute(1, 2, 3, 0)        # NCHW ---> CHWN
        # x_miu_cat = jt.cat(x_miu, dim=0).permute(1, 2, 3, 0)        # NCHW ---> CHWN
        
        # std_result = nn.linear(x_std_cat, self.sigma_weight).permute(3, 0, 1, 2)        # CHWN ---> NCHW
        # miu_result = nn.linear(x_miu_cat, self.miu_weight).permute(3, 0, 1, 2)          # CHWN ---> NCHW

        # style_img_input = std_result*jt.randn_like(std_result) + miu_result  # 风格特征的采样

        return tem_x.view(1, 77, -1)


class vae_img_feature_change2(nn.Module):
    """将VAE特征维度变换到文本向量的特征维度"""    

    def __init__(self, in_channel=8, out_channel=77):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel//2, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel//2)
        self.activate_fn1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channel//2, out_channel//2, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel//2)
        self.activate_fn2 = nn.ReLU()


        self.conv3 = nn.Conv2d(out_channel//2, out_channel, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.activate_fn3 = nn.ReLU()
        # self.linear_layer = nn.Linear(8*64*64, 77*1024)
        # self.linear_layer_weight = nn.Parameter(jt.zeros((77*1024, 8*64*64)))
        # self.ln_layer = nn.LayerNorm(77*1024)

        # 将权重初始化为0
        self.conv1.weight = jt.zeros(self.conv1.weight.shape)
        # 如果需要，也可以初始化偏置为 0
        if self.conv1.bias is not None:
            self.conv1.bias = jt.zeros(self.conv1.bias.shape)
        # with jt.init_scope():
        # nn.init.constant(self.linear_layer.weight, "float32", 0.0)

        # # 如果你也想将偏置初始化为0，可以这样设置
        # if self.linear_layer.bias is not None:
        #     # with jt.init_scope():
        #     nn.init.constant(self.linear_layer.bias, "float32", 0.0)
        # self.miu_weight = nn.Parameter(jt.zeros((1, num_img)))
        # self.sigma_weight = nn.Parameter(jt.zeros((1, num_img)))

        
    def execute(self, x_std, x_miu):  # x为list, 第一个元素为风格特征, 第二个元素为真实噪声
        x_img = jt.cat([x_std, x_miu], dim=1)       # 将vae encode的结果进行拼接、展平
        

        tem_x = self.activate_fn1(self.bn1(self.conv1(x_img)))
        tem_x = self.activate_fn2(self.bn2(self.conv2(tem_x)))
        tem_x = self.activate_fn3(self.bn3(self.conv3(tem_x)))
        # tem_result = self.ln_layer(nn.linear(x_img, self.linear_layer_weight)).view(1, 77, -1)
        # x_std_cat = jt.cat(x_std, dim=0).permute(1, 2, 3, 0)        # NCHW ---> CHWN
        # x_miu_cat = jt.cat(x_miu, dim=0).permute(1, 2, 3, 0)        # NCHW ---> CHWN
        
        # std_result = nn.linear(x_std_cat, self.sigma_weight).permute(3, 0, 1, 2)        # CHWN ---> NCHW
        # miu_result = nn.linear(x_miu_cat, self.miu_weight).permute(3, 0, 1, 2)          # CHWN ---> NCHW

        # style_img_input = std_result*jt.randn_like(std_result) + miu_result  # 风格特征的采样

        return tem_x.view(1, 77, -1)



class vae_img_feature_change4(nn.Module):
    """将VAE特征维度变换到文本向量的特征维度"""    
    """在2的基础上使用GSC模块"""
    def __init__(self, in_channel=8, out_channel=77):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.bn1 = nn.GroupNorm(num_channels=in_channel, num_groups=4)
        self.activate_fn1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channel, out_channel//2, 3, 2, 1)     # , bias=False

        
        self.bn2 = nn.GroupNorm(num_channels=out_channel//2, num_groups=19)
        self.activate_fn2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channel//2, out_channel//2, 3, 1, 1)

        
        self.bn3 = nn.GroupNorm(num_channels=out_channel//2, num_groups=19)
        self.activate_fn3 = nn.SiLU()
        self.conv3 = nn.Conv2d(out_channel//2, out_channel//2, 3, 1, 1)

        self.bn4 = nn.GroupNorm(num_channels=out_channel//2, num_groups=2)
        self.activate_fn4 = nn.SiLU()
        self.conv4 = nn.Conv2d(out_channel//2, out_channel, 3, 1, 1)
        # self.linear_layer = nn.Linear(8*64*64, 77*1024)
        # self.linear_layer_weight = nn.Parameter(jt.zeros((77*1024, 8*64*64)))
        # self.ln_layer = nn.LayerNorm(77*1024)

        # 将权重初始化为0
        self.conv1.weight = jt.zeros(self.conv1.weight.shape)
        # 如果需要，也可以初始化偏置为 0
        if self.conv1.bias is not None:
            self.conv1.bias = jt.zeros(self.conv1.bias.shape)
        # with jt.init_scope():
        # nn.init.constant(self.linear_layer.weight, "float32", 0.0)

        # # 如果你也想将偏置初始化为0，可以这样设置
        # if self.linear_layer.bias is not None:
        #     # with jt.init_scope():
        #     nn.init.constant(self.linear_layer.bias, "float32", 0.0)
        # self.miu_weight = nn.Parameter(jt.zeros((1, num_img)))
        # self.sigma_weight = nn.Parameter(jt.zeros((1, num_img)))

        
    def execute(self, x_std, x_miu):  # x为list, 第一个元素为风格特征, 第二个元素为真实噪声
        x_img = jt.cat([x_std, x_miu], dim=1)       # 将vae encode的结果进行拼接、展平
        
        tem_x_1 = self.conv1(self.activate_fn1(self.bn1(x_img)))
        tem_x_2 = self.conv2(self.activate_fn2(self.bn2(tem_x_1)))
        tem_x_2 = self.conv3(self.activate_fn3(self.bn3(tem_x_2)))
        tem_x_3 = tem_x_2 + tem_x_1

        tem_x = self.conv4(self.activate_fn4(self.bn4(tem_x_3)))
        # tem_result = self.ln_layer(nn.linear(x_img, self.linear_layer_weight)).view(1, 77, -1)
        # x_std_cat = jt.cat(x_std, dim=0).permute(1, 2, 3, 0)        # NCHW ---> CHWN
        # x_miu_cat = jt.cat(x_miu, dim=0).permute(1, 2, 3, 0)        # NCHW ---> CHWN
        
        # std_result = nn.linear(x_std_cat, self.sigma_weight).permute(3, 0, 1, 2)        # CHWN ---> NCHW
        # miu_result = nn.linear(x_miu_cat, self.miu_weight).permute(3, 0, 1, 2)          # CHWN ---> NCHW

        # style_img_input = std_result*jt.randn_like(std_result) + miu_result  # 风格特征的采样

        return tem_x.view(1, 77, -1)



class vae_img_feature_change3_instance(nn.Module):
    """将VAE特征维度变换到文本向量的特征维度"""    
    """在2的基础上使用GSC模块,      instance normal"""
    def __init__(self, in_channel=8, out_channel=77):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.bn1 = nn.GroupNorm(num_channels=in_channel, num_groups=8)
        self.activate_fn1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channel, out_channel//2, 3, 2, 1)     # , bias=False

        
        self.bn2 = nn.GroupNorm(num_channels=out_channel//2, num_groups=out_channel//2)
        self.activate_fn2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channel//2, out_channel//2, 3, 1, 1)

        
        self.bn3 = nn.GroupNorm(num_channels=out_channel//2, num_groups=out_channel//2)
        self.activate_fn3 = nn.SiLU()
        self.conv3 = nn.Conv2d(out_channel//2, out_channel, 3, 1, 1)
        # self.linear_layer = nn.Linear(8*64*64, 77*1024)
        # self.linear_layer_weight = nn.Parameter(jt.zeros((77*1024, 8*64*64)))
        # self.ln_layer = nn.LayerNorm(77*1024)

        # 将权重初始化为0
        self.conv1.weight = jt.zeros(self.conv1.weight.shape)
        # 如果需要，也可以初始化偏置为 0
        if self.conv1.bias is not None:
            self.conv1.bias = jt.zeros(self.conv1.bias.shape)
        # with jt.init_scope():
        # nn.init.constant(self.linear_layer.weight, "float32", 0.0)

        # # 如果你也想将偏置初始化为0，可以这样设置
        # if self.linear_layer.bias is not None:
        #     # with jt.init_scope():
        #     nn.init.constant(self.linear_layer.bias, "float32", 0.0)
        # self.miu_weight = nn.Parameter(jt.zeros((1, num_img)))
        # self.sigma_weight = nn.Parameter(jt.zeros((1, num_img)))

        
    def execute(self, x_std, x_miu):  # x为list, 第一个元素为风格特征, 第二个元素为真实噪声
        x_img = jt.cat([x_std, x_miu], dim=1)       # 将vae encode的结果进行拼接、展平
        
        tem_x = self.conv1(self.activate_fn1(self.bn1(x_img)))
        tem_x = self.conv2(self.activate_fn2(self.bn2(tem_x)))
        tem_x = self.conv3(self.activate_fn3(self.bn3(tem_x)))
        # tem_result = self.ln_layer(nn.linear(x_img, self.linear_layer_weight)).view(1, 77, -1)
        # x_std_cat = jt.cat(x_std, dim=0).permute(1, 2, 3, 0)        # NCHW ---> CHWN
        # x_miu_cat = jt.cat(x_miu, dim=0).permute(1, 2, 3, 0)        # NCHW ---> CHWN
        
        # std_result = nn.linear(x_std_cat, self.sigma_weight).permute(3, 0, 1, 2)        # CHWN ---> NCHW
        # miu_result = nn.linear(x_miu_cat, self.miu_weight).permute(3, 0, 1, 2)          # CHWN ---> NCHW

        # style_img_input = std_result*jt.randn_like(std_result) + miu_result  # 风格特征的采样

        return tem_x.view(1, 77, -1)



class vae_img_feature_change3(nn.Module):
    """将VAE特征维度变换到文本向量的特征维度"""    
    """在2的基础上使用GSC模块"""
    def __init__(self, in_channel=8, out_channel=77):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.bn1 = nn.GroupNorm(num_channels=in_channel, num_groups=4)
        self.activate_fn1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channel, out_channel//2, 3, 2, 1)     # , bias=False

        
        self.bn2 = nn.GroupNorm(num_channels=out_channel//2, num_groups=19)
        self.activate_fn2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channel//2, out_channel//2, 3, 1, 1)

        
        self.bn3 = nn.GroupNorm(num_channels=out_channel//2, num_groups=2)      # num_groups=2、19
        self.activate_fn3 = nn.SiLU()
        self.conv3 = nn.Conv2d(out_channel//2, out_channel, 3, 1, 1)
        # self.linear_layer = nn.Linear(8*64*64, 77*1024)
        # self.linear_layer_weight = nn.Parameter(jt.zeros((77*1024, 8*64*64)))
        # self.ln_layer = nn.LayerNorm(77*1024)

        # 将权重初始化为0
        self.conv1.weight = jt.zeros(self.conv1.weight.shape)
        # 如果需要，也可以初始化偏置为 0
        if self.conv1.bias is not None:
            self.conv1.bias = jt.zeros(self.conv1.bias.shape)
        # with jt.init_scope():
        # nn.init.constant(self.linear_layer.weight, "float32", 0.0)

        # # 如果你也想将偏置初始化为0，可以这样设置
        # if self.linear_layer.bias is not None:
        #     # with jt.init_scope():
        #     nn.init.constant(self.linear_layer.bias, "float32", 0.0)
        # self.miu_weight = nn.Parameter(jt.zeros((1, num_img)))
        # self.sigma_weight = nn.Parameter(jt.zeros((1, num_img)))

        
    def execute(self, x_std, x_miu):  # x为list, 第一个元素为风格特征, 第二个元素为真实噪声
        x_img = jt.cat([x_std, x_miu], dim=1)       # 将vae encode的结果进行拼接、展平
        
        tem_x = self.conv1(self.activate_fn1(self.bn1(x_img)))
        tem_x = self.conv2(self.activate_fn2(self.bn2(tem_x)))
        tem_x = self.conv3(self.activate_fn3(self.bn3(tem_x)))
        # tem_result = self.ln_layer(nn.linear(x_img, self.linear_layer_weight)).view(1, 77, -1)
        # x_std_cat = jt.cat(x_std, dim=0).permute(1, 2, 3, 0)        # NCHW ---> CHWN
        # x_miu_cat = jt.cat(x_miu, dim=0).permute(1, 2, 3, 0)        # NCHW ---> CHWN
        
        # std_result = nn.linear(x_std_cat, self.sigma_weight).permute(3, 0, 1, 2)        # CHWN ---> NCHW
        # miu_result = nn.linear(x_miu_cat, self.miu_weight).permute(3, 0, 1, 2)          # CHWN ---> NCHW

        # style_img_input = std_result*jt.randn_like(std_result) + miu_result  # 风格特征的采样

        return tem_x.view(1, 77, -1)


#region
class vae_img_feature_change3_1(nn.Module):
    """将VAE特征维度变换到文本向量的特征维度"""    
    """在2的基础上使用GSC模块"""
    def __init__(self, in_channel=8, out_channel=77):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.bn1 = nn.GroupNorm(num_channels=in_channel, num_groups=4)
        self.activate_fn1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channel, out_channel//2, 3, 2, 1, bias=False)

        
        self.bn2 = nn.GroupNorm(num_channels=out_channel//2, num_groups=19)
        self.activate_fn2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channel//2, out_channel//2, 3, 1, 1, bias=False)

        
        self.bn3 = nn.GroupNorm(num_channels=out_channel//2, num_groups=2)
        self.activate_fn3 = nn.SiLU()
        self.conv3 = nn.Conv2d(out_channel//2, out_channel, 3, 1, 1, bias=False)        # bias=False
        # self.linear_layer = nn.Linear(8*64*64, 77*1024)
        # self.linear_layer_weight = nn.Parameter(jt.zeros((77*1024, 8*64*64)))
        # self.ln_layer = nn.LayerNorm(77*1024)

        # 将权重初始化为0
        self.conv1.weight = jt.zeros(self.conv1.weight.shape)
        # 如果需要，也可以初始化偏置为 0
        if self.conv1.bias is not None:
            self.conv1.bias = jt.zeros(self.conv1.bias.shape)
        # with jt.init_scope():
        # nn.init.constant(self.linear_layer.weight, "float32", 0.0)

        # # 如果你也想将偏置初始化为0，可以这样设置
        # if self.linear_layer.bias is not None:
        #     # with jt.init_scope():
        #     nn.init.constant(self.linear_layer.bias, "float32", 0.0)
        # self.miu_weight = nn.Parameter(jt.zeros((1, num_img)))
        # self.sigma_weight = nn.Parameter(jt.zeros((1, num_img)))

        
    def execute(self, x_std, x_miu):  # x为list, 第一个元素为风格特征, 第二个元素为真实噪声
        x_img = jt.cat([x_std, x_miu], dim=1)       # 将vae encode的结果进行拼接、展平
        
        tem_x = self.conv1(self.activate_fn1(self.bn1(x_img)))
        tem_x = self.conv2(self.activate_fn2(self.bn2(tem_x)))
        tem_x = self.conv3(self.activate_fn3(self.bn3(tem_x)))
        # tem_result = self.ln_layer(nn.linear(x_img, self.linear_layer_weight)).view(1, 77, -1)
        # x_std_cat = jt.cat(x_std, dim=0).permute(1, 2, 3, 0)        # NCHW ---> CHWN
        # x_miu_cat = jt.cat(x_miu, dim=0).permute(1, 2, 3, 0)        # NCHW ---> CHWN
        
        # std_result = nn.linear(x_std_cat, self.sigma_weight).permute(3, 0, 1, 2)        # CHWN ---> NCHW
        # miu_result = nn.linear(x_miu_cat, self.miu_weight).permute(3, 0, 1, 2)          # CHWN ---> NCHW

        # style_img_input = std_result*jt.randn_like(std_result) + miu_result  # 风格特征的采样

        return tem_x.view(1, 77, -1)
#endregion


with torch.no_grad():
    for taskid in tqdm.tqdm(range(start_num, end_num)):
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")        # "stabilityai/stable-diffusion-2-1"  "/mnt/disk0/share/fc/pretrain_model/stable diffusion2.1"  ,  use_safetensors=True
        pipe.load_lora_weights(f"{root_path_weight}/style_{taskid:02d}/pytorch_lora_weights.bin")
        pipe.to("cuda:1")       # 主要还是要设置shell脚本中可见GPU,这个也应该配套。
        print(f"Model is on device: {pipe.device}")  # 打印模型所在的设备

        vae = pipe.vae
        weight_dtype = jt.float32

        img_linear = vae_img_feature_change3()
        img_linear.eval()
        
        # 导入权重
        import pickle
        with open(f"{root_path_weight}/style_{taskid:02d}/addce4.bin", 'rb') as f:    # 加载保存的字典 
            loaded_dict = pickle.load(f)
        
        for name, param in img_linear.state_dict().items():
            param.data = loaded_dict[name]

        # for i_key in loaded_dict.keys():
        #     loaded_dict[i_key] = jt.array(loaded_dict[i_key])
            
        #     if "weight" in i_key:
        #         img_linear.weight = loaded_dict[i_key]
        #     elif "bias" in i_key:
        #         img_linear.liner_proj.bias = loaded_dict[i_key]
        #     else:
        #         warnings.warn(f"{i_key} load false.")


        #region
        # # 导入加权值层
        # addce_layer = AddcE4()
        
        # import pickle
        # with open(f"{root_path_weight}/style_{taskid:02d}/addce2.bin", 'rb') as f:    # 加载保存的字典 
        #     loaded_dict = pickle.load(f)
        # for i_key in loaded_dict.keys():
        #     loaded_dict[i_key] = jt.array(loaded_dict[i_key])
            
        #     if "weight" in i_key:
        #         addce_layer.weight = loaded_dict[i_key]
        #     elif "bias" in i_key:
        #         addce_layer.liner_proj.bias = loaded_dict[i_key]
        #     else:
        #         warnings.warn(f"{i_key} load false.")
                
            # for name, params in addce_layer.state_dict().items():
            #     if name == i_key:
            #         params = loaded_dict[i_key]
            # addce_layer.liner_proj.weight = loaded_dict["liner_proj.weight"]
            # eval(f'addce_layer.{i_key} = loaded_dict["{i_key}"]')
        # addce_layer.fl_weight = loaded_dict[i_key]
        #endregion

        # 准备style_dataset, vae提取均值特征
        train_dataset = DreamBoothDataset(instance_data_root=f"{dataset_root}/{taskid:02d}/images")
        ave_img_mean = None
        ave_img_std = None
        for i_index in range(len(train_dataset)):
            i_img = train_dataset[i_index]
            if ave_img_mean is None:
                i_img_latent = vae.encode(jt.stack([i_img["instance_images"]]).to("cuda", dtype=weight_dtype)).latent_dist
                ave_img_mean = i_img_latent.mean
                ave_img_std = i_img_latent.std
            else:
                i_img_latent = vae.encode(jt.stack([i_img["instance_images"]]).to("cuda", dtype=weight_dtype)).latent_dist
                ave_img_mean += i_img_latent.mean
                ave_img_std += i_img_latent.std

        ave_img_mean /= len(train_dataset)
        ave_img_std /= len(train_dataset)


        # pipe.load_lora_weights(f"/home/fc/pycharm_project/JDiffusion-master/examples/dreambooth/style_5_4/style_all")
        # load json
        with open(f"{dataset_root}/{taskid:02d}/prompt.json", "r") as file:
            prompts = json.load(file)

        for id, prompt in prompts.items():
            set_seed_fc(args, taskid, prompt)
            prompt_total = prompt + f", style{taskid:02}"

            if taskid == 25:
                negative_prompt = ""
            else:
                # text, logos, worst quality, low quality, signature, watermark, username
                negative_prompt = "text, logos, worst quality, low quality, signature, watermark, username"
            # # image = pipe(prompt + "Vincent van Gogh Work Style",
            # #             height = 512,
            # #             width= 512, num_inference_steps=200).images[0]
            # if taskid == 6:
            #     # prompt_total = "origami " + prompt + f" in style{taskid:02}"
            #     prompt_total = prompt + f", style{taskid:02}"
            #     # negative_prompt = "lowres, text, logos, worst quality, low quality, signature, watermark, username"
            #     negative_prompt = ""
            # elif taskid == 13:
            #     # prompt_total = "origami " + prompt + f" in style{taskid:02}"
            #     prompt_total = prompt + f", style{taskid:02}"
            #     # negative_prompt = "lowres, text, logos, worst quality, low quality, signature, watermark, username"
            #     negative_prompt = ""
            # elif taskid == 0:
            #     # prompt_total = "origami " + prompt + f" in style{taskid:02}"
            #     prompt_total = prompt + f", style{taskid:02}"
            #     # negative_prompt = "lowres, text, logos, worst quality, low quality, signature, watermark, username"
            #     negative_prompt = ""
            # elif taskid == 2:
            #     # prompt_total = "origami " + prompt + f" in style{taskid:02}"
            #     prompt_total = prompt + f", style{taskid:02}"
            #     # negative_prompt = "lowres, text, logos, worst quality, low quality, signature, watermark, username"
            #     negative_prompt = ""
            # elif taskid == 4:
            #     # prompt_total = "origami " + prompt + f" in style{taskid:02}"
            #     prompt_total = prompt + f", style{taskid:02}"
            #     negative_prompt = "lowres, text, logos, worst quality, low quality, signature, watermark, username"
            #     # negative_prompt = ""
            # elif taskid == 7:
            #     # prompt_total = "origami " + prompt + f" in style{taskid:02}"
            #     # prompt_total = prompt + f", black background, no white in image, style{taskid:02}"  # f", black background, no white in image, style{taskid:02}" 感觉no white并不能避免出现白色，
            #     prompt_total = f"a style{taskid:02} " + prompt + f", black background" 
            #     # negative_prompt = "lowres, text, logos, worst quality, low quality, signature, watermark, username"
            #     negative_prompt = ""       # 在反向提示词中设置白色，来避免出现白色
            # else:
            #     prompt_total = prompt + f", style{taskid:02}"
            #     negative_prompt = "text, logos, worst quality, low quality, signature, watermark, username"
            #     # negative_prompt = "text, signature, watermark, username, low quality"
            #     # negative_prompt = ""
                
            # # prompt_total = f"a style{taskid:02} {prompt}"

            print(prompt_total)
            style_img_input = img_linear(ave_img_std, ave_img_mean)
            # 生成latents噪声
            # model_input = jt.randn_like(ave_img_std)     # encoder图像特征的采样
            # style_img_input = ave_img_std*jt.randn_like(ave_img_std) + ave_img_mean  # 风格特征的采样
            # model_latents = 0.01*style_img_input + model_input      # 图像特征与风格特征的加权求和, 列表的第一个元素需为风格特征

            image = pipe(prompt_total,
                        height = 512, width= 512,
                        negative_prompt=negative_prompt,
                        num_inference_steps=70, style_img_feature=style_img_input).images[0]        # latents = model_latents, 步数太大，生成的图像FID太大，生成图像太假了, guide_scale=9增强文本对图像的控制
            os.makedirs(f"{args.output_dir}/{taskid:02d}", exist_ok=True)
            image.save(f"{args.output_dir}/{taskid:02d}/{prompt}.png")