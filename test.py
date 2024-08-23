import sys
import os
sys.path.append(f"/home/fanchen/fc_jittor_diffusion_b/B_competition/diffusers_jittor/src")
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

def set_seed(seed_fc_my):
    # 设置 NumPy 的随机种子
    np.random.seed(seed_fc_my)
    
    # 设置 Jittor 的随机种子
    jt.set_global_seed(seed_fc_my)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_result_8_22",
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

args = parse_args()


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



start_num = 0
end_num = start_num + 1
dataset_root = "./B"      # 保存每个风格的训练数据
root_path_weight = f"./style_8_20_text_45_35_addce_ip_text_13_001_700_3"

# 读取保存生成图像的物体信息和seed的json文件
with open(f"{root_path_weight}/seed_img.json", "r") as file:
    seed_img_tem = json.load(file)
seed_img = dict()
for i_key in seed_img_tem.keys():   # 将字典键值转化为int类型
    seed_img[int(i_key)] = seed_img_tem[i_key]

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


        # with open(f"{dataset_root}/{taskid:02d}/prompt.json", "r") as file:
        #     prompts = json.load(file)
        prompts = seed_img[taskid]

        for prompt, seed_i in prompts:
            set_seed(seed_i)    # 设置图像的随机种子
            prompt_total = prompt + f", style{taskid:02}"

            if taskid == 25:
                negative_prompt = ""
            else:
                # text, logos, worst quality, low quality, signature, watermark, username
                negative_prompt = "text, logos, worst quality, low quality, signature, watermark, username"
            

            print(prompt_total)
            style_img_input = img_linear(ave_img_std, ave_img_mean)
            
            image = pipe(prompt_total,
                        height = 512, width= 512,
                        negative_prompt=negative_prompt,
                        num_inference_steps=70, style_img_feature=style_img_input).images[0]        # latents = model_latents, 步数太大，生成的图像FID太大，生成图像太假了, guide_scale=9增强文本对图像的控制
            os.makedirs(f"{args.output_dir}/{taskid:02d}", exist_ok=True)
            image.save(f"{args.output_dir}/{taskid:02d}/{prompt}.png")