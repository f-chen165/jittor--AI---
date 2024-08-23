#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import sys
sys.path.append(fr"./diffusers_jittor/src")
# sys.path.append(fr"/home/fc/pycharm_project/JDiffusion-master/transformers_jittor/src")


import jittor as jt
import jittor.nn as nn
import argparse
import time
import itertools
import copy
import logging
import math
import os
import warnings
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


import numpy as np
import transformers
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from jittor import transform
from jittor.compatibility.optim import AdamW
from jittor.compatibility.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from JDiffusion import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from diffusers import DDPMScheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    convert_state_dict_to_diffusers,
)

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import sys
# 定义一个函数来将print输出写入文件
def print_to_file(file_path, my_net):
    # 保存原始的stdout
    original_stdout = sys.stdout
    # 将stdout重定向到文件
    with open(file_path, 'w') as file:
        sys.stdout = file
        print(my_net)
        # 确保刷新缓冲区，将内容写入文件
        sys.stdout.flush()
    # 将stdout重置回原始的标准输出
    sys.stdout = original_stdout





def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--num_process",
        type=int,
        default=1
    )
    parser.add_argument(
        "--train_text_encoder",
        # type=bool,
        default=True,
        # action="store_true",
        # required=True,
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--train_addce5",
        # type=bool,
        default=False,
        # action="store_true",
        # required=True,
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--train_vae_encoder",
        # type=bool,
        default=False,
        # action="store_true",
        # required=True,
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--train_addce",
        # type=bool,
        default=True,
        # action="store_true",
        # required=True,
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--learning_rate_text",
        type=float,
        default=2e-5,
        help="Initial learning rate for text encoder (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_vae",
        type=float,
        default=2e-5,
        help="Initial learning rate for text encoder (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_addce",
        type=float,
        default=2e-4,
        help="Initial learning rate for text encoder (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=rf"stabilityai/stable-diffusion-2-1",
        # default=rf"/mnt/disk0/share/fc/pretrain_model/stable diffusion2.1",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=f"/mnt/disk0/share/fc/dataset/A_test/00/images",
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="style00",
        required=False,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/fc/pycharm_project/JDiffusion-master/examples/dreambooth/style/style_00",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=0)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=None,
        required=False,
        help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )
    parser.add_argument(
        "--text_encoder_use_attention_mask",
        action="store_true",
        required=False,
        help="Whether to use attention mask for the text encoder",
    )
    parser.add_argument(
        "--validation_images",
        required=False,
        default=None,
        nargs="+",
        help="Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.",
    )
    parser.add_argument(
        "--class_labels_conditioning",
        required=False,
        default=None,
        help="The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=2,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=2,
        help=("lora_alpha."),
    )
    

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # logger is not available yet
    if args.class_data_dir is not None:
        warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
    if args.class_prompt is not None:
        warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
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
        print(f"======================{instance_data_root}=========================")
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

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            # print(f"{instance_image_name} in {self.instance_prompt}")
            if "07" not in self.instance_prompt:
                text_inputs = tokenize_prompt(
                    self.tokenizer, f"{instance_image_name}, {self.instance_prompt}",
                    tokenizer_max_length=self.tokenizer_max_length
                )
            else:   # {instance_image_name}, {self.instance_prompt}        a {self.instance_prompt} {instance_image_name}
                text_inputs = tokenize_prompt(
                    self.tokenizer, f"{instance_image_name}, {self.instance_prompt}",
                    tokenizer_max_length=self.tokenizer_max_length
                )
            # text_inputs = tokenize_prompt(
            #     self.tokenizer, f"{instance_image_name}, {self.instance_prompt}",
            #     tokenizer_max_length=self.tokenizer_max_length
            # )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            if self.class_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask

        return example


# 将所有的数据都用于权重的训练，不是分风格的来训练，增加数据量来缓解模型对先验知识的遗忘，当时感觉本至上还是无法完成对内容的控制，主要是有些风格的图像他就是缺类别;这样可能导致各个风格之间出现混叠
class DreamBoothDataset_fc(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
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

        self.instance_images_path = [i_img for i_num in Path(instance_data_root).iterdir() for i_img in Path(f"{i_num}/images").iterdir()]
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        # 额外的数据，防止模型遗忘以前的知识
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
        instance_image_sytle_num = self.instance_images_path[index % self.num_instance_images].__str__().split("/")[-3]
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer, f"{instance_image_name} in style_{instance_image_sytle_num}", tokenizer_max_length=self.tokenizer_max_length
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            if self.class_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask

        return example


def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    pixel_values = jt.stack(pixel_values)
    pixel_values = pixel_values.float()

    input_ids = jt.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    if has_attention_mask:
        batch["attention_mask"] = attention_mask

    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None, img_feature=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    if img_feature is not None:     # 将映射之后的风格特征加权到提示词向量中
        prompt_embeds = prompt_embeds + 0.01*img_feature[0]
        pass


    return prompt_embeds


def set_jittor_random_seed(args):
    """
    设置 Jittor 框架的随机种子，确保实验的可重复性。
    
    参数:
    - seed: 一个整数，用作随机种子。
    """
    if args.seed is None:
        args.seed = int(time.time())
    with open(f"{args.output_dir}/seed.txt", mode="w") as file_obj:
        file_obj.write(f"{args.seed}")
    # 设置 NumPy 的随机种子
    np.random.seed(args.seed)
    
    # 设置 Jittor 的随机种子
    jt.set_global_seed(args.seed)

    # 打印设置的随机种子值
    print(f"Random seed set to: {args.seed}")



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
        self.fl_weight = nn.Parameter(jt.zeros(self.input_num))     # 第一个参数乘加权风格特征
        self.fl_weight[1] = 1     # 第2个参数乘加权原本特征
        
        # self.add_parameters("fl_weight", jt.zeros(self.input_num - 1))

    def execute(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        tem = x[0]*self.fl_weight[0] + x[1]*self.fl_weight[1]       # 删除权重和为1的限制，
                
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
        # self.liner_proj.weight[:, half_weight:] = jt.ones((out_channel, half_weight))     # 通道的后一半对应的是真实特征; 设置单位矩阵        
        self.liner_proj.weight[:, half_weight:] = jt.diag(jt.array([1]*half_weight))     # 通道的后一半对应的是真实特征; 设置单位矩阵, 为保证模型        

        # self.fl_weight = nn.Parameter(jt.zeros(self.input_num))     # 第一个参数乘加权风格特征
        # self.fl_weight[1] = 1     # 第2个参数乘加权原本特征
        
        # self.add_parameters("fl_weight", jt.zeros(self.input_num - 1))

    def execute(self, x):  # x为list, 第一个元素为风格特征, 第二个元素为真实噪声
        x_cat = jt.cat(x, dim=1).permute(0, 2, 3, 1)        # NCHW ---> NHWC
        tem = self.liner_proj(x_cat)

        return tem.permute(0, 3, 1, 2)      # NHWC ----> NCHW
    
class AddcE4(nn.Module):
    """使用softmax 将特征的加权值之和限制为1"""     # 垃圾的很，完全没啥用

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

class AddcE5(nn.Module):
    """对各个风格的训练特征进行加权操作, 自适应学习加权参数"""    

    def __init__(self, num_img=10):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        
        self.miu_weight = nn.Parameter(jt.zeros((1, num_img)))
        self.sigma_weight = nn.Parameter(jt.zeros((1, num_img)))

        
    def execute(self, x_std, x_miu):  # x为list, 第一个元素为风格特征, 第二个元素为真实噪声
        x_std_cat = jt.cat(x_std, dim=0).permute(1, 2, 3, 0)        # NCHW ---> CHWN
        x_miu_cat = jt.cat(x_miu, dim=0).permute(1, 2, 3, 0)        # NCHW ---> CHWN
        
        std_result = nn.linear(x_std_cat, self.sigma_weight).permute(3, 0, 1, 2)        # CHWN ---> NCHW
        miu_result = nn.linear(x_miu_cat, self.miu_weight).permute(3, 0, 1, 2)          # CHWN ---> NCHW

        style_img_input = std_result*jt.randn_like(std_result) + miu_result  # 风格特征的采样

        return style_img_input      

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

        # # 将权重初始化为0
        # self.conv1.weight = jt.zeros(self.conv1.weight.shape)
        # # 如果需要，也可以初始化偏置为 0
        # if self.conv1.bias is not None:
        #     self.conv1.bias = jt.zeros(self.conv1.bias.shape)
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

        
        self.bn3 = nn.GroupNorm(num_channels=out_channel//2, num_groups=2)      # num_groups=2
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



class vae_img_feature_change3_bias_false(nn.Module):
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
        self.conv3 = nn.Conv2d(out_channel//2, out_channel, 3, 1, 1, bias=False)
        # self.linear_layer = nn.Linear(8*64*64, 77*1024)
        # self.linear_layer_weight = nn.Parameter(jt.zeros((77*1024, 8*64*64)))
        # self.ln_layer = nn.LayerNorm(77*1024)

        # # 将权重初始化为0
        # self.conv1.weight = jt.zeros(self.conv1.weight.shape)
        # # 如果需要，也可以初始化偏置为 0
        # if self.conv1.bias is not None:
        #     self.conv1.bias = jt.zeros(self.conv1.bias.shape)
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


def main(args):
    print("---------------mian--------------")
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()
    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    set_jittor_random_seed(args)    # 设置随机种子
    # 创建一个Summary Writer实例（默认保存在runs目录下）
    # writer = SummaryWriter(f'{args.output_dir}/exp-01')


    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )


    # 设置模型为训练模型，同时保持参数的梯度计算功能的关闭，主要是设置normal层的计算方式
    unet.train()
    text_encoder.train()
    vae.train()
    for name, param in unet.named_parameters():
        param.requires_grad = False
    for name, param in vae.named_parameters():
        param.requires_grad = False
    for name, param in text_encoder.named_parameters():
        param.requires_grad = False

    # We only train the additional adapter LoRA layers
   #if vae is not None:
   #    vae.requires_grad_(False)
   #text_encoder.requires_grad_(False)
   #unet.requires_grad_(False)
    # print(unet)
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = jt.float32

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # unet.to("cuda:1", dtype=weight_dtype)
    # if vae is not None:
    #     vae.to("cuda:1", dtype=weight_dtype)
    # text_encoder.to("cuda:1", dtype=weight_dtype)

    
    for name, param in unet.named_parameters():
        assert param.requires_grad == False, name
    # print(f"unet parameters: {sum(p.numel() for p in unet.parameters())}")
    # now we will add new LoRA weights to the attention layers
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        # lora_dropout=0.1,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj"],            # proj用于GEGIU, , "proj", , "proj_in", "proj_out"
    )
    # unet = get_peft_model(unet, unet_lora_config)
    unet.add_adapter(unet_lora_config)
    unet_params_to_update = [param for param in unet.parameters()]
    # tem_i = 0
    # for name, param in unet.named_parameters():
    #         if param.requires_grad:
    #             tem_i += 1
    #             print(name)
    # assert tem_i == len(unet_params_to_update), f"text no same len {len(unet_params_to_update)}  tem_i:{tem_i}"
    
    # print(f"unet_add_lora parameters: {sum(p.numel() for p in unet.parameters())}")
    # print(f"unet_add_lora grad parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}")

    if args.train_text_encoder:
        print(f"--------------------start train text_encoder---------------------")
        for name, param in text_encoder.named_parameters():
            assert param.requires_grad == False, name

        # print(f"CLIP parameters: {sum(p.numel() for p in text_encoder.parameters())}")
        
        text_encoder_lora_config = LoraConfig(
                            r=args.rank,
                            lora_alpha=args.lora_alpha,
                            # lora_dropout=0.1,
                            init_lora_weights="gaussian",
                            target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],            # proj用于GEGIU, , "out_proj",, "out_proj"
                            )
        text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)
        text_encoder_params_to_update = [param for param in text_encoder.parameters()]      # if param.requires_grad
        tem_i = 0
        # for name, param in text_encoder.named_parameters():
        #     if param.requires_grad:
        #         tem_i += 1
        #         print(name)
        # assert tem_i == len(text_encoder_params_to_update), f"text no same len {len(text_encoder_params_to_update)}, tem_i:{tem_i}"
        # for i in range(20):
        print(f"***************************************")
        print(f"unet_train parameter len:{len(unet_params_to_update)}")
        print(f"text_train parameter:{len(text_encoder_params_to_update)}")
            
        # print(f"CLIP_add_lora parameters: {sum(p.numel() for p in text_encoder.parameters())}")
        # print(f"CLIP_add_lora grad parameters: {sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)}")
        # text_encoder.add_adapter(text_encoder_lora_config)
    
    if args.train_vae_encoder:
        print(f"-----------------------start train vae_encoder--------------------------")
        print_to_file("./vae_jittor.yaml", vae)
        for name, param in vae.encoder.named_parameters():
            assert param.requires_grad == False, name

        # print(f"CLIP parameters: {sum(p.numel() for p in text_encoder.parameters())}")
        
        vae_encoder_lora_config = LoraConfig(
                            r=args.rank,
                            lora_alpha=args.lora_alpha,
                            # lora_dropout=0.1,
                            init_lora_weights="gaussian",
                            target_modules=["to_q", "to_v", "to_k", "to_out.0"],            # proj用于GEGIU, , "out_proj",, "out_proj"
                            )
        vae = get_peft_model(vae, vae_encoder_lora_config)
        vae_encoder_params_to_update = [param for param in vae.encoder.parameters() if param.requires_grad]      # if param.requires_grad
        tem_i = 0
        # for name, param in text_encoder.named_parameters():
        #     if param.requires_grad:
        #         tem_i += 1
        #         print(name)
        # assert tem_i == len(text_encoder_params_to_update), f"text no same len {len(text_encoder_params_to_update)}, tem_i:{tem_i}"
        # for i in range(20):
        print_to_file("./vae_jittor_lora.yaml", vae)
        print(f"***************************************")
        print(f"unet_train parameter len:{len(unet_params_to_update)}")
        print(f"text_train parameter:{len(text_encoder_params_to_update)}")
        print(f"vae_encoder_train parameter:{len(vae_encoder_params_to_update)}")
            
        # print(f"CLIP_add_lora parameters: {sum(p.numel() for p in text_encoder.parameters())}")
        # print(f"CLIP_add_lora grad parameters: {sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)}")
        # text_encoder.add_adapter(text_encoder_lora_config)


    # text_encoder_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder))
    # 使用函数
    # print_to_file('./text_encode_lora.yaml', text_encoder)
    # print(unet)

    # 风格特征的尺度变换层
    add_ce_layer = vae_img_feature_change3()
    add_ce_layer.train()        # 训练模式

    if args.train_addce:
        print("--------------------------train AddcE2---------------------------------")
        for i_parameters in add_ce_layer.parameters():  # 开启addce层的参数训练
            i_parameters.requires_grad = True
        add_ce_layer_params_to_update = [param for param in add_ce_layer.parameters() if param.requires_grad]
    else:
        add_ce_layer_params_to_update = None

    # Optimizer creation
    text_lr = (
        args.learning_rate
        if args.learning_rate_text is None
        else args.learning_rate_text
    )
    vae_lr = (
        args.learning_rate
        if args.learning_rate_vae is None
        else args.learning_rate_vae
    )
    add_ce_layer_lr = (
        args.learning_rate
        if args.learning_rate_addce is None
        else args.learning_rate_addce
    )


    # 可能性没有列举完全后续优化
    if args.train_addce and args.train_text_encoder and args.train_vae_encoder:
        params_to_optimize = [
            {"params": unet_params_to_update, "lr": args.learning_rate},
            {"params": text_encoder_params_to_update, "lr": text_lr},
            {"params": vae_encoder_params_to_update, "lr": vae_lr},
            {"params": add_ce_layer_params_to_update, "lr": add_ce_layer_lr},
        ]
    elif args.train_text_encoder and args.train_vae_encoder:
        params_to_optimize = [
            {"params": unet_params_to_update, "lr": args.learning_rate},
            {"params": text_encoder_params_to_update, "lr": text_lr},
            {"params": vae_encoder_params_to_update, "lr": vae_lr},
        ]
    elif args.train_text_encoder and args.train_addce:
        params_to_optimize = [
            {"params": unet_params_to_update, "lr": args.learning_rate},
            {"params": text_encoder_params_to_update, "lr": text_lr},
            {"params": add_ce_layer_params_to_update, "lr": add_ce_layer_lr},
        ]
        print(f"----------------------addcf_rl3:{add_ce_layer_lr}------------------------")
    elif args.train_text_encoder:
        params_to_optimize = [
            {"params": unet_params_to_update, "lr": args.learning_rate},
            {"params": text_encoder_params_to_update, "lr": text_lr},
        ]
        print(f"----------------------unet_rl:{args.learning_rate}  text_rl:{text_lr}------------------------")
    else:
        params_to_optimize = unet_params_to_update
    # params_to_optimize = (
    #     [
    #         {"params": unet_params_to_update, "lr": args.learning_rate},
    #         {"params": text_encoder_params_to_update, "lr": text_lr,},
    #     ]
    #     if args.train_text_encoder and args.train_vae_encoder
    #     else unet_params_to_update
    #     )
    print(f"text_rl:{text_lr}, unet_rl: {args.learning_rate}, vae_rl: {vae_lr}")
    optimizer = AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # optimizer = AdamW(
    #     list(unet.parameters()),
    #     lr=args.learning_rate,
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,
    # )

    pre_computed_encoder_hidden_states = None
    pre_computed_class_prompt_encoder_hidden_states = None

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        class_num=args.num_class_images,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        encoder_hidden_states=pre_computed_encoder_hidden_states,
        class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
        tokenizer_max_length=args.tokenizer_max_length,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, False),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.num_process,
        num_training_steps=args.max_train_steps * args.num_process,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    tracker_config = vars(copy.deepcopy(args))
    tracker_config.pop("validation_images")

    # Train!
    total_batch_size = args.train_batch_size * args.num_process * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    print(f"  num train timesteps = {noise_scheduler.config.num_train_timesteps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    # 计算同一风格的平均特征
    # ave_img_latent = []
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

        # ave_img_latent.append(i_img_latent)     # 因该存储均值和方差，目的是为了改变噪声的分布
    

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=False,
    )

    logs_fc_loss = []  # 记录损失
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to("cuda", dtype=weight_dtype)

            # Convert images to latent space
            model_input = vae.encode(pixel_values).latent_dist.sample()     # encoder图像特征的采样
            
            # model_input = add_ce_layer([style_img_input, model_input])      # 图像特征与风格特征的加权求和, 列表的第一个元素需为风格特征
            model_input = model_input * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            style_img_input = add_ce_layer(ave_img_std, ave_img_mean)
            # ave_img_std*jt.randn_like(ave_img_std) + ave_img_mean  # 风格特征的采样
            # noise = jt.randn_like(model_input) + 0.01*style_img_input       # 0.01做为一个超参数，后续调调
            noise = jt.randn_like(model_input)
            bsz, channels, height, width = model_input.shape
            # Sample a random timestep for each image
            timesteps = jt.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), 
            ).to(device=model_input.device)
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = encode_prompt(
                text_encoder,
                batch["input_ids"],
                batch["attention_mask"],
                text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                img_feature=style_img_input
            )

            if unet.config.in_channels == channels * 2:
                noisy_model_input = jt.cat([noisy_model_input, noisy_model_input], dim=1)

            if args.class_labels_conditioning == "timesteps":
                class_labels = timesteps
            else:
                class_labels = None

            # Predict the noise residual
            model_pred = unet(
                noisy_model_input,
                timesteps,
                encoder_hidden_states,
                class_labels=class_labels,
                return_dict=False,
            )[0]

            # if model predicts variance, throw away the prediction. we will only train on the
            # simplified training objective. This means that all schedulers using the fine tuned
            # model must be configured to use one of the fixed variance variance types.
            if model_pred.shape[1] == 6:
                model_pred, _ = jt.chunk(model_pred, 2, dim=1)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = nn.mse_loss(model_pred, target)
            loss.backward()
            

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            # 记录损失到TensorBoard
            
            # writer.add_scalar(f'{args.output_dir}/Loss/train', loss, global_step)
            global_step += 1

            logs = {"loss": loss.detach().item()}
            logs_fc_loss.append(logs["loss"])
            #logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    # writer.close()  # 训练结束后，关闭Summary Writer
    with open(f'{args.output_dir}/loss.txt', mode="w") as file_obj:
        for i_loss in logs_fc_loss:
            file_obj.write(f"{i_loss}"+"\n")


    unet = unet.to(jt.float32)
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

    text_encoder_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder))
    if args.train_vae_encoder:
        vae_encoder_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(vae))
    else:
        vae_encoder_state_dict = None


    LoraLoaderMixin.save_lora_weights(
        save_directory=args.output_dir,
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=text_encoder_state_dict,
        vae_lora_layers=vae_encoder_state_dict,
        safe_serialization=False,
        # weight_name="vincent_style_lora_weights.bin"
    )

    weight_dict = add_ce_layer.state_dict()
    for i_key in weight_dict.keys():    # jt.var()转化为numpy
        weight_dict[i_key] = weight_dict[i_key].numpy()
    
    import pickle
    # 保存权重字典到文件
    with open(f"{args.output_dir}/addce4.bin", 'wb') as f:
        pickle.dump(weight_dict, f)



def main1(args):
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    set_jittor_random_seed(args)    # 设置随机种子



    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # We only train the additional adapter LoRA layers
   #if vae is not None:
   #    vae.requires_grad_(False)
   #text_encoder.requires_grad_(False)
   #unet.requires_grad_(False)
    # print(unet)
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = jt.float32

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # unet.to("cuda:1", dtype=weight_dtype)
    # if vae is not None:
    #     vae.to("cuda:1", dtype=weight_dtype)
    # text_encoder.to("cuda:1", dtype=weight_dtype)

    for name, param in unet.named_parameters():
        assert param.requires_grad == False, name
    # print(f"unet parameters: {sum(p.numel() for p in unet.parameters())}")
    # now we will add new LoRA weights to the attention layers
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        # lora_dropout=0.1,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj"],            # proj用于GEGIU, , "proj"
    )
    # unet = get_peft_model(unet, unet_lora_config)
    unet.add_adapter(unet_lora_config)
    unet_params_to_update = [param for param in unet.parameters()]
    # tem_i = 0
    # for name, param in unet.named_parameters():
    #         if param.requires_grad:
    #             tem_i += 1
    #             print(name)
    # assert tem_i == len(unet_params_to_update), f"text no same len {len(unet_params_to_update)}  tem_i:{tem_i}"
    
    # print(f"unet_add_lora parameters: {sum(p.numel() for p in unet.parameters())}")
    # print(f"unet_add_lora grad parameters: {sum(p.numel() for p in unet.parameters() if p.requires_grad)}")

    if args.train_text_encoder:
        print(f"--------------------start train text_encoder---------------------")
        for name, param in text_encoder.named_parameters():
            assert param.requires_grad == False, name

        # print(f"CLIP parameters: {sum(p.numel() for p in text_encoder.parameters())}")
        
        text_encoder_lora_config = LoraConfig(
                            r=args.rank,
                            lora_alpha=args.lora_alpha,
                            # lora_dropout=0.1,
                            init_lora_weights="gaussian",
                            target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],            # proj用于GEGIU, , "out_proj",, "out_proj"
                            )
        text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)
        text_encoder_params_to_update = [param for param in text_encoder.parameters()]      # if param.requires_grad
        tem_i = 0
        # for name, param in text_encoder.named_parameters():
        #     if param.requires_grad:
        #         tem_i += 1
        #         print(name)
        # assert tem_i == len(text_encoder_params_to_update), f"text no same len {len(text_encoder_params_to_update)}, tem_i:{tem_i}"
        # for i in range(20):
        print(f"***************************************")
        print(f"unet_train parameter len:{len(unet_params_to_update)}")
        print(f"text_train parameter:{len(text_encoder_params_to_update)}")
            
        # print(f"CLIP_add_lora parameters: {sum(p.numel() for p in text_encoder.parameters())}")
        # print(f"CLIP_add_lora grad parameters: {sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)}")
        # text_encoder.add_adapter(text_encoder_lora_config)
    
    if args.train_vae_encoder:
        print(f"-----------------------start train vae_encoder--------------------------")
        print_to_file("./vae_jittor.yaml", vae)
        for name, param in vae.encoder.named_parameters():
            assert param.requires_grad == False, name

        # print(f"CLIP parameters: {sum(p.numel() for p in text_encoder.parameters())}")
        
        vae_encoder_lora_config = LoraConfig(
                            r=args.rank,
                            lora_alpha=args.lora_alpha,
                            # lora_dropout=0.1,
                            init_lora_weights="gaussian",
                            target_modules=["to_q", "to_v", "to_k", "to_out.0"],            # proj用于GEGIU, , "out_proj",, "out_proj"
                            )
        vae = get_peft_model(vae, vae_encoder_lora_config)
        vae_encoder_params_to_update = [param for param in vae.encoder.parameters() if param.requires_grad]      # if param.requires_grad
        tem_i = 0
        # for name, param in text_encoder.named_parameters():
        #     if param.requires_grad:
        #         tem_i += 1
        #         print(name)
        # assert tem_i == len(text_encoder_params_to_update), f"text no same len {len(text_encoder_params_to_update)}, tem_i:{tem_i}"
        # for i in range(20):
        print_to_file("./vae_jittor_lora.yaml", vae)
        print(f"***************************************")
        print(f"unet_train parameter len:{len(unet_params_to_update)}")
        print(f"text_train parameter:{len(text_encoder_params_to_update)}")
        print(f"vae_encoder_train parameter:{len(vae_encoder_params_to_update)}")
            
        # print(f"CLIP_add_lora parameters: {sum(p.numel() for p in text_encoder.parameters())}")
        # print(f"CLIP_add_lora grad parameters: {sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)}")
        # text_encoder.add_adapter(text_encoder_lora_config)


    # text_encoder_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder))
    # 使用函数
    # print_to_file('./text_encode_lora.yaml', text_encoder)
    # print(unet)

    # 风格特征与随机噪声的加权层
    add_ce_layer = AddcE5()

    if args.train_addce:
        print("--------------------------train AddcE2---------------------------------")
        for i_parameters in add_ce_layer.parameters():  # 开启addce层的参数训练
            i_parameters.requires_grad = True
        add_ce_layer_params_to_update = [param for param in add_ce_layer.parameters() if param.requires_grad]
    else:
        add_ce_layer_params_to_update = None

    # Optimizer creation
    text_lr = (
        args.learning_rate
        if args.learning_rate_text is None
        else args.learning_rate_text
    )
    vae_lr = (
        args.learning_rate
        if args.learning_rate_vae is None
        else args.learning_rate_vae
    )
    add_ce_layer_lr = (
        args.learning_rate
        if args.learning_rate_addce is None
        else args.learning_rate_addce
    )


    # 可能性没有列举完全后续优化
    if args.train_addce and args.train_text_encoder and args.train_vae_encoder:
        params_to_optimize = [
            {"params": unet_params_to_update, "lr": args.learning_rate},
            {"params": text_encoder_params_to_update, "lr": text_lr},
            {"params": vae_encoder_params_to_update, "lr": vae_lr},
            {"params": add_ce_layer_params_to_update, "lr": add_ce_layer_lr},
        ]
    elif args.train_text_encoder and args.train_vae_encoder:
        params_to_optimize = [
            {"params": unet_params_to_update, "lr": args.learning_rate},
            {"params": text_encoder_params_to_update, "lr": text_lr},
            {"params": vae_encoder_params_to_update, "lr": vae_lr},
        ]
    elif args.train_text_encoder and args.train_addce:
        params_to_optimize = [
            {"params": unet_params_to_update, "lr": args.learning_rate},
            {"params": text_encoder_params_to_update, "lr": text_lr},
            {"params": add_ce_layer_params_to_update, "lr": add_ce_layer_lr},
        ]
        print(f"----------------------addcf_rl:{add_ce_layer_lr}------------------------")
    elif args.train_text_encoder:
        params_to_optimize = [
            {"params": unet_params_to_update, "lr": args.learning_rate},
            {"params": text_encoder_params_to_update, "lr": text_lr},
        ]
    else:
        params_to_optimize = unet_params_to_update
    # params_to_optimize = (
    #     [
    #         {"params": unet_params_to_update, "lr": args.learning_rate},
    #         {"params": text_encoder_params_to_update, "lr": text_lr,},
    #     ]
    #     if args.train_text_encoder and args.train_vae_encoder
    #     else unet_params_to_update
    #     )
    print(f"text_rl:{text_lr}, unet_rl: {args.learning_rate}, vae_rl: {vae_lr}")
    optimizer = AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # optimizer = AdamW(
    #     list(unet.parameters()),
    #     lr=args.learning_rate,
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,
    # )

    pre_computed_encoder_hidden_states = None
    pre_computed_class_prompt_encoder_hidden_states = None

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        class_num=args.num_class_images,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        encoder_hidden_states=pre_computed_encoder_hidden_states,
        class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
        tokenizer_max_length=args.tokenizer_max_length,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, False),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.num_process,
        num_training_steps=args.max_train_steps * args.num_process,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    tracker_config = vars(copy.deepcopy(args))
    tracker_config.pop("validation_images")

    # Train!
    total_batch_size = args.train_batch_size * args.num_process * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    print(f"  num train timesteps = {noise_scheduler.config.num_train_timesteps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    # 计算同一风格的平均特征
    # ave_img_latent = []
    ave_img_mean = []
    ave_img_std = []
    for i_index in range(len(train_dataset)):
        i_img = train_dataset[i_index]
        i_img_latent = vae.encode(jt.stack([i_img["instance_images"]]).to("cuda", dtype=weight_dtype)).latent_dist
        
        ave_img_mean.append(i_img_latent.mean)
        ave_img_std.append(i_img_latent.std)


    # ave_img_mean /= len(train_dataset)
    # ave_img_std /= len(train_dataset)

    # ave_img_latent.append(i_img_latent)     # 因该存储均值和方差，目的是为了改变噪声的分布
    

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=False,
    )


    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to("cuda", dtype=weight_dtype)

            # Convert images to latent space
            model_input = vae.encode(pixel_values).latent_dist.sample()     # encoder图像特征的采样

            model_input = model_input * vae.config.scaling_factor
            style_img_input = add_ce_layer(ave_img_std, ave_img_mean)       # 均值和方差加权，并进行风格特征的采样
            # style_img_input = ave_img_std*jt.randn_like(ave_img_std) + ave_img_mean  # 风格特征的采样
            
            # model_input = style_img_input + model_input      # 图像特征与风格特征的加权求和, 列表的第一个元素需为风格特征; 后续在考虑系数的问题, 0初始化自适应系数应该不太行, 无法进行梯度的反向更新
            # Sample noise that we'll add to the latents
            noise = jt.randn_like(model_input)
            bsz, channels, height, width = model_input.shape
            # Sample a random timestep for each image
            timesteps = jt.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), 
            ).to(device=model_input.device)
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = encode_prompt(
                text_encoder,
                batch["input_ids"],
                batch["attention_mask"],
                text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                img_feature=style_img_input
            )

            if unet.config.in_channels == channels * 2:
                noisy_model_input = jt.cat([noisy_model_input, noisy_model_input], dim=1)

            if args.class_labels_conditioning == "timesteps":
                class_labels = timesteps
            else:
                class_labels = None

            # Predict the noise residual
            model_pred = unet(
                noisy_model_input,
                timesteps,
                encoder_hidden_states,
                class_labels=class_labels,
                return_dict=False,
            )[0]

            # if model predicts variance, throw away the prediction. we will only train on the
            # simplified training objective. This means that all schedulers using the fine tuned
            # model must be configured to use one of the fixed variance variance types.
            if model_pred.shape[1] == 6:
                model_pred, _ = jt.chunk(model_pred, 2, dim=1)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = nn.mse_loss(model_pred, target)
            loss.backward()
            

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            logs = {"loss": loss.detach().item()}
            #logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    unet = unet.to(jt.float32)
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

    text_encoder_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder))
    if args.train_vae_encoder:
        vae_encoder_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(vae))
    else:
        vae_encoder_state_dict = None


    LoraLoaderMixin.save_lora_weights(
        save_directory=args.output_dir,
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=text_encoder_state_dict,
        vae_lora_layers=vae_encoder_state_dict,
        safe_serialization=False,
        # weight_name="vincent_style_lora_weights.bin"
    )


    if args.train_addce:
        weight_dict = add_ce_layer.state_dict()
        for i_key in weight_dict.keys():    # jt.var()转化为numpy
            weight_dict[i_key] = weight_dict[i_key].numpy()
        
        import pickle
        # 保存权重字典到文件
        with open(f"{args.output_dir}/addce2.bin", 'wb') as f:
            pickle.dump(weight_dict, f)





if __name__ == "__main__":
    args = parse_args()

    if args.train_addce5:      # 训练addce5
        main1(args)
    else:
        main(args)