import sys
sys.path.append(f"./diffusers_jittor/src")
import time
import numpy as np
import random
import argparse

import json, os, tqdm, torch
import jittor as jt

from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline


os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

def set_jittor_random_seed(args):
    """
    设置 Jittor 框架的随机种子，确保实验的可重复性。
    
    参数:
    - seed: 一个整数，用作随机种子。
    """
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


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
        default="./output_result_8_19_add_negative_1445_700_4_same_prompt",
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
# set_jittor_random_seed(args)

start_num = 14
end_num = start_num + 14
dataset_root = "./B"

with torch.no_grad():
    for taskid in tqdm.tqdm(range(start_num, end_num)):
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")        # "stabilityai/stable-diffusion-2-1"  "/mnt/disk0/share/fc/pretrain_model/stable diffusion2.1"  ,  use_safetensors=True
        pipe.load_lora_weights(f"./style_8_19_text_14_45_700_4/style_{taskid:02d}/pytorch_lora_weights.bin")
        pipe.to("cuda:1")       # 主要还是要设置shell脚本中可见GPU,这个也应该配套。
        print(f"Model is on device: {pipe.device}")  # 打印模型所在的设备
        # pipe.load_lora_weights(f"/home/fc/pycharm_project/JDiffusion-master/examples/dreambooth/style_5_4/style_all")
        # load json
        with open(f"{dataset_root}/{taskid:02d}/prompt.json", "r") as file:
            prompts = json.load(file)

        for id, prompt in prompts.items():
            set_seed_fc(args, taskid, prompt)

            prompt_total = prompt + f", style{taskid:02}"
            # prompt_total = f"a style{taskid:02} {prompt}"
            if taskid == 25:
                negative_prompt = ""
            else:
                negative_prompt = "lowres, text, logos, worst quality, low quality, signature, watermark, username"
            
            #region
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
            #     negative_prompt = "text, logos, signature, watermark, username"
            #     # negative_prompt = ""
            # elif taskid == 4:
            #     # prompt_total = "origami " + prompt + f" in style{taskid:02}"
            #     prompt_total = prompt + f", style{taskid:02}"
            #     # negative_prompt = "lowres, text, logos, worst quality, low quality, signature, watermark, username"
            #     negative_prompt = ""
            # elif taskid == 7:
            #     # prompt_total = "origami " + prompt + f" in style{taskid:02}"
            #     # prompt_total = prompt + f", black background, no white in image, style{taskid:02}"  # f", black background, no white in image, style{taskid:02}" 感觉no white并不能避免出现白色，
            #     prompt_total = f"a style{taskid:02} " + prompt + f", black background" 
            #     # negative_prompt = "lowres, text, logos, worst quality, low quality, signature, watermark, username"
            #     negative_prompt = ""       # 在反向提示词中设置白色，来避免出现白色
            # elif taskid == 9:
            #     prompt_total = prompt + f", style{taskid:02}"
            #     negative_prompt = "text, logos, signature, watermark, username"
            # else:
            #     prompt_total = prompt + f", style{taskid:02}"
            #     negative_prompt = "text, logos, worst quality, low quality, signature, watermark, username"
            #     # negative_prompt = "text, signature, watermark, username, low quality"
            #     # negative_prompt = ""
                
            # # prompt_total = f"a style{taskid:02} {prompt}"
            #endregion


            print(prompt_total)
            image = pipe(prompt_total,
                        height = 512, width= 512,
                        negative_prompt = negative_prompt,
                        num_inference_steps = 70).images[0]        # 步数太大，生成的图像FID太大，生成图像太假了, guide_scale=9增强文本对图像的控制
            os.makedirs(f"{args.output_dir}/{taskid:02d}", exist_ok=True)
            image.save(f"{args.output_dir}/{taskid:02d}/{prompt}.png")   # _ipadapter3_65_no