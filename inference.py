
import torch
import cv2
import numpy as np
import sys
import os, numpy
from PIL import Image
from datasets import load_dataset
import json
import argparse
# from diffusers.models.controlnets.controlnet_VAE import ControlNetModelVAE_before
# from diffusers.models.controlnets.controlnet_VAE import ControlNetModelVAE

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # '/data/jingshirou/diffusers/examples/controlnet
parent_dir = os.path.dirname(parent_dir) # /data/jingshirou/diffusers/examples
parent_dir = os.path.dirname(parent_dir) # /data/jingshirou/diffusers
src_path = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_path)

from diffusers import StableDiffusionXLControlNetPipeline, UniPCMultistepScheduler, DiffusionPipeline, AutoencoderKL
from diffusers.utils import load_image

from diffusers.models.controlnets.controlora_multi import ControlNetLoraMulti
# from diffusers.models.controlnets.controlora_multi import ControlNetLoraMulti

from diffusers import ControlNetModel
from torchvision import transforms
import peft
from peft import get_peft_model, LoraConfig, TaskType
from diffusers import UNet2DConditionModel
from safetensors.torch import load_file

turn = [2925]
turn_list = [(0,2925)]
ckpt = 2925
# 7500*x+1~15
image_w = 7
image_h = len(turn_list)

resize = 1024

base_model_path = "/data/jingshirou/diffusers/examples/controlnet/stable-diffusion-xl-base-1.0"

vae_path = os.path.join(base_model_path, "vae")
vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float32)

# test_id_list = [2, 3, 4, 6, 8, 9]

def preprocess_condition(image):
    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    image = image.convert("RGB")
    image = conditioning_image_transforms(image)
    return image.unsqueeze(0)


def get_test_examples(condition, test_num):
    dataset_root = "/data/jingshirou/diffusers/examples/controlnet/gen_test_img/BASE-02/SDXL-Ctrlora"
    dataset_path = os.path.join(dataset_root, f"{condition}-28514-result")
    annotation_list_path = os.path.join(dataset_path, "prompt.json")
    ds = load_dataset('json', data_files=annotation_list_path, split='train')
    test_num = min(test_num, len(ds))
    ds = ds.shuffle(seed=42).select(range(test_num))

    condition_image_names = ds['source']
    images_name = ds['target']
    prompts = ds['prompt']
    
    img_paths = [os.path.join(dataset_path, image_name) for image_name in images_name]
    anno_paths = [os.path.join(dataset_path, condition_image_name) for condition_image_name in condition_image_names]
        

    return img_paths, anno_paths, prompts


def get_test_images(img_path, anno_path, prompt):

    image = Image.open(img_path).convert("RGB")
    condition_image = Image.open(anno_path).convert("RGB")

    control_tensor = preprocess_condition(condition_image)
    control_tensor = control_tensor.to(device=vae.device, dtype=vae.dtype)

    latent = vae.encode(control_tensor).latent_dist.sample()
    latent = latent * vae.config.scaling_factor
    latent.to(dtype=torch.float16)
    prompt_pos = "award-winning, best quality, high resolution, extremely detailed, photorealistic, HDR, cinematic. "
    prompt = prompt_pos + prompt

    return image, condition_image, latent, prompt



def local_mean_std(image, window_size, sigma):
    assert len(image.shape) in [2, 3]
    assert window_size >= 0
    if window_size > 0 and window_size % 2 == 0:
        window_size += 1
    image = image.astype(np.float64)
    mean = cv2.GaussianBlur(image, (window_size, window_size), sigma, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.GaussianBlur(image ** 2, (window_size, window_size), sigma, borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    return mean, std


def local_mean_std_fast(image, window_size):
    assert len(image.shape) in [2, 3]
    assert window_size > 0

    if window_size % 2 == 0:
        window_size += 1
    k = int(window_size // 2)
    area = window_size * window_size

    if len(image.shape) == 2:
        image = np.pad(image, ((k, k), (k, k)), mode='reflect')
    elif len(image.shape) == 3:
        image = np.pad(image, ((k, k), (k, k), (0, 0)), mode='reflect')

    image = image.astype(np.float64)
    integral = cv2.integral(image)
    integral_sq = cv2.integral(image ** 2)

    sum_val = (integral[window_size:, window_size:]
               - integral[:-window_size, window_size:]
               - integral[window_size:, :-window_size]
               + integral[:-window_size, :-window_size])
    sum_sq = (integral_sq[window_size:, window_size:]
              - integral_sq[:-window_size, window_size:]
              - integral_sq[window_size:, :-window_size]
              + integral_sq[:-window_size, :-window_size])
    mean = sum_val / area
    std = np.sqrt(sum_sq / area - mean ** 2)

    return mean, std


def adpative_color_correction(image, reference, window_size, sigma=None):
    image = np.array(image)
    reference = np.array(reference.resize((1024, 1024)))
    if sigma is not None:
        img_mean, img_std = local_mean_std(image, window_size=window_size, sigma=sigma)
        ref_mean, ref_std = local_mean_std(reference, window_size=window_size, sigma=sigma)
    else:
        img_mean, img_std = local_mean_std_fast(image, window_size=window_size)
        ref_mean, ref_std = local_mean_std_fast(reference, window_size=window_size)

    eps = 1e-8
    img_fixed = (image - img_mean) / (img_std + eps) * ref_std + ref_mean
    img_fixed = np.clip(img_fixed, 0, 255).astype(np.uint8)
    img_fixed = Image.fromarray(img_fixed)

    return img_fixed



def correct(image, condition, ref):
    img = image
    if condition == "downsample":
        correct_image = adpative_color_correction(img, ref, window_size=128, sigma=128/6)
    elif condition == "jpeg" or condition == "pixel":
        correct_image = adpative_color_correction(img, ref, window_size=256, sigma=256/6)
    elif condition == "palette":
        correct_image = adpative_color_correction(img, ref, window_size=768, sigma=768/6)

    return correct_image
    
def get_parser():
    parser = argparse.ArgumentParser(description="args")
    parser.add_argument("--conditions", type=str, required=False, help='conditions, like canny, edge')
    parser.add_argument("--experiment", type=str, required=True, help='BASE-01, BASE-02')
    parser.add_argument("--ckpt", type=int, required=True, help='checkpoint')
    parser.add_argument("--cn_path", type=str, required=True, help='controlnet path')
    parser.add_argument("--test_num", type=int, required=True, help='test number')
    parser.add_argument("--seed", type=int, required=True, help='test number')
    parser.add_argument("--prefix", type=str, required=True, help='prefix')
    parser.add_argument("--guidance_scale", type=float, required=True, help='guidance scale')
    return parser

def main():
    args = get_parser().parse_args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_num = args.test_num
    if args.conditions:
        conditions = args.conditions
    else:
        conditions = "depth,lotus_normal,canny,edgesketch,oneformer_ade20k,openpose"
    condition_list = [(i + 1, condition) for i, condition in enumerate(conditions.split(","))]

    prompt_pos = "award-winning, best quality, high resolution, extremely detailed."
    prompt_neg = "bad anatomy, jagged edges, pixelated, lowres, blurry, deformed, distorted, extra limbs, duplicate, monochrome, grayscale, dull, washed out, dull, low contrast, painting, crayon, graphite, abstract glitch, blurry"
    
    color_correct_conditions = ["downsample", "jpeg", "palette", "pixel"]

    unet = UNet2DConditionModel.from_pretrained(
        base_model_path, subfolder="unet", revision=None, variant=None
    )
    
    lora_config = LoraConfig(
        r=64,
        target_modules=["to_q", "to_k", "to_v"],
        lora_alpha=128)

    ckpt = args.ckpt
    controlnet_folder = args.cn_path
    controlnet_ckpt = os.path.join(controlnet_folder, f"checkpoint-{ckpt}-densepose-controlnet")
    ckpt_path = os.path.join(controlnet_ckpt,'diffusion_pytorch_model.safetensors')
    state_dict = load_file(ckpt_path)

    for (condition_id, condition) in condition_list:
        
        img_paths, anno_paths, prompts = get_test_examples(condition, test_num)
        
        controlnet = ControlNetLoraMulti.from_unet(unet, load_weights_from_unet=False, ft_with_lora=False)
        
        condition_pre = condition
        if condition == "hedsketch" or condition == "hed" or condition == "pidisketch":
            condition = "edgesketch"
        elif condition == "lineart":
            condition = "lineart_lineart_color"
        elif condition == "normal":
            condition = "lotus_normal"
        elif condition == "seg":
            condition = "oneformer_ade20k"
        elif condition == "skeleton":
            condition = "openpose"
        elif condition == "outpainting" or condition == "inpainting_brush":
            condition = "masker_brush_border_masker"
        
        controlnet = peft.PeftModel(controlnet, lora_config, adapter_name=condition)
        controlnet.to(torch.float16)
        controlnet.base_model.model.load_state_dict(state_dict, strict=False)
        
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_path, 
            controlnet=controlnet, 
            torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)

        test_image_save_path = f"/data/jingshirou/diffusers/examples/controlnet/gen_test_img/{args.experiment}/SDXL-Ctrlora/{condition_pre}-{ckpt}-{args.prefix}"
        if not os.path.exists(test_image_save_path):
            os.makedirs(test_image_save_path)
    
        sample_folder = f"{test_image_save_path}/sample"
        img_folder = f"{test_image_save_path}/img"
        control_folder = f"{test_image_save_path}/control"
        prompt_name = f"{test_image_save_path}/prompt.txt"
        json_name = f"{test_image_save_path}/generate_image.jsonl"
        with open(json_name, 'w') as f_json, open(prompt_name, 'w') as f_prompt:
            pass
            
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        if not os.path.exists(control_folder):
            os.makedirs(control_folder)
        
        gen_images = []
        condition_images = []
        origin_images = []

        for test_id in range(len(img_paths)):
            prompt = prompts[test_id]
            img_path = img_paths[test_id]
            anno_path = anno_paths[test_id]

            origin_image, condition_image, latent, prompt = get_test_images(img_path, anno_path, prompt)
            condition_images.append(condition_image)
            origin_images.append(origin_image)
 
            generator = torch.manual_seed(args.seed)
            
            image = pipe(
                prompt, num_inference_steps=50, generator=generator, image=latent, negative_prompt=prompt_neg, guidance_scale=args.guidance_scale
            ).images[0]

            if condition in color_correct_conditions:
                image = correct(image, condition, condition_image)
            
            gen_images.append(image)

            image.save(f"{sample_folder}/{test_id}.png")
            condition_image.save(f"{control_folder}/{test_id}.png")
            origin_image.save(f"{img_folder}/{test_id}.png")
            
            with open(prompt_name, "a") as prompt_file:
                prompt_file.write(f"{prompt}\n")
            with open(json_name, "a") as json_file:
                json_file.write(json.dumps({
                    "image_name": f"sample/{test_id}.png",
                }) + "\n")
            print(f"generate {condition} {test_id} images done!")
        
        
        # # import pdb; pdb.set_trace()
        # with open(prompt_name, "a") as prompt_file: 
        #     with open(json_name, "a") as json_file:
        #         for i in range(len(img_paths)):
        #             gen_image = gen_images[i]
        #             condition_image = condition_images[i]
        #             origin_image = origin_images[i]
        #             prompt = prompts[i]

        #             gen_image.save(f"{sample_folder}/{i}.png")
        #             condition_image.save(f"{control_folder}/{i}.png")
        #             origin_image.save(f"{img_folder}/{i}.png")

        #             prompt_file.write(f"{prompt}\n")
                    
        #             json_file.write(json.dumps({
        #                 "image_name": f"sample/{i}.png",
        #             }) + "\n")
                    
        #             print(f"save {condition} {i} images done!")

            
if __name__ == "__main__":
    

    main()
