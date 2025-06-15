import sys
import os, numpy

import torch
import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset
import json

import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from annotator.util import resize_image, HWC3



current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # '/data/jingshirou/diffusers/examples/controlnet
parent_dir = os.path.dirname(parent_dir) # /data/jingshirou/diffusers/examples
parent_dir = os.path.dirname(parent_dir) # /data/jingshirou/diffusers
src_path = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_path)

from diffusers import StableDiffusionXLControlNetPipeline, UniPCMultistepScheduler, DiffusionPipeline, AutoencoderKL
from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl_fuse import StableDiffusionXLMultiControlNetPipeline

from diffusers.utils import load_image

from diffusers.models.controlnets.controlora_multi import ControlNetLoraMulti
# from diffusers.models.controlnets.controlora_multi import ControlNetLoraMulti
from typing import Any

from diffusers import ControlNetModel
from torchvision import transforms
import peft
from peft import get_peft_model, LoraConfig, TaskType
from diffusers import UNet2DConditionModel
from safetensors.torch import load_file
import gradio as gr


os.environ['GRADIO_TEMP_DIR'] = './tmp'

CKPT_DIR = './ckpts'
CKPT_SDXL_DIR = os.path.join(CKPT_DIR, 'sdxl')
CKPT_BASECN_DIR = os.path.join(CKPT_DIR, 'ctrlora-xl-base')
CKPT_LORAS_DIR = os.path.join(CKPT_DIR, 'ctrlora-xl-loras')

model: Any = None
last_ckpts = (None, None, None)
vae: Any = None

det_choices = [
    'none', 'canny', 'sketch', 'seg', 'depth', 'normal', 'openpose', 'densepose', 'grayscale', 
    'jpeg','downsample','lineart_color','color_prompt','palette','masker_brush',  # base conditions
    'mlsd', 'palette', 'pixel', 'illusion', 'desnow','dehaze','APT','brush_mixed_lineart', 'brush_mixed_sketch',
    'Low-light-enhancement', 'ControlSketch'    # proposed new conditions
]

add_prompts = {
    'General-short': 'award-winning, best quality, high resolution, extremely detailed, photorealistic, HDR, cinematic. ',
    'General-long': 'award-winning, best quality, high resolution, extremely detailed, photorealistic, HDR, cinematic. ',
    'Realistic': 'RAW photo, 8K UHD, DSLR, film grain, highres, high resolution, high detail, extremely detailed, soft lighting, award winning photography',
}

neg_prompts = {
    'General-short': 'bad anatomy, jagged edges, pixelated, lowres, blurry, deformed, distorted, extra limbs, duplicate, monochrome, grayscale, dull, washed out, dull, low contrast, painting, crayon, graphite, abstract glitch, blurry',
    'General-long': 'bad anatomy, jagged edges, pixelated, lowres, blurry, deformed, distorted, extra limbs, duplicate, monochrome, grayscale, dull, washed out, dull, low contrast, painting, crayon, graphite, abstract glitch, blurry',
    'General-human': 'bad anatomy, wrong anatomy, bad proportions, gross proportions, deformed, deformed iris, deformed pupils, inaccurate eyes, cross-eye, cloned face, bad hands, mutation, mutated hands, mutation hands, mutated fingers, mutation fingers, fused fingers, too many fingers, extra fingers, extra digit, missing fingers, fewer digits, malformed limbs, inaccurate limb, extra limbs, missing limbs, floating limbs, disconnected limbs, extra arms, extra legs, missing arms, missing legs, error, bad legs, error legs, bad feet, long neck, disfigured, amputation, dehydrated, nude, thighs, cleavage',
    'Realistic': 'semi-realistic, CGI, 3D, render, sketch, drawing, comic, cartoon, anime, vector art',
    '2.5D': 'sketch, drawing, comic, cartoon, anime, vector art',
    'Painting': 'photorealistic, CGI, 3D, render',
}

def modify_condition(condition):
    if condition == "sketch":
        condition = "edgesketch"
    elif condition == "seg":
        condition = "oneformer_ade20k"
    elif condition == "normal":
        condition = "lotus_normal"
    elif condition == "lineart_color":
        condition = "lineart_lineart_color"
    return condition

def build_model(sd_ckpt, cn_ckpt, lora_ckpts, conditions, lora_num=1):
    global model, last_ckpts, vae
    assert sd_ckpt is not None
    assert cn_ckpt is not None
    assert lora_ckpts is not None
    assert len(lora_ckpts) == lora_num

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if last_ckpts != (sd_ckpt, cn_ckpt, lora_ckpts):
        print(f'Loading checkpoints')
        sd_ckpt = os.path.join(CKPT_SDXL_DIR, sd_ckpt)
        # import pdb; pdb.set_trace()
        if sd_ckpt.startswith("./ckpts/sdxl/converted"):
            unet = UNet2DConditionModel.from_pretrained(
                sd_ckpt, subfolder="unet", revision=None, variant=None, use_safetensors=False
            )
        else:
            unet = UNet2DConditionModel.from_pretrained(
                sd_ckpt, subfolder="unet", revision=None, variant=None
            )
        controlnet = ControlNetLoraMulti.from_unet(unet, load_weights_from_unet=False, ft_with_lora=False)
        vae_path = os.path.join(sd_ckpt, "vae")
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
        
        lora_config = LoraConfig(
            r=64,
            target_modules=["to_q", "to_k", "to_v"],
            lora_alpha=128)

        state_dict = load_file(os.path.join(CKPT_BASECN_DIR, cn_ckpt))
        controlnet = peft.PeftModel(controlnet, lora_config, adapter_name=conditions[0])
        controlnet.base_model.model.load_state_dict(state_dict, strict=False)
        controlnet.set_adapter(conditions[0])

        print("loaded first lora: ",conditions[0])

        for i, (condition, lora_ckpt) in enumerate(zip(conditions, lora_ckpts)):
            if i==0: 
                continue
            controlnet.add_adapter(condition)
            state_dict = load_file(os.path.join(CKPT_LORAS_DIR, lora_ckpt))
            controlnet.load_adapter(
                state_dict=state_dict, adapter_name=condition, is_trainable=False)
            # controlnet.base_model.model.load_state_dict(state_dict, strict=False)
        print("loaded all lora: ",conditions)

        controlnet.to(dtype=torch.float16, device=DEVICE)
        # vae.to(dtype=torch.float16, device=DEVICE)
        model = controlnet

        last_ckpts = (sd_ckpt, cn_ckpt, tuple(lora_ckpts))
        print(f'Checkpoints loaded')



def detect(det, input_image, detect_resolution, image_resolution):
    global preprocessor
    print("selecting detectors and detect...")
    preprocessor = None

    if det == 'none':
        preprocessor = None
    elif det == 'canny':
        from annotator.canny import CannyDetector
        if not isinstance(preprocessor, CannyDetector):
            preprocessor = CannyDetector()
        params = dict(low_threshold=100, high_threshold=200)
    elif det == 'hed':
        from annotator.hed import HEDdetector
        if not isinstance(preprocessor, HEDdetector):
            preprocessor = HEDdetector()
        params = dict()
    elif det == 'seg':
        from annotator.uniformer import UniformerDetector
        if not isinstance(preprocessor, UniformerDetector):
            preprocessor = UniformerDetector()
        params = dict()
    elif det in ['depth', 'normal']:
        from annotator.midas import MidasDetector
        if not isinstance(preprocessor, MidasDetector):
            preprocessor = MidasDetector()
        params = dict()
    elif det == 'openpose':
        from annotator.openpose import OpenposeDetector
        if not isinstance(preprocessor, OpenposeDetector):
            preprocessor = OpenposeDetector()
        params = dict()
    elif det == 'sketch' or det == 'hedsketch':
        from annotator.hedsketch import HEDSketchDetector
        if not isinstance(preprocessor, HEDSketchDetector):
            preprocessor = HEDSketchDetector()
        params = dict()
    elif det == 'grayscale':
        from annotator.grayscale import GrayscaleConverter
        if not isinstance(preprocessor, GrayscaleConverter):
            preprocessor = GrayscaleConverter()
        params = dict()
    elif det == 'blur':
        from annotator.blur import Blurrer
        if not isinstance(preprocessor, Blurrer):
            preprocessor = Blurrer()
        ksize = np.random.randn() * 0.5 + 0.5
        ksize = int(ksize * (50 - 5)) + 5
        ksize = ksize * 2 + 1
        params = dict(ksize=ksize)
    elif det == 'pad':
        from annotator.pad import Padder
        if not isinstance(preprocessor, Padder):
            preprocessor = Padder()
        params = dict(top_ratio=0.50, bottom_ratio=0.50, left_ratio=0.50, right_ratio=0.50)
    elif det in ['lineart', 'lineart_coarse']:
        from annotator.lineart import LineartDetector
        if not isinstance(preprocessor, LineartDetector):
            preprocessor = LineartDetector()
        params = dict(coarse=(det == 'lineart_coarse'))
    elif det in ['lineart_anime', 'lineart_anime_with_color_prompt']:
        from annotator.lineart_anime import LineartAnimeDetector
        if not isinstance(preprocessor, LineartAnimeDetector):
            preprocessor = LineartAnimeDetector()
        params = dict()
    elif det == 'shuffle':
        from annotator.shuffle import ContentShuffleDetector
        if not isinstance(preprocessor, ContentShuffleDetector):
            preprocessor = ContentShuffleDetector()
        params = dict()
    elif det == 'mlsd':
        from annotator.mlsd import MLSDdetector
        if not isinstance(preprocessor, MLSDdetector):
            preprocessor = MLSDdetector()
        thr_v = np.random.rand() * 1.9 + 0.1  # [0.1, 2.0]
        thr_d = np.random.rand() * 19.9 + 0.1  # [0.1, 20.0]
        params = dict(thr_v=thr_v, thr_d=thr_d)
    elif det == 'palette':
        from annotator.palette import PaletteDetector
        if not isinstance(preprocessor, PaletteDetector):
            preprocessor = PaletteDetector()
        params = dict()
    elif det == 'pixel':
        from annotator.pixel import Pixelater
        if not isinstance(preprocessor, Pixelater):
            preprocessor = Pixelater()
        n_colors = np.random.randint(8, 17)  # [8,16] -> 3-4 bits
        scale = np.random.randint(4, 9)  # [4,8]
        params = dict(n_colors=n_colors, scale=scale, down_interpolation=cv2.INTER_LANCZOS4)
    elif det == 'illusion':
        from annotator.illusion import IllusionConverter
        if not isinstance(preprocessor, IllusionConverter):
            preprocessor = IllusionConverter()
        params = dict()
    elif det == 'densepose':
        from annotator.densepose import DenseposeDetector
        if not isinstance(preprocessor, DenseposeDetector):
            preprocessor = DenseposeDetector()
        params = dict()
    else:
        raise ValueError('Unknown preprocessor')
    print("selected det: ", det)
    # import pdb;pdb.set_trace()
    if isinstance(input_image, dict):
        input_image = input_image['composite']

    with torch.no_grad():
        input_image = HWC3(input_image)
        print("input_image get...")
        if preprocessor is not None:
            resized_image = resize_image(input_image, detect_resolution)
            detected_map = preprocessor(resized_image, **params)
            if det == 'depth':
                detected_map = detected_map[0]
            elif det == 'normal':
                detected_map = detected_map[1]
        else:
            detected_map = input_image
        detected_map = HWC3(detected_map)
        H, W, C = resize_image(input_image, image_resolution).shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    print("finish detected map!!")
    return detected_map


def preprocess_condition(image, image_resolution):
    # import pdb;pdb.set_trace()
    if isinstance(image, np.ndarray):                 # numpy -> PIL
        image = Image.fromarray(image.astype(np.uint8))
    elif isinstance(image, torch.Tensor):             # Tensor(C,H,W) -> PIL
        image = transforms.ToPILImage()(image)
    image = image.convert("RGB")
    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(image_resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    # image = image.convert("RGB")
    image = conditioning_image_transforms(image) # change to tensor
    return image.unsqueeze(0)




def get_test_condition(cond_img, image_resolution):
    # condition_image = Image.open(cond_img).convert("RGB")
    global vae
    # import pdb;pdb.set_trace()
    vae = vae.to(dtype=torch.float32)
    control_tensor = preprocess_condition(cond_img, image_resolution)
    control_tensor = control_tensor.to(device=vae.device, dtype=vae.dtype)
    control_tensor = control_tensor

    latent = vae.encode(control_tensor).latent_dist.sample()
    latent = latent * vae.config.scaling_factor
    latent.to(dtype=torch.float16).to(dtype=torch.float32)
    
    return latent

def process(det, detected_image, image_resolution, prompt, n_prompt, num_samples, scheduler_steps, guess_mode, strength, guidance_scale, seed, sd_ckpt, cn_ckpt, lora_ckpt):
    # det, detected_image, prompt, n_prompt, num_samples, scheduler_steps, guess_mode, strength, guidance_scale, seed, sd_ckpt, cn_ckpt, lora_ckpt
    global model, last_ckpts
    det = modify_condition(det)
    build_model(sd_ckpt, cn_ckpt, [lora_ckpt], [det], lora_num=1)
    if isinstance(detected_image, dict):
        detected_image = detected_image['composite']
    detected_image = HWC3(detected_image)
    # import pdb;pdb.set_trace()

    latent = get_test_condition(detected_image, image_resolution)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if sd_ckpt.startswith("./ckpts/sdxl/converted"):
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            sd_ckpt, 
            controlnet=model, 
            torch_dtype=torch.float16, 
            use_safetensors=False
        )
    else:
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            sd_ckpt, 
            controlnet=model, 
            torch_dtype=torch.float16
        )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # pipe.enable_model_cpu_offload()
    pipe.to(dtype=torch.float16,device=device)
    latent = latent.to(dtype=torch.float16,device=device)
    print("Successfully loaded pipeline!")
    generator = torch.manual_seed(seed)
    # import pdb;pdb.set_trace()

    image = pipe(
        prompt, num_inference_steps=scheduler_steps, generator=generator, image=latent, negative_prompt=n_prompt, guidance_scale=guidance_scale
    ).images[0]
    return [detected_image] + [image]

def process2(det, det2, detected_image, image_resolution, detected_image2, prompt, n_prompt, num_samples, scheduler_steps, guess_mode, strength, guidance_scale, seed, eta, sd_ckpt, cn_ckpt, lora_ckpt, lora2_ckpt, lora_weight, lora2_weight):
    global model, last_ckpts
    det = modify_condition(det)
    det2 = modify_condition(det2)
    build_model(sd_ckpt, cn_ckpt, [lora_ckpt, lora2_ckpt], [det, det2], lora_num=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(detected_image, dict):
        detected_image = detected_image['composite']
    detected_image = HWC3(detected_image)
    if isinstance(detected_image2, dict):
        detected_image2 = detected_image2['composite']
    detected_image2 = HWC3(detected_image2)
    latent = get_test_condition(detected_image, image_resolution)
    latent2 = get_test_condition(detected_image2, image_resolution)
    pipe = StableDiffusionXLMultiControlNetPipeline.from_pretrained(
        sd_ckpt, 
        controlnet=model,
        torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    generator = torch.manual_seed(seed)
    image_mix = pipe(
        prompt=prompt, 
        num_inference_steps=20, 
        generator=generator, 
        image=[latent,latent2], 
        negative_prompt=n_prompt, 
        guidance_scale=guidance_scale,
        prompt_2=det,
        negative_prompt_2=det2,
        sigmas=[lora_weight,lora2_weight]
    ).images[0]
    return [detected_image,detected_image2] + [image_mix]

def listdir_r(path):
    path = os.path.expanduser(path)
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(path, followlinks=True) for f in fn]
    files = [f[len(path) + 1:] for f in files]
    return files

def list_model_dirs(path):
    path = os.path.expanduser(path)
    print(path)
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return sorted(dirs)


def update_ckpts():
    sd_ckpt = gr.Dropdown(label='Select stable diffusion checkpoint', choices=list_model_dirs(CKPT_SDXL_DIR))
    cn_ckpt = gr.Dropdown(label='Select base controlnet checkpoint', choices=sorted(listdir_r(CKPT_BASECN_DIR)))
    lora_ckpt = gr.Dropdown(label='Select lora checkpoint', choices=sorted(listdir_r(CKPT_LORAS_DIR)))
    return sd_ckpt, cn_ckpt, lora_ckpt

def update_ckpts2():
    sd_ckpt = gr.Dropdown(label='Select stable diffusion checkpoint', choices=list_model_dirs(CKPT_SDXL_DIR))
    cn_ckpt = gr.Dropdown(label='Select base controlnet checkpoint', choices=sorted(listdir_r(CKPT_BASECN_DIR)))
    lora_ckpt = gr.Dropdown(label='Select lora1 checkpoint', choices=sorted(listdir_r(CKPT_LORAS_DIR)))
    lora2_ckpt = gr.Dropdown(label='Select lora2 checkpoint', choices=sorted(listdir_r(CKPT_LORAS_DIR)))
    return sd_ckpt, cn_ckpt, lora_ckpt, lora2_ckpt

# def update_prompt(prompt, evt: gr.SelectData):
#     if evt.selected:
#         prompt = prompt.strip() + '\n' + f'[[ {add_prompts[evt.value]} ]]'
#     else:
#         prompt = prompt.replace(f'[[ {add_prompts[evt.value]} ]]', '').replace('\n\n', '\n')
#     prompt = prompt.strip()
#     if prompt.endswith(']]'):
#         prompt = prompt + '\n'
#     return prompt


# def update_n_prompt(n_prompt, evt: gr.SelectData):
#     if evt.selected:
#         n_prompt = n_prompt.strip() + '\n' + f'[[ {neg_prompts[evt.value]} ]]'
#     else:
#         n_prompt = n_prompt.replace(f'[[ {neg_prompts[evt.value]} ]]', '').replace('\n\n', '\n')
#     n_prompt = n_prompt.strip()
#     if n_prompt.endswith(']]'):
#         n_prompt = n_prompt + '\n'
#     return n_prompt

def update_prompt(prompt: str, evt: gr.SelectData):
    """单选版本：正向 preset 只能存在 0 或 1 条。"""
    new_tag = add_prompts[evt.value]

    # ① 先把所有已知 preset 通通移除
    for t in add_prompts.values():
        prompt = prompt.replace(t + ' ', '').replace(t, '')

    # ② 如果本次操作是“选中”，再前置新 tag
    if evt.selected:
        prompt = new_tag + prompt.lstrip()

    # ③ 统一去重空格
    prompt = ' '.join(prompt.split())
    return prompt


def update_n_prompt(n_prompt: str, evt: gr.SelectData):
    """单选版本：负向 preset 只能存在 0 或 1 条。"""
    new_tag = neg_prompts[evt.value]

    for t in neg_prompts.values():
        n_prompt = n_prompt.replace(t + ' ', '').replace(t, '')

    if evt.selected:
        n_prompt = f'{new_tag} ' + n_prompt.lstrip()

    n_prompt = ' '.join(n_prompt.split())
    return n_prompt


def tab1():
    with gr.Row():
        sd_ckpt = gr.Dropdown(label='Select stable diffusion checkpoint', choices=sorted(listdir_r(CKPT_SDXL_DIR)), scale=3)
        cn_ckpt = gr.Dropdown(label='Select base controlnet checkpoint', choices=sorted(listdir_r(CKPT_BASECN_DIR)), scale=3)
        lora_ckpt = gr.Dropdown(label='Select lora checkpoint', choices=sorted(listdir_r(CKPT_LORAS_DIR)), scale=3)
        refresh_button = gr.Button(value="Refresh", scale=1)
        run_button = gr.Button(value="Run", scale=1, variant='primary')

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                prompt = gr.Textbox(label="Prompt", lines=3)
                a_prompt_choices = gr.CheckboxGroup(choices=list(add_prompts.keys()), type="value", label="Examples")

            with gr.Group():
                n_prompt = gr.Textbox(label="Negative Prompt", lines=2)
                n_prompt_choices = gr.CheckboxGroup(choices=list(neg_prompts.keys()), type="value", label="Examples")

            with gr.Accordion("Basic options", open=True):
                with gr.Group():
                    with gr.Row():
                        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)
                        num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                        image_resolution = gr.Slider(label="Image Resolution", minimum=768, maximum=1536, value=1024, step=64)
                        guess_mode = gr.Checkbox(label='Guess Mode', value=False, visible=False)
                    with gr.Row():
                        scheduler_steps = gr.Slider(label="Scheduler Steps", minimum=1, maximum=100, value=20, step=1)
                        # eta = gr.Slider(label="DDIM ETA", minimum=0.0, maximum=1.0, value=0.0, step=0.01)
                        strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                        guidance_scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)

            with gr.Accordion("Condition", open=True):
                with gr.Row():
                    input_image = gr.ImageEditor(sources=['upload', 'clipboard'], type="numpy", layers=False)
                    detected_image = gr.ImageEditor(sources=['upload', 'clipboard'], type="numpy", layers=False)
                det = gr.Radio(choices=det_choices, type="value", value="none", label="Preprocessor")
                detect_resolution = gr.Slider(label="Preprocessor Resolution", minimum=512, maximum=1536, value=1024, step=1)
                detect_button = gr.Button(value="Detect")

        with gr.Column(scale=1):
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", object_fit='scale-down', height=650)

    refresh_button.click(fn=update_ckpts, inputs=[], outputs=[sd_ckpt, cn_ckpt, lora_ckpt])
    a_prompt_choices.select(fn=update_prompt, inputs=[prompt], outputs=[prompt])
    n_prompt_choices.select(fn=update_n_prompt, inputs=[n_prompt], outputs=[n_prompt])
    detect_button.click(fn=detect, inputs=[det, input_image, detect_resolution, image_resolution], outputs=[detected_image])
    run_button.click(fn=process, inputs=[det, detected_image, image_resolution, prompt, n_prompt, num_samples, scheduler_steps, guess_mode, strength, guidance_scale, seed, sd_ckpt, cn_ckpt, lora_ckpt], outputs=[result_gallery])



def tab2():
    with gr.Row():
        sd_ckpt = gr.Dropdown(label='Select stable diffusion checkpoint', choices=sorted(listdir_r(CKPT_SDXL_DIR)), scale=3)
        cn_ckpt = gr.Dropdown(label='Select base controlnet checkpoint', choices=sorted(listdir_r(CKPT_BASECN_DIR)), scale=3)
        lora_ckpt = gr.Dropdown(label='Select lora1 checkpoint', choices=sorted(listdir_r(CKPT_LORAS_DIR)), scale=3)
        lora2_ckpt = gr.Dropdown(label='Select lora2 checkpoint', choices=sorted(listdir_r(CKPT_LORAS_DIR)), scale=3)
        refresh_button = gr.Button(value="Refresh", scale=1)
        run_button = gr.Button(value="Run", scale=1, variant='primary')

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                prompt = gr.Textbox(label="Prompt", lines=3)
                a_prompt_choices = gr.CheckboxGroup(choices=list(add_prompts.keys()), type="value", label="Examples")

            with gr.Group():
                n_prompt = gr.Textbox(label="Negative Prompt", lines=2)
                n_prompt_choices = gr.CheckboxGroup(choices=list(neg_prompts.keys()), type="value", label="Examples")

            with gr.Accordion("Basic options", open=True):
                with gr.Group():
                    with gr.Row():
                        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)
                        num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                        image_resolution = gr.Slider(label="Image Resolution", minimum=768, maximum=1536, value=1024, step=64)
                        guess_mode = gr.Checkbox(label='Guess Mode', value=False, visible=False)
                    with gr.Row():
                        scheduler_steps = gr.Slider(label="Scheduler Steps", minimum=1, maximum=100, value=20, step=1)
                        strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                        guidance_scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                    with gr.Row():
                        lora_weight = gr.Slider(label="Condition 1 Weight", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                        lora2_weight = gr.Slider(label="Condition 2 Weight", minimum=0.0, maximum=2.0, value=1.0, step=0.01)

            with gr.Accordion("Condition1", open=True):
                with gr.Row():
                    input_image = gr.ImageEditor(sources=['upload', 'clipboard'], type="numpy", layers=False)
                    detected_image = gr.ImageEditor(sources=['upload', 'clipboard'], type="numpy", layers=False)
                det = gr.Radio(choices=det_choices, type="value", value="none", label="Preprocessor")
                detect_resolution = gr.Slider(label="Preprocessor Resolution", minimum=512, maximum=1536, value=1024, step=1)
                detect_button = gr.Button(value="Detect")

            with gr.Accordion("Condition2", open=True):
                with gr.Row():
                    input_image2 = gr.ImageEditor(sources=['upload', 'clipboard'], type="numpy", layers=False)
                    detected_image2 = gr.ImageEditor(sources=['upload', 'clipboard'], type="numpy", layers=False)
                det2 = gr.Radio(choices=det_choices, type="value", value="none", label="Preprocessor")
                detect_resolution2 = gr.Slider(label="Preprocessor Resolution", minimum=512, maximum=1536, value=1024, step=1)
                detect_button2 = gr.Button(value="Detect")
        
        with gr.Column(scale=1):
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", object_fit='scale-down', height=650)

    refresh_button.click(fn=update_ckpts2, inputs=[], outputs=[sd_ckpt, cn_ckpt, lora_ckpt, lora2_ckpt])
    a_prompt_choices.select(fn=update_prompt, inputs=[prompt], outputs=[prompt])
    n_prompt_choices.select(fn=update_n_prompt, inputs=[n_prompt], outputs=[n_prompt])
    detect_button.click(fn=detect, inputs=[det, input_image, detect_resolution, image_resolution], outputs=[detected_image])
    detect_button2.click(fn=detect, inputs=[det2, input_image2, detect_resolution2, image_resolution], outputs=[detected_image2])
    run_button.click(fn=process2, inputs=[det, det2, detected_image, image_resolution, detected_image2, prompt, n_prompt, num_samples, scheduler_steps, guess_mode, strength, guidance_scale, seed, sd_ckpt, cn_ckpt, lora_ckpt, lora2_ckpt, lora_weight, lora2_weight], outputs=[result_gallery])


def main():
    blocks = gr.Blocks().queue()
    with blocks:
        with gr.Row():
            gr.Markdown("## CtrLoRA-XL")
        with gr.Tab(label='Single condition'):
            tab1()
        with gr.Tab(label='Two conditions'):
            tab2()
    blocks.launch(server_name='0.0.0.0')


if __name__ == '__main__':
    main()
