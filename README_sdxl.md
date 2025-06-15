# ControlNet training example for Stable Diffusion XL (SDXL)

The `train_controlnet_sdxl.py` script shows how to implement the ControlNet training procedure and adapt it for [Stable Diffusion XL](https://huggingface.co/papers/2307.01952).

## Running locally with PyTorch

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd in the `examples/controlnet` folder and run
```bash
pip install -r requirements_sdxl.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell (e.g., a notebook)

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

When running `accelerate config`, if we specify torch compile mode to True there can be dramatic speedups.

## Circle filling dataset

The original dataset is hosted in the [ControlNet repo](https://huggingface.co/lllyasviel/ControlNet/blob/main/training/fill50k.zip). We re-uploaded it to be compatible with `datasets` [here](https://huggingface.co/datasets/fusing/fill50k). Note that `datasets` handles dataloading within the training script.

## Training

Our training examples use two test conditioning images. They can be downloaded by running

```sh
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png

wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

Then run `huggingface-cli login` to log into your Hugging Face account. This is needed to be able to push the trained ControlNet parameters to Hugging Face Hub.

```bash
export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="path to save model"

accelerate launch train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --validation_steps=100 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --seed=42 \
 --push_to_hub
```
huggingface-cli login
export MODEL_DIR="/data/jingshirou/diffusers/examples/controlnet/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="sdxl_weight_densepose_VAE"
export HF_DATASETS_TRUST_REMOTE_CODE=True

accelerate launch train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=/data/jingshirou/diffusers/examples/controlnet/densepose_1024 \
 --mixed_precision="fp16" \
 --resolution=256 \
 --learning_rate=1e-5 \
 --max_train_steps=10000 \
 --validation_steps=100 \
 --max_train_samples=1000 \
 --train_batch_size=30 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --seed=42 \
 --image_column="file_name" \
 --caption_column="caption"


 accelerate launch train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=/data/jingshirou/diffusers/examples/controlnet/densepose_1024 \
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --controlnet_model_name_or_path=/data/jingshirou/diffusers/examples/controlnet/sdxl_weight_densepose_init/checkpoint-6000/controlnet \
 --max_train_steps=3000 \
 --checkpointing_steps=50 \
 --validation_steps=100 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --seed=42 \
 --image_column="file_name" \
 --caption_column="caption"


accelerate launch train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=/data/jingshirou/diffusers/examples/controlnet/densepose_1024 \
 --mixed_precision="fp16" \
 --resolution=256 \
 --learning_rate=1e-5 \
 --max_train_steps=9000 \
 --validation_steps=100 \
 --train_batch_size=1 \
 --checkpointing_steps=100 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --seed=42 \
 --image_column="file_name" \
 --caption_column="caption"


 accelerate launch train_controlnet_sdxlvae.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=/data/jingshirou/diffusers/examples/controlnet/densepose_1024 \
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --max_train_steps=9000 \
 --validation_steps=100 \
 --train_batch_size=2 \
 --checkpointing_steps=100 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --seed=42 \
 --image_column="file_name" \
 --caption_column="caption"


accelerate launch train_ctrlora_sdxlvae.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=/data/jingshirou/diffusers/examples/controlnet/densepose_1024 \
 --controlnet_model_name_or_path=/data/jingshirou/diffusers/examples/controlnet/sdxl_weight_densepose_VAE_right/checkpoint-1500/controlnet \
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --max_train_steps=9000 \
 --validation_steps=100 \
 --train_batch_size=2 \
 --checkpointing_steps=100 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --seed=42 \
 --image_column="file_name" \
 --caption_column="caption"





 accelerate launch train_ctrlora_sdxlvae.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=/data/jingshirou/diffusers/examples/controlnet/densepose_1024 \
 --mixed_precision="fp16" \
 --resolution=256 \
 --learning_rate=1e-5 \
 --max_train_steps=9000 \
 --validation_steps=100 \
 --train_batch_size=1 \
 --checkpointing_steps=100 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --seed=42 \
 --image_column="file_name" \
 --caption_column="caption"



 /data/jingshirou/diffusers/examples/controlnet/densepose_1024



 --validation_image "./dense_conditioning_image_1.png" "./dense_conditioning_image_2.png"\
 --validation_prompt "a picture of a person riding a bike" "a picture of a man playing golf"\

train_controlnet_webdatasets.py 
--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-0.9 
--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix 
--output_dir=controlnet-0-9-canny 
--mixed_precision=fp16 
--resolution=1024 
--learning_rate=1e-5 
--max_train_steps=30000 
--max_train_samples=3000000 
--dataloader_num_workers=4 
--validation_image ./c_image_0.png ./c_image_1.png ./c_image_2.png ./c_image_3.png ./c_image_4.png ./c_image_5.png ./c_image_6.png ./c_image_7.png 
--validation_prompt "two birds" "a snowy mountain" "a lake with clouds" "a woman using her phone" "a couple getting married" "a wedding" "a house at a lake" "a boat in nature" 
--train_shards_path_or_url "pipe:aws s3 cp s3://muse-datasets/laion-aesthetic6plus-data/{00000..01208}.tar -" 
--eval_shards_path_or_url "pipe:aws s3 cp s3://muse-datasets/laion-aesthetic6plus-data/{01209..01210}.tar -" 
--proportion_empty_prompts 0.5 
--validation_steps=1000 
--train_batch_size=12 
--gradient_checkpointing 
--use_8bit_adam 
--enable_xformers_memory_efficient_attention 
--gradient_accumulation_steps=1 
--report_to=wandb 
--seed=42 
--push_to_hub

To better track our training experiments, we're using the following flags in the command above:

* `report_to="wandb` will ensure the training runs are tracked on Weights and Biases. To use it, be sure to install `wandb` with `pip install wandb`.
* `validation_image`, `validation_prompt`, and `validation_steps` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected.

Our experiments were conducted on a single 40GB A100 GPU.

### Inference

Once training is done, we can perform inference like so:

```python
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_path = "path to controlnet"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

control_image = load_image("./conditioning_image_1.png").resize((1024, 1024))
prompt = "pale golden rod circle with old lace background"

# generate image
generator = torch.manual_seed(0)
image = pipe(
    prompt, num_inference_steps=20, generator=generator, image=control_image
).images[0]
image.save("./output.png")
```

## Notes

### Specifying a better VAE

SDXL's VAE is known to suffer from numerical instability issues. This is why we also expose a CLI argument namely `--pretrained_vae_model_name_or_path` that lets you specify the location of an alternative VAE (such as [`madebyollin/sdxl-vae-fp16-fix`](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)).

If you're using this VAE during training, you need to ensure you're using it during inference too. You do so by:

```diff
+ vae = AutoencoderKL.from_pretrained(vae_path_or_repo_id, torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16,
+   vae=vae,
)

```
## train dataset
```python
train_dataset = get_train_dataset(args, accelerator) # download/load the dataset and process column names
compute_embeddings_fn = functools.partial(
        compute_embeddings,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        proportion_empty_prompts=args.proportion_empty_prompts,
    )
train_dataset = prepare_train_dataset(train_dataset, accelerator) # use transforms to process the dataset
train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    ) # collate_fn returns 
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt_ids": prompt_ids,
        "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
    }
controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

```

In all steps of the training, learning rate fixed at 1e-5

logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)