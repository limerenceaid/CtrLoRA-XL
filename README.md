# CtrLoRA-XL
official implementation for CtrLoRA-XL

### Installation

First, create a new environment

```bash
conda create -n ctrlora_sdxl python=3.10
```
Then install the environment follows:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then run
```bash
pip install -r requirements_sdxl.txt
```


## 🤖️ Download Pretrained Models

We provide our pretrained models [here](https://huggingface.co/Savlim/CtrLoRA-XL/tree/main). Please put the **Base ControlNet** (`ctrlora-xl-base.safetensors`) into `./ckpts/ctrlora-xl-base` and the **LoRAs** into `./ckpts/ctrlora-xl-loras`.


You also need to download the **SDXL-based Models** and put them into `./ckpts/sdxl`. Models used in our work:


- Stable Diffusion XL (`v1-5-pruned.ckpt`): [official](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main)
- [DreamShaper XL](https://civitai.com/models/112902/dreamshaper-xl)
- [Anything XL](https://civitai.com/models/9409/or-anything-xl)
- [Juggernaut XL](https://civitai.com/models/133005/juggernaut-xl)
- [Pixel Art Diffusion XL](https://civitai.com/models/277680/pixel-art-diffusion-xl)
- [PVC Style Model](https://civitai.com/models/338712/pvc-style-modelmovable-figure-model-xl)
- [RealVisXL V5.0](https://civitai.com/models/139562/realvisxl-v50)

## 


## Inference


```bash
python scripts/inference.py   --sd_path=SDXL_PATH \
 --cn_path=CONTROLNET_PATH \
 --lora_path=LORA_PATH \
 --task=[single, multi] \
 --data_root=DATAROOT \
 --test_num=TEST_NUM \
 --condition=CONDITION \
 --condition2=CONDITION2 \
 --seed=SEED \
 --guidance_scale=GUIDANCE_SCALE
```

## Train your own lora

```bash
bash train_lora_script.sh
```