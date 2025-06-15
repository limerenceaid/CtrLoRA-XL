# CtrLoRA-XL
official repository for CtrLoRA-XL

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

We provide our pretrained models [here](https://huggingface.co/xyfJASON/ctrlora/tree/main). Please put the **Base ControlNet** (`ctrlora_sd15_basecn700k.ckpt`) into `./ckpts/ctrlora-basecn` and the **LoRAs** into `./ckpts/ctrlora-loras`.
The naming convention of the LoRAs is `ctrlora_sd15_<basecn>_<condition>.ckpt` for base conditions and `ctrlora_sd15_<basecn>_<condition>_<images>_<steps>.ckpt` for novel conditions.

You also need to download the **SDXL-based Models** and put them into `./ckpts/sdxl`. Models used in our work:


- Stable Diffusion XL (`v1-5-pruned.ckpt`): [official](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main)
- [DreamShaper XL](https://civitai.com/models/112902/dreamshaper-xl)
- [Anything XL](https://civitai.com/models/9409/or-anything-xl)
- [Juggernaut XL](https://civitai.com/models/133005/juggernaut-xl)
- [Pixel Art Diffusion XL](https://civitai.com/models/277680/pixel-art-diffusion-xl)
- [PVC Style Model](https://civitai.com/models/338712/pvc-style-modelmovable-figure-model-xl)
- [RealVisXL V5.0](https://civitai.com/models/139562/realvisxl-v50)

