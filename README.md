<div align="center">

## 4DNeX: Feed-Forward 4D Generative Modeling Made Easy

</div>

<div>
<div align="center">
    <a href='https://frozenburning.github.io/' target='_blank'>Zhaoxi Chen</a><sup>1*</sup>&emsp;
    <a href='http://tqtqliu.github.io/' target='_blank'>Tianqi Liu</a><sup>1*</sup>&emsp;
    <a href='https://zhuolong3.github.io/' target='_blank'>Long Zhuo</a><sup>2*</sup>&emsp;   
    <a href='https://jiawei-ren.github.io/' target='_blank'>Jiawei Ren</a><sup>1</sup>&emsp; <br>
    <a href='https://zeng-tao.github.io/' target='_blank'>Zeng Tao</a><sup>2</sup>&emsp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=rI0fhugAAAAJ' target='_blank'>He Zhu</a><sup>2</sup>&emsp;
    <a href='https://hongfz16.github.io/' target='_blank'>FangZhou Hong</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=lSDISOcAAAAJ&hl=zh-CN' target='_blank'>Liang Pan</a><sup>2†</sup>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a><sup>1†</sup>
</div>
<div>
<div align="center">
    <sup>1</sup>Nanyang Technological University&emsp;
    <sup>2</sup>Shanghai AI Laboratory
</div>
<div align="center">
<sup>*</sup>Equal Contribution&emsp;  <sup>†</sup>Corresponding Authors
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2508.13154" target='_blank'>
    <img src="http://img.shields.io/badge/arXiv-2508.13154-b31b1b?logo=arxiv&logoColor=b31b1b" alt="ArXiv">
  </a>
  <a href="https://4dnex.github.io/4DNeX.pdf" target='_blank'><img src="https://img.shields.io/badge/Paper-PDF-f5cac3?logo=adobeacrobatreader&logoColor=red"/></a>
  <a href="https://4dnex.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-Page-red?logo=googlechrome&logoColor=red">
  </a>
  <a href="https://huggingface.co/datasets/3DTopia/4DNeX-10M" target='_blank'>
    <img src="https://img.shields.io/badge/Dataset-Download-green?logo=googledrive&logoColor=white">
  </a>
  <a href="https://www.youtube.com/watch?v=jaXNU1-0zgk">
    <img src="https://img.shields.io/badge/YouTube-Video-blue?logo=youtube&logoColor=blue">
  </a>
  <a href="#">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=3DTopia.4DNeX" alt="Visitors">
  </a>
</p>


>**TL;DR**: <em>4DNeX is a feed-forward framework for generating 4D scene representations from a single image by fine-tuning a video diffusion model. It produces high-quality dynamic point clouds and enables downstream tasks such as novel-view video synthesis with strong generalizability.</em>


<video controls autoplay src="https://github.com/user-attachments/assets/7e158b4c-4da6-44f6-b9d0-fa3a427606e5"></video>

## 🌟 Abstract

We present **4DNeX**, the first feed-forward framework for generating 4D (i.e., dynamic 3D) scene representations from a single image. In contrast to existing methods that rely on computationally intensive optimization or require multi-frame video inputs, 4DNeX enables efficient, end-to-end image-to-4D generation by fine-tuning a pretrained video diffusion model. Specifically, **1)** To alleviate the scarcity of 4D data, we construct 4DNeX-10M, a large-scale dataset with high-quality 4D annotations generated using advanced reconstruction approaches. **2)** We introduce a unified 6D video representation that jointly models RGB and XYZ sequences, facilitating structured learning of both appearance and geometry. **3)** We propose a set of simple yet effective adaptation strategies to repurpose pretrained video diffusion models for the 4D generation task. 4DNeX produces high-quality dynamic point clouds that enable novel-view video synthesis. Extensive experiments demonstrate that 4DNeX achieves competitive performance compared to existing 4D generation approaches, offering a scalable and generalizable solution for single-image-based 4D scene generation.

## 🚧 TODO List
- [ ] Data Preprocessing Scripts
- [x] Training Scripts
- [x] Inference Scripts
- [x] Pointmap Registration Scripts
- [x] Visualization Scripts

## 🚀 Quick Start

### Environment Setup
We use anaconda or miniconda to manage the python environment:
```bash
conda create -n "4dnex" python=3.10 -y
conda activate 4dnex
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# git lfs and rerun
conda install -c conda-forge git-lfs
conda install -c conda-forge rerun-sdk
```

### Pretrained Model
Our model is developed on top of [Wan2.1 I2V 14B](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers), please download the pretrained model from Hugging Face and place it in the `pretrained` directory as following structure:
```
4DNeX/
└── pretrained/
    └── Wan2.1-I2V-14B-480P-Diffusers/
        ├── model_index.json
        ├── scheduler/
        ├── unet/
        ├── vae/
        ├── text_encoder/
        ├── tokenizer/
        └── ...
```
Then, you may download our pretrained LoRA weights from HuggingFace [here](https://huggingface.co/FrozenBurning/4DNex-Lora) and place it in the `./pretrained` directory:
```bash
cd pretrained
mkdir 4dnex-lora
cd 4dnex-lora
huggingface-cli download FrozenBurning/4DNex-Lora --local-dir .
cd ../..
export PRETRAINED_LORA_PATH=./pretrained/4dnex-lora
```

### Inference 
After setup the environment and pretrained model, you can run the following command to generate 4D scene representations from a single image, the output video and point map will be saved in the `OUTPUT_DIR` directory. Assuming we are going to save the results in the `./results` directory, we can run the following command:
```bash
export OUTPUT_DIR=./results
python inference.py --prompt ./example/prompt.txt --image ./example/image.txt --out $OUTPUT_DIR --sft_path ./pretrained/Wan2.1-I2V-14B-480P-Diffusers/transformer  --type i2vwbw-demb-samerope --mode xyzrgb --lora_path $PRETRAINED_LORA_PATH --lora_rank 64
```
We store the path to the image in the `./example/image.txt` file, and the prompt in the `./example/prompt.txt` file for inference. Feel free to modify the prompt and image path to generate your own 4D scene representations.

### Visualization
To visualize the generated 4D scene representations, you may first perform pointmap registration using the following command:
```bash
python pm_registration.py --pkl_dir $OUTPUT_DIR
```
Then, you may visualize the pointmap registration results using [Rerun](https://github.com/rerun-io/rerun) as follows:
```bash
python rerun_vis.py --rr_recording test_log.rrd --pkl_dir $OUTPUT_DIR
rerun test_log.rrd --web-viewer
```
![rerun_demo_github](https://github.com/user-attachments/assets/433b2df2-711f-4360-a2d2-144837ef3944)


## 🔥 Training

### Prepare Data
Please checkout our 10M 4D dataset from [here](https://huggingface.co/datasets/3DTopia/4DNeX-10M), and place it in the `./data` directory. 

TODO: add the data preparation script.

### Launch Training
To launch training, we assume all data are in the `./data/wan21` directory, and run the following command:
```bash
bash scripts/finetune.sh
```

### Convert Zero Checkpoint to FP32
After training, you may convert the zero checkpoint to fp32 checkpoint for inference. For example, the output will be saved in the `./training/4dnex/5000-out` directory as follows:
```bash
python scripts/zero_to_fp32.py ./training/4dnex/checkpoint-5000 ./training/4dnex/5000-out --safe_serialization
```

## 📚 Citation
If you find our work useful for your research, please consider citing our paper:

```
@article{chen20254dnex,
    title={4DNeX: Feed-Forward 4D Generative Modeling Made Easy},
    author={Chen, Zhaoxi and Liu, Tianqi and Zhuo, Long and Ren, Jiawei and Tao, Zeng and Zhu, He and Hong, Fangzhou and Pan, Liang and Liu, Ziwei},
    journal={arXiv preprint arXiv:2508.13154},
    year={2025}
}
```
