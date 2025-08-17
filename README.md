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
  <a href="" target='_blank'>
    <img src="http://img.shields.io/badge/arXiv-25xx.xxxxx-b31b1b?logo=arxiv&logoColor=b31b1b" alt="ArXiv">
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


## 📚 Citation
If you find our work useful for your research, please consider citing our paper:

```

```