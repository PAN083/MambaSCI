# 🐍 MambaSCI: Efficient Mamba-UNet for Quad-Bayer Patterned Video Snapshot Compressive Imaging
[![arxiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](http://arxiv.org/abs/2410.14214)
![Language](https://img.shields.io/badge/language-python-red) 
![visitors](https://visitor-badge.laobi.icu/badge?page_id=PAN083/MambaSCI)

---
<p align="center">
    <img width=60% src="figs/demo.gif"/>
</p>

Try to [drag](https://imgsli.com/MzA1MjE3) by yourself to feel the "before/after slider albums" of obtaining the reconstructed first RGB video frame from the 2D compressed measurement.

---
**Abstract:** Color video snapshot compressive imaging (SCI) employs computational imaging techniques to capture multiple sequential video frames in a single Bayer-patterned measurement. With the increasing popularity of quad-Bayer pattern in mainstream smartphone cameras for capturing high-resolution videos, mobile photography has become more accessible to a wider audience. However, existing color video SCI reconstruction algorithms are designed based on the traditional Bayer pattern. When applied to videos captured by quad-Bayer cameras, these algorithms often result in color distortion and ineffective demosaicing, rendering them impractical for primary equipment. To address this challenge, we propose the MambaSCI method, which leverages the Mamba and UNet architectures for efficient reconstruction of quad-Bayer patterned color video SCI. To the best of our knowledge, our work presents the first algorithm for quad-Bayer patterned SCI reconstruction, and also the initial application of the Mamba model to this task. Specifically, we customize Residual-Mamba-Blocks, which residually connect the Spatial-Temporal Mamba (STMamba), Edge-Detail-Reconstruction (EDR) module, and Channel Attention (CA) module. Respectively, STMamba is used to model long-range spatial-temporal dependencies with linear complexity, EDR is for better edge-detail reconstruction, and CA is used to compensate for the missing channel information interaction in Mamba model. Experiments demonstrate that MambaSCI surpasses state-of-the-art methods with lower computational and memory costs. PyTorch style pseudo-code for the core modules is provided in the supplementary materials.

<p align="center">
    <img width=80% src="figs/sci.png"/>
</p>


### • Framework
<p align="center">
    <img width=90% src="/figs/pipeline.png"/>
</p>


### • Model efficiency and effectiveness
<p align="center">
    <img width=60% src="/figs/first.png"/>
</p>

## 📧 News
- **Sept. 26, 2024:** MambaSCI is submitted to NeurIPS 2024 🍁

## 🔗 Contents
- [x] [Installation](#environment-installation)
- [x] [Training](#training)
- [x] [Testing](#testing)
- [ ] [Results](#Results)


<h2 id="environment-installation">🔨 Environment Installation</h2>

**1️⃣**
Make conda environment

```shell
git clone https://github.com/PAN083/MambaSCI.git

cd MambaSCI

conda create -n MambaSCI

conda activate MambaSCI

pip install -r requirements.txt
```
**2️⃣**
Install Mamba

```shell
cd mamba

python setup.py install
```

**3️⃣**
Install casual-conv1d

```shell
cd causal-conv1d

python setup.py install
```

<h2 id="training">🏋️ Training</h2>

First download DAVIS 2017 dataset from [DAVIS website](https://davischallenge.org/), then modify data_root value in configs/_base_/davis.py file, make sure data_root link to your training dataset path.

Launch multi GPU training by the statement below:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/Mamba/mamba_dp.py --distributed=True
```
Launch single GPU training by the statement below.
```shell
python tools/train.py configs/Mamba/mamba_dp.py
```


<h2 id="testing">⚡ Testing</h2>

Download [checkpoint](https://drive.google.com/file/d/1PqgCah35nNRbZGX7ctPvNVRZideaGabe/view?usp=drive_link) from website and put it into ./checkpoints

Then run
```shell
python tools/test.py configs/Mamba/mamba_dp.py --weights= "way to checkpoints"
```

<h2 id="citation">🎓 Citation</h2>
If you find this repository helpful to your research, please consider citing the following:

```
@inproceedings{panmambasci,
  title={MambaSCI: Efficient Mamba-UNet for Quad-Bayer Patterned Video Snapshot Compressive Imaging},
  author={Pan, Zhenghao and Zeng, Haijin and Cao, Jiezhang and Chen, Yongyong and Zhang, Kai and Xu, Yong},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```


🤗 Acknowledgement

This code is based on [STFormer](https://github.com/ucaswangls/STFormer), [LightM-UNet](https://github.com/MrBlankness/LightM-UNet), [Vivim](https://github.com/scott-yjyang/Vivim), [MambaIR](https://github.com/csguoh/MambaIR). Thank them for their outstanding work. 

