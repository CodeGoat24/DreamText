<div align="center">
<h1>DreamText: High Fidelity Scene Text Synthesis</h1>


[Yibin Wang](https://codegoat24.github.io), [Weizhong Zhang](https://weizhonz.github.io/)&#8224;, [Honghui Xu](https://scholar.google.com.hk/citations?user=_cZgJawAAAAJ&hl=zh-CN), [Cheng Jin](https://cjinfdu.github.io/)&#8224; 

(&#8224;corresponding author)

[Fudan University]

[Shanghai Innovation Intuition]

CVPR2025

<a href="https://arxiv.org/pdf/2405.14701">
<img src='https://img.shields.io/badge/arxiv-DreamText-blue' alt='Paper PDF'></a>
<a href="https://codegoat24.github.io/DreamText/">
<img src='https://img.shields.io/badge/Project-Website-orange' alt='Project Page'></a>
</div>

## 📖 Abstract
Scene text synthesis involves rendering specified texts onto arbitrary images. Current methods typically formulate this task in an end-to-end manner but lack effective character-level guidance during training. Besides, their text encoders, pre-trained on a single font type, struggle to adapt to the diverse font styles encountered in practical applications. Consequently, these methods suffer from character distortion, repetition, and absence, particularly in polystylistic scenarios. To this end, this paper proposes DreamText for high-fidelity scene text synthesis. Our key idea is to reconstruct the diffusion training process, introducing more refined guidance tailored to this task, to expose and rectify the model's attention at the character level and strengthen its learning of text regions. This transformation poses a hybrid optimization challenge, involving both discrete and continuous variables. To effectively tackle this challenge, we employ a heuristic alternate optimization strategy. Meanwhile, we jointly train the text encoder and generator to comprehensively learn and utilize the diverse font present in the training dataset. This joint training is seamlessly integrated into the alternate optimization process, fostering a synergistic relationship between learning character embedding and re-estimating character attention. Specifically, in each step, we first encode potential character-generated position information from cross-attention maps into latent character masks. These masks are then utilized to update the representation of specific characters in the current step, which, in turn, enables the generator to correct the character's attention in the subsequent steps. Both qualitative and quantitative results demonstrate the superiority of our method to the state of the art.

![DreamText Teaser](demo/teaser.png)

## 🔧 Usage

### Environment Setup

```bash
conda create -n dreamtext python=3.11
conda activate dreamtext
pip install -r requirements.txt
```

### Download our Pre-trained Models
Download our available [checkpoints](https://drive.google.com/file/d/1Q4B0oAnksORsPJS5TwoJU5uPRSFEbwS5/view?usp=sharing) and put them in the corresponding directories in `./checkpoints`.


## 🚀 Gradio Demo
You can run the demo locally by
```
python run_gradio.py
```
<img src=demo/gradio.png style="zoom:30%" />


## 🎨 Preparing Datasets


### LAION-OCR
- Create a data directory `{your data root}/LAION-OCR` in your disk and put your data in it. Then set the **data_root** field in `./configs/dataset/locr.yaml`.
- For the downloading and preprocessing of Laion-OCR dataset, please refer to [TextDiffuser](https://github.com/microsoft/unilm/tree/master/textdiffuser) and `./scripts/preprocess/laion_ocr_pre.ipynb`.

### ICDAR13
- Create a data directory `{your data root}/ICDAR13` in your disk and put your data in it. Then set the **data_root** field in `./configs/dataset/icd13.yaml`.
- Build the tree structure as below:
```
ICDAR13
├── train                  // training set
    ├── annos              // annotations
        ├── gt_x.txt
        ├── ...
    └── images             // images
        ├── img_x.jpg
        ├── ...
└── val                    // validation set
    ├── annos              // annotations
        ├── gt_img_x.txt
        ├── ...
    └── images             // images
        ├── img_x.jpg
        ├── ...
```

### TextSeg
- Create a data directory `{your data root}/TextSeg` in your disk and put your data in it. Then set the **data_root** field in `./configs/dataset/tsg.yaml`.
- Build the tree structure as below:
```
TextSeg
├── train                  // training set
    ├── annotation         // annotations
        ├── x_anno.json    // annotation json file
        ├── x_mask.png     // character-level mask
        ├── ...
    └── image              // images
        ├── x.jpg.jpg
        ├── ...
└── val                    // validation set
    ├── annotation         // annotations
        ├── x_anno.json    // annotation json file
        ├── x_mask.png     // character-level mask
        ├── ...
    └── image              // images
        ├── x.jpg
        ├── ...
```

### SynthText
- Create a data directory `{your data root}/SynthText` in your disk and put your data in it. Then set the **data_root** field in `./configs/dataset/st.yaml`.
- Build the tree structure as below:
```
SynthText
├── 1                      // part 1
    ├── ant+hill_1_0.jpg   // image
    ├── ant+hill_1_1.jpg
    ├── ...
├── 2                      // part 2
├── ...
└── gt.mat                 // annotation file
```



## 💻 Training
Download the [stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/blob/main/512-inpainting-ema.ckpt) and put it in `./checkpoints/pretrained/`.

Set the parameters in `./configs/train.yaml` and run:

```
python train.py
```

## ✨ Evaluation
Set the parameters in `./configs/test.yaml` and run:

```
python test.py
```



## 🎫 License
For non-commercial academic use, this project is licensed under [the 2-clause BSD License](https://opensource.org/license/bsd-2-clause). 
For commercial use, please contact [Cheng Jin](jc@fudan.edu.cn).


## ⭐ BibTeX
If you find our work helpful, please leave us a star and cite our paper.

```bibtex
@inproceedings{DreamText,
      title={High Fidelity Scene Text Synthesis},
      author={Wang, Yibin and Zhang, Weizhong and Honghui, Xu and Jin, Cheng},
      booktitle={CVPR},
      year={2025}
    }
```


## 📧 Contact

If you have any technical comments or questions, please open a new issue or feel free to contact [Yibin Wang](https://codegoat24.github.io).


## 🙏 Acknowledgements

Our work is based on [Stable-Diffusion](https://github.com/Stability-AI/stablediffusion), thanks to all the contributors!
