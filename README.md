# AdaIN-style (PyTorch)

This repository contains a PyTorch implementation of **Adaptive Instance Normalization (AdaIN)** for arbitrary style transfer.

**Paper:** [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization (ICCV 2017)](https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf)  
**Original Source Code:** [GitHub - xunhuang1995/AdaIN-style](https://github.com/xunhuang1995/AdaIN-style)

---

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Pretrained Model
Download the pretrained model from Google Drive: [pretrain_weight](https://drive.google.com/file/d/1nCgYgbQh0bdi7SZq7-2CgKttkv3bbUoK/view?usp=sharing)

Place it in the weights/ directory or any folder you prefer.

## Usage (Inference)
Run the style transfer on a content and style image:

```bash
python test.py \
    -c /path/to/content.jpg \
    -s /path/to/style.jpg \
    -o /path/to/output.jpg \
    -a 1.0 \
    -g 0 \
    -m /path/to/model.pth
```

## Architecture
<p align='center'>
  <img src='architecture.png' width="600px">
</p>

## Example
<p float="left">
  <img src="/content/000000000138.jpg" width="200" />
  <img src="/style/Hieronymus_Bosch_107.jpg" width="200" />
  <img src="/results/out_1.jpg" width="200" />
</p>
# Citation
```bash
@inproceedings{huang2017adain,
  title={Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization},
  author={Huang, Xun and Belongie, Serge},
  booktitle={ICCV},
  year={2017}
}
```