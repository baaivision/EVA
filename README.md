<div align="center">
<h1>EVA: An Open Billion-Scale Vision Foundation Model </h1>
<h3><a href="https://arxiv.org/abs/2211.07636">EVA: Exploring the Limits of Masked Visual Representation Learning at Scale</a></h3>

[Yuxin Fang](https://bit.ly/YuxinFang_GoogleScholar)<sup>2,1</sup>, [Wen Wang](https://scholar.google.com/citations?user=1ks0R04AAAAJ&hl)<sup>3,1</sup>, [Binhui Xie](https://binhuixie.github.io/)<sup>4,1</sup>, [Quan Sun](https://github.com/Quan-Sun)<sup>1</sup>, [Ledell Wu](https://scholar.google.com/citations?user=-eJHVt8AAAAJ&hl=en)<sup>1</sup>, [Xinggang Wang](https://xinggangw.info/)<sup>2</sup>, [Tiejun Huang](https://scholar.google.com/citations?user=knvEK4AAAAAJ&hl=en)<sup>1</sup>, [Xinlong Wang](https://www.xloong.wang/)<sup>1</sup>, [Yue Cao](http://yue-cao.me/)<sup>1</sup>
 
<sup>1</sup>[BAAI](https://www.baai.ac.cn/english.html), <sup>2</sup>[HUST](http://english.hust.edu.cn/), <sup>3</sup>[ZJU](https://www.zju.edu.cn/english/), <sup>4</sup>[BIT](https://english.bit.edu.cn/)

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.2211.07636-B31B1B.svg)](https://arxiv.org/abs/2211.07636) -->
<!-- ArXiv Preprint ([arXiv 2211.07636](https://arxiv.org/abs/2211.07636)) -->

\
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/self-supervised-image-classification-with-2)](https://paperswithcode.com/sota/self-supervised-image-classification-with-2?p=eva-exploring-the-limits-of-masked-visual) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/self-supervised-image-classification-with-3)](https://paperswithcode.com/sota/self-supervised-image-classification-with-3?p=eva-exploring-the-limits-of-masked-visual) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/self-supervised-image-classification-with)](https://paperswithcode.com/sota/self-supervised-image-classification-with?p=eva-exploring-the-limits-of-masked-visual) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/instance-segmentation-on-coco)](https://paperswithcode.com/sota/instance-segmentation-on-coco?p=eva-exploring-the-limits-of-masked-visual) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/instance-segmentation-on-coco-minival)](https://paperswithcode.com/sota/instance-segmentation-on-coco-minival?p=eva-exploring-the-limits-of-masked-visual) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/instance-segmentation-on-lvis-v1-0-val)](https://paperswithcode.com/sota/instance-segmentation-on-lvis-v1-0-val?p=eva-exploring-the-limits-of-masked-visual) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/object-detection-on-lvis-v1-0-val)](https://paperswithcode.com/sota/object-detection-on-lvis-v1-0-val?p=eva-exploring-the-limits-of-masked-visual) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/object-detection-on-coco-minival)](https://paperswithcode.com/sota/object-detection-on-coco-minival?p=eva-exploring-the-limits-of-masked-visual) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=eva-exploring-the-limits-of-masked-visual) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/semantic-segmentation-on-ade20k)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k?p=eva-exploring-the-limits-of-masked-visual) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/action-classification-on-kinetics-700)](https://paperswithcode.com/sota/action-classification-on-kinetics-700?p=eva-exploring-the-limits-of-masked-visual) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/action-classification-on-kinetics-400)](https://paperswithcode.com/sota/action-classification-on-kinetics-400?p=eva-exploring-the-limits-of-masked-visual) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/action-classification-on-kinetics-600)](https://paperswithcode.com/sota/action-classification-on-kinetics-600?p=eva-exploring-the-limits-of-masked-visual)
</div>


We launch **EVA**, a vision-centric foundation model to **E**xplore the limits of **V**isual representation at sc**A**le using only publicly accessible data and academic resources. **EVA** is a vanilla ViT pre-trained to reconstruct the masked out image-text aligned vision features (*i.e.*, CLIP features) conditioned on visible image patches. Via this pretext task, we can efficiently scale up EVA to one billion parameters, and sets new records on a broad range of representative vision downstream tasks.

***EVA is the first open-sourced billion-scale vision foundation model that achieves state-of-the-art performance on a broad range of downstream tasks.***

## News

<div align="center">

### $\color{red}{\text{All the code and dozens of state-of-the-art billion-scale models are open-sourced!}}$

</div>

- `Dec 12, 2022`: [EVA](https://github.com/rwightman/pytorch-image-models#dec-6-2022) and [EVA-L](https://github.com/rwightman/pytorch-image-models#dec-8-2022) model weights are added to the awesome [`timm`](https://github.com/rwightman/pytorch-image-models) library, thanks @[rwightman](https://github.com/rwightman)!
- `Dec 07, 2022`: launch [**EVA-L**](https://github.com/baaivision/EVA/tree/master/eva#eva-l-learning-better-mim-representations-from-eva-clip), the **best** ViT-L (304M) to date that can reach up to **89.2** top-1 acc on IN-1K ([weights & logs](https://github.com/baaivision/EVA/tree/master/eva#eva-l-learning-better-mim-representations-from-eva-clip)) by leveraging vision features from [EVA-CLIP](clip/README.md).
- `Nov 25, 2022`: release EVA-CLIP zero-shot [evaluation results](https://github.com/baaivision/EVA/tree/master/clip#eva-clip-zero-shot-evaluation-results) on 35 benchmarks.
- `Nov 22, 2022`: release code & model of [object detection and instance segmentation](det/README.md).
- `Nov 21, 2022`: release code & model of [video classification](video/README.md), [semantic segmentation](seg/README.md), [EVA-CLIP](clip/README.md).
- `Nov 20, 2022`: release code & model of [pre-training and image classification](eva/README.md).
- `Nov 18, 2022`: release wandb [log & statistics](https://wandb.ai/baaivision/eva-clip/reports/ViT-g-14--VmlldzoyOTkwMDYy) of 1.1B EVA-CLIP training.

<span id="eva_performance_summary"></span>

## Catalog

All EVA model checkpoints are now available at [🤗 Hugging Face Models](https://huggingface.co/BAAI/EVA/tree/main) and [BAAI ModelHub](https://model.baai.ac.cn/models) ([EVA](https://model.baai.ac.cn/model-detail/100081) & [EVA-CLIP](https://model.baai.ac.cn/model-detail/100080)). Try them out!

- [x] [Pre-training](eva)
- [x] [Image Classification](eva)
- [x] [Video Classification](video)
- [x] [Object Detection & Instance Segmentation](det)
- [x] [Semantic Segmentation](seg)
- [x] [CLIP](clip)

## Summary of EVA's performance

<div align="center">

**image & video classification**
<table border="1" width="100%">
	<tr align="center">
        <th colspan="2"> </th><th colspan="4">image classification</th><th colspan="3">video classification</th>
    </tr>
    <tr align="center">
        <th>model</th><th>#param.</th><th>IN-1K, e2e ft</th><th>IN-1K, linear</th><th>IN-1K, zero-shot</th><th>12 avg. zero-shot</th><th>K400</th><th>K600</th><th>K700</th>
    </tr>
    <tr align="center">
        <th>EVA or EVA-CLIP</th><th>1.0B</th><th><a href="https://github.com/baaivision/EVA/blob/master/logs/cls/ft_1k_cls_sz560_89p7.txt">89.7</a></th><th><a href="https://github.com/baaivision/EVA/blob/master/logs/cls/linear_eva_clip_vision_enc_1k_cls_sz336_86p5.txt">86.5</a></th><th><a href="https://wandb.ai/baaivision/eva-clip/reports/ViT-g-14--VmlldzoyOTkwMDYy">78.5</a></th><th>75.7</th><th>89.7</th><th>89.8</th><th>82.9</th>
    </tr>
</table>
<br>

**object detection & segmentation**
<table border="1" width="200%">
	<tr align="center">
        <th> </th><th> </th><th colspan="4">COCO det & ins seg</th><th colspan="2">LVIS det & ins seg</th><th colspan="2">sem seg</th>
    </tr>
    <tr align="center">
        <th>model</th><th>#param.</th><th>det (test)</th><th>det (val)</th><th>seg (test)</th><th>seg (val)</th><th>det</th><th>seg</th><th>COCO-Stuff</th><th>ADE20K</th>
    </tr>
    <tr align="center">
        <th>EVA</th><th>1.0B</th><th><a href="https://codalab.lisn.upsaclay.fr/competitions/7384#results">64.7</a></th><th>64.5</th><th><a href="https://codalab.lisn.upsaclay.fr/competitions/7383#results">55.5</th><th>55.0</th><th>62.2</th><th>55.0</th><th><a href="https://github.com/baaivision/EVA/blob/master/logs/sem_seg/ft_cocstuff164k_sem_seg_ss_53p4.txt">53.4</a></th><th><a href="https://github.com/baaivision/EVA/blob/master/logs/sem_seg/ft_ade20k_sem_seg_ms_62p3.txt">62.3</a></th>
    </tr>
</table>
<br>
</div>


## Citation
If you find our work helpful, please **star🌟** this repo and **cite📑** our paper. Thanks for your support!

```
@article{EVA,
  title={EVA: Exploring the Limits of Masked Visual Representation Learning at Scale},
  author={Fang, Yuxin and Wang, Wen and Xie, Binhui and Sun, Quan and Wu, Ledell and Wang, Xinggang and Huang, Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2211.07636},
  year={2022}
}
```

If you find our open-sourced code & models helpful to your research, please also consider **cite📑** this repo.

```
@misc{EVA_code_models,
  author={Fang, Yuxin and Wang, Wen and Xie, Binhui and Sun, Quan and Wu, Ledell and Wang, Xinggang and Huang, Tiejun and Wang, Xinlong and Cao, Yue},
  title={Code and Models of EVA: Exploring the Limits of Masked Visual Representation Learning at Scale},
  year={2022},
  howpublished = {\url{https://github.com/baaivision/EVA}}
}
```

## License

The content of this project itself is licensed under [LICENSE](LICENSE).

## Contact

- For help and issues associated with EVA, or reporting a bug, please open a [GitHub Issue](https://github.com/baaivision/EVA/issues/new). 
Let's build a better & stronger EVA together :)

- **We are hiring** at all levels at BAAI Vision Team, including full-time researchers, engineers and interns. 
If you are interested in working with us on **foundation model, self-supervised learning and multimodal learning**, please contact [Yue Cao](http://yue-cao.me/) (`caoyue@baai.ac.cn`) and [Xinlong Wang](https://www.xloong.wang/) (`wangxinlong@baai.ac.cn`).


## Misc
### &#8627; Stargazers, thank you for your support!
[![Stargazers repo roster for @baaivision/EVA](https://reporoster.com/stars/baaivision/EVA)](https://github.com/baaivision/EVA/stargazers)

### &#8627; Forkers, thank you for your support!
[![Forkers repo roster for @baaivision/EVA](https://reporoster.com/forks/baaivision/EVA)](https://github.com/baaivision/EVA/network/members)

### &#8627; Star History
<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=baaivision/EVA&type=Date)](https://star-history.com/#baaivision/EVA&Date)

</div>