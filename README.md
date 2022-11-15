<div align="center">
<h1>EVA</h1>
<h3>Exploring the Limits of Masked Visual Representation Learning at Scale</h3>


[Yuxin Fang](https://bit.ly/YuxinFang_GoogleScholar)<sup>2,1</sup>, [Wen Wang](https://scholar.google.com/citations?user=1ks0R04AAAAJ&hl)<sup>3,1</sup>, [Binhui Xie](https://binhuixie.github.io/)<sup>4,1</sup>, [Quan Sun](https://github.com/Quan-Sun)<sup>1</sup>, [Ledell Wu](https://scholar.google.com/citations?user=-eJHVt8AAAAJ&hl=en)<sup>1</sup>, [Xinggang Wang](https://xinggangw.info/)<sup>2</sup>, [Tiejun Huang](https://scholar.google.com/citations?user=knvEK4AAAAAJ&hl=en)<sup>1</sup>, [Xinlong Wang](https://www.xloong.wang/)<sup>1</sup>, [Yue Cao](http://yue-cao.me/)<sup>1</sup>
 
<sup>1</sup>[BAAI](https://www.baai.ac.cn/english.html), <sup>2</sup>[HUST](http://english.hust.edu.cn/), <sup>3</sup>[ZJU](https://www.zju.edu.cn/english/), <sup>4</sup>[BIT](https://english.bit.edu.cn/)


ArXiv Preprint ([arXiv 2211.07636](https://arxiv.org/abs/2211.07636))

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/instance-segmentation-on-coco-minival)](https://paperswithcode.com/sota/instance-segmentation-on-coco-minival?p=eva-exploring-the-limits-of-masked-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/instance-segmentation-on-coco)](https://paperswithcode.com/sota/instance-segmentation-on-coco?p=eva-exploring-the-limits-of-masked-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/object-detection-on-lvis-v1-0-val)](https://paperswithcode.com/sota/object-detection-on-lvis-v1-0-val?p=eva-exploring-the-limits-of-masked-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/instance-segmentation-on-lvis-v1-0-val)](https://paperswithcode.com/sota/instance-segmentation-on-lvis-v1-0-val?p=eva-exploring-the-limits-of-masked-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/object-detection-on-coco-minival)](https://paperswithcode.com/sota/object-detection-on-coco-minival?p=eva-exploring-the-limits-of-masked-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=eva-exploring-the-limits-of-masked-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eva-exploring-the-limits-of-masked-visual/semantic-segmentation-on-ade20k)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k?p=eva-exploring-the-limits-of-masked-visual)

</div>


We launch **EVA**, a vision-centric foundation model to **E**xplore the limits of **V**isual representation at sc**A**le using only publicly accessible data and academic resources. **EVA** is a vanilla ViT pre-trained to reconstruct the masked out image-text aligned vision features (*i.e.*, CLIP features) conditioned on visible image patches. Via this pretext task, we can efficiently scale up EVA to one billion parameters, and sets new records on a broad range of representative vision downstream tasks.

Code and model weights will be released here.


<span id="eva_performance_summary"></span>
## Summary of EVA's performance

**image & video classification**
<table border="1" width="100%">
	<tr align="center">
        <th> </th><th> </th><th colspan="3">image classification</th><th colspan="3">video classification</th>
    </tr>
    <tr align="center">
        <th>model</th><th>#param.</th><th>IN-1K</th><th>IN-1K, zero-shot</th><th>12 avg. zero-shot</th><th>K400</th><th>K600</th><th>K700</th>
    </tr>
    <tr align="center">
        <th>EVA</th><th>1.0B</th><th><a href="https://github.com/baaivision/EVA/blob/master/logs/cls/ft_1k_cls_sz560_89p7.txt">89.7</a></th><th>78.2</th><th>72.5</th><th>89.7</th><th>89.8</th><th>82.9</th>
    </tr>
</table>
<br>

**object detection & segmentation**
<table border="1" width="200%">
	<tr align="center">
        <th> </th><th> </th><th colspan="4">COCO object detection & instance segmentation</th><th colspan="2">LVIS object detection & instance segmentation</th><th colspan="2">semantic segmentation</th>
    </tr>
    <tr align="center">
        <th>model</th><th>#param.</th><th>COCO det (test)</th><th>COCO det (val)</th><th>COCO ins seg (test)</th><th>COCO ins seg (val)</th><th>LVIS det</th><th>LVIS ins seg</th><th>COCO-Stuff</th><th>ADE20K</th>
    </tr>
    <tr align="center">
        <th>EVA</th><th>1.0B</th><th><a href="https://codalab.lisn.upsaclay.fr/competitions/7384#results">64.7</a></th><th>64.5</th><th><a href="https://codalab.lisn.upsaclay.fr/competitions/7383#results">55.5</th><th>55.0</th><th>62.2</th><th>55.0</th><th><a href="https://github.com/baaivision/EVA/blob/master/logs/sem_seg/ft_cocstuff164k_sem_seg_ss_53p4.txt">53.4</a></th><th><a href="https://github.com/baaivision/EVA/blob/master/logs/sem_seg/ft_ade20k_sem_seg_ms_62p3.txt">62.3</a></th>
    </tr>
</table>
<br>



## Citation
If you find our work helpful, please star this repo and cite the related articles. Thanks for your support!

```
@article{EVA,
  title={EVA: Exploring the Limits of Masked Visual Representation Learning at Scale},
  author={Fang, Yuxin and Wang, Wen and Xie, Binhui and Sun, Quan and Wu, Ledell and Wang, Xinggang and Huang, Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2211.07636},
  year={2022}
}
```

## We are Hiring

We are hiring at all levels at BAAI Vision Team, including full-time researchers, engineers and interns. 

If you are interested in working with us on **foundation model, self-supervised learning and multimodal learning**, please contact [Yue Cao](http://yue-cao.me/) (`caoyue@baai.ac.cn`) and [Xinlong Wang](https://www.xloong.wang/) (`wangxinlong@baai.ac.cn`).