# EVA-CLIP Zero-shot Evaluation Results

We provide a thorough evaluation of EVA-CLIP on 35 popular zero-shot benchmarks (27 image classification benchmarks + 4 video classification benchmarks + 2x2 retrieval benchmarks). The evaluation testbed is heavily based on [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark). Thanks for their awesome work.


**Table of Contents**

- [EVA-CLIP Zero-shot Evaluation Results](#eva-clip-zero-shot-evaluation-results)
  - [Zero-shot Image Classification Evaluation](#zero-shot-image-classification-evaluation)
    - [Averaged performance on all the 27 benchmarks.](#averaged-performance-on-all-the-27-benchmarks)
    - [Detailed results](#detailed-results)
  - [Zero-shot Video Action Recognition Evaluation](#zero-shot-video-action-recognition-evaluation)
  - [Zero-shot Retrieval Evaluation](#zero-shot-retrieval-evaluation)

## Zero-shot Image Classification Evaluation

### Averaged performance on all the 27 benchmarks.



<div align="center">

| model | model size| precision | training data | samples seen |  avg. acc. |
|-------|:-----:|:-----:|:----:|:----:|:----:|
| OpenAI CLIP-L | 430M | `fp16`| WIT-400M | 12 | 69.18 |  
| Open CLIP-H | 1.0B | `pytorch amp bf16` | LAION-2B | 32B | 72.39 | 
| Open CLIP-g | 1.3B | `pytorch amp bf16` | LAION-2B | 12B | 70.74 | 
| EVA CLIP-g | 1.1B | `deepspeed fp16` | LAION-400M | 11B | 71.43 |
 
</div>

EVA-CLIP shows very promising sample-efficiency, and we believe sufficient data scaling can further boost the performance. 


### Detailed results


<div align="center">
<table style="text-align:center">
   <tr align="center">
      <td rowspan=1>Dataset (27 in total)</td>
      <td rowspan=1 >Model</td>
      <td rowspan=1>acc@1</td>
      <td rowspan=1>acc@5</td>
      <td rowspan=1>mean_per_class_recall</td>
   </tr>
   <tr align="center">
      <td rowspan=4>ImageNet-1K</td>
      <td>OpenAI CLIP-L</td>
      <td>75.55 </td>
      <td>94.57 </td>
      <td>75.55 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>77.96</td>
      <td>95.23</td>
      <td>77.93</td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>76.65 </td>
      <td>94.84 </td>
      <td>76.66 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>78.53 </td>
      <td>95.51 </td>
      <td>78.51 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>ImageNet-Adversarial</td>
      <td>OpenAI CLIP-L</td>
      <td>70.76 </td>
      <td>90.76 </td>
      <td>67.88 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>59.33 </td>
      <td>85.64 </td>
      <td>58.18 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>57.19 </td>
      <td>83.41 </td>
      <td>56.55 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>73.59 </td>
      <td>90.93 </td>
      <td>69.97 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>ImageNet-Rendition</td>
      <td>OpenAI CLIP-L</td>
      <td>87.83 </td>
      <td>97.11 </td>
      <td>86.44 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>89.33 </td>
      <td>97.36 </td>
      <td>88.1 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>88.69 </td>
      <td>96.96 </td>
      <td>87.51 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>92.5 </td>
      <td>98.24 </td>
      <td>91.19 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>ImageNet-Sketch</td>
      <td>OpenAI CLIP-L</td>
      <td>59.58 </td>
      <td>84.25 </td>
      <td>59.61 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>66.58 </td>
      <td>88.12 </td>
      <td>66.57 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>65.17 </td>
      <td>87.46 </td>
      <td>65.21 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>67.31 </td>
      <td>89.07 </td>
      <td>67.31 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>ImageNet-V2</td>
      <td>OpenAI CLIP-L</td>
      <td>69.86 </td>
      <td>90.91 </td>
      <td>69.85 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>70.87 </td>
      <td>91.67 </td>
      <td>70.92 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>69.56 </td>
      <td>90.86 </td>
      <td>69.61 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>71.52 </td>
      <td>92.11 </td>
      <td>71.56 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>ObjectNet</td>
      <td>OpenAI CLIP-L</td>
      <td>68.98 </td>
      <td>88.06 </td>
      <td>67.37 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>69.71 </td>
      <td>87.74 </td>
      <td>68.45 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>67.53 </td>
      <td>86.7 </td>
      <td>66.56 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>72.33 </td>
      <td>89.37 </td>
      <td>70.88 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>SUN397</td>
      <td>OpenAI CLIP-L</td>
      <td>67.57 </td>
      <td>93.69 </td>
      <td>68.3 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>75.2 </td>
      <td>96.08 </td>
      <td>75.15 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>75.41 </td>
      <td>96.17 </td>
      <td>75.28 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>74.15 </td>
      <td>95.52 </td>
      <td>73.27 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>VOC2007</td>
      <td>OpenAI CLIP-L</td>
      <td>78.27 </td>
      <td>96.88 </td>
      <td>86.45 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>77.68 </td>
      <td>94.22 </td>
      <td>84.97 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>81.07 </td>
      <td>96.57 </td>
      <td>85.75 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>83.23 </td>
      <td>96.94 </td>
      <td>88.7 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>Birdsnap</td>
      <td>OpenAI CLIP-L</td>
      <td>40.52 </td>
      <td>64.81 </td>
      <td>40.12 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>52.92 </td>
      <td>73.47 </td>
      <td>52.91 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>48.68 </td>
      <td>71.1 </td>
      <td>48.61 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>50 </td>
      <td>70.97 </td>
      <td>50.07 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>Caltech101</td>
      <td>OpenAI CLIP-L</td>
      <td>86.67 </td>
      <td>96.85 </td>
      <td>93.27 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>88.24 </td>
      <td>88.24 </td>
      <td>94.56 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>88.21 </td>
      <td>97.24 </td>
      <td>94.13 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>87.72 </td>
      <td>95.68 </td>
      <td>94.81 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>Stanford Cars</td>
      <td>OpenAI CLIP-L</td>
      <td>77.86 </td>
      <td>98.4 </td>
      <td>77.84 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>93.4 </td>
      <td>99.89 </td>
      <td>93.41 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>92.92 </td>
      <td>99.88 </td>
      <td>93.45 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>91.71 </td>
      <td>99.76 </td>
      <td>91.66 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>CIFAR10</td>
      <td>OpenAI CLIP-L</td>
      <td>95.6 </td>
      <td>99.63 </td>
      <td>95.6 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>97.45 </td>
      <td>99.93 </td>
      <td>97.45 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>97.05 </td>
      <td>99.93 </td>
      <td>97.06 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>98.31 </td>
      <td>99.96 </td>
      <td>98.29 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>CIFAR100</td>
      <td>OpenAI CLIP-L</td>
      <td>75.81 </td>
      <td>92.76 </td>
      <td>75.81 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>84.73 </td>
      <td>97.34 </td>
      <td>84.73 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>83.91 </td>
      <td>97.31 </td>
      <td>83.92 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>88.66 </td>
      <td>88.71 </td>
      <td>88.65 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>Country211</td>
      <td>OpenAI CLIP-L</td>
      <td>31.86 </td>
      <td>59.36 </td>
      <td>31.87 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>29.88 </td>
      <td>55.76 </td>
      <td>29.86 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>28.8 </td>
      <td>54.24 </td>
      <td>28.82 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>28.63 </td>
      <td>55.37 </td>
      <td>28.64 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>Describable Textures</td>
      <td>OpenAI CLIP-L</td>
      <td>55.43 </td>
      <td>84.15 </td>
      <td>55.48 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>67.82 </td>
      <td>92.45 </td>
      <td>67.82 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>68.03 </td>
      <td>92.39 </td>
      <td>68.09 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>61.33 </td>
      <td>87.5 </td>
      <td>61.38 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>EuroSAT</td>
      <td>OpenAI CLIP-L</td>
      <td>62.4 </td>
      <td>95.14 </td>
      <td>63.72 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>72.7 </td>
      <td>95.09 </td>
      <td>72.91 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>63.22 </td>
      <td>98.1 </td>
      <td>63.71 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>73.57 </td>
      <td>98.75 </td>
      <td>74.39 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>Facial Emotion Recognition 2013</td>
      <td>OpenAI CLIP-L</td>
      <td>49.89 </td>
      <td>97.23 </td>
      <td>49.33 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>52.01 </td>
      <td>96.55 </td>
      <td>50.68 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>47.16 </td>
      <td>94.6 </td>
      <td>48.44 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>52.17 </td>
      <td>94.79 </td>
      <td>48.57 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>FGVC Aircraft</td>
      <td>OpenAI CLIP-L</td>
      <td>31.44 </td>
      <td>78.04 </td>
      <td>31.48 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>42.75 </td>
      <td>83.74 </td>
      <td>42.65 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>37.71 </td>
      <td>79.9 </td>
      <td>37.61 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>32.37 </td>
      <td>73.75 </td>
      <td>32.29 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>Oxford Flowers 102</td>
      <td>OpenAI CLIP-L</td>
      <td>79.2 </td>
      <td>92.16 </td>
      <td>79.32 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>80.11 </td>
      <td>92.91 </td>
      <td>79.92 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>77.41 </td>
      <td>90.6 </td>
      <td>77.92 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>74.47 </td>
      <td>90.65 </td>
      <td>74.29 </td>
   </tr>
   <tr align="center">
      <td rowspan=5>Food101</td>
      <td>OpenAI CLIP-L</td>
      <td>93.05 </td>
      <td>99.3 </td>
      <td>93.06 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>92.74 </td>
      <td>99.22 </td>
      <td>92.73 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>91.55 </td>
      <td>99.07 </td>
      <td>91.55 </td>
   </tr>
   <tr align="center">
      <td>FLIP-L</td>
      <td> 89.3 </td>
      <td> - </td>
      <td> - </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>93.46 </td>
      <td>99.32 </td>
      <td>93.46 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>GTSRB</td>
      <td>OpenAI CLIP-L</td>
      <td>50.55 </td>
      <td>76.08 </td>
      <td>43.96 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>58.36 </td>
      <td>82.1 </td>
      <td>54.32 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>49.8 </td>
      <td>76.88 </td>
      <td>46.77 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>49.12 </td>
      <td>84.56 </td>
      <td>47.08 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>MNIST</td>
      <td>OpenAI CLIP-L</td>
      <td>76.35 </td>
      <td>93.53 </td>
      <td>75.88 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>72.86 </td>
      <td>94.21 </td>
      <td>73.65 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>68.57 </td>
      <td>95.15 </td>
      <td>68.97 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>62.34 </td>
      <td>90.81 </td>
      <td>62.35 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>Oxford-IIIT Pets</td>
      <td>OpenAI CLIP-L</td>
      <td>93.49 </td>
      <td>99.78 </td>
      <td>93.45 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>94.55 </td>
      <td>99.86 </td>
      <td>94.51 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>94.36 </td>
      <td>99.81 </td>
      <td>94.35 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>94.22 </td>
      <td>99.86 </td>
      <td>94.2 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>STL10</td>
      <td>OpenAI CLIP-L</td>
      <td>99.36 </td>
      <td>100 </td>
      <td>99.36 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>98.48 </td>
      <td>99.99 </td>
      <td>98.48 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>98.65 </td>
      <td>99.98 </td>
      <td>98.68 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>98.89 </td>
      <td>100 </td>
      <td>98.89 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>RESISC45</td>
      <td>OpenAI CLIP-L</td>
      <td>64.64 </td>
      <td>93.21 </td>
      <td>64.68 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>70.54 </td>
      <td>96.03 </td>
      <td>70.55 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>72.5 </td>
      <td>96.12 </td>
      <td>72.54 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>70.3 </td>
      <td>94.66 </td>
      <td>70.3 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>PatchCamelyon</td>
      <td>OpenAI CLIP-L</td>
      <td>51.98 </td>
      <td>- </td>
      <td>51.97 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>54.24 </td>
      <td>- </td>
      <td>54.22 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>56.11 </td>
      <td>- </td>
      <td>56.11 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>49.88 </td>
      <td>- </td>
      <td>49.86 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>Rendered SST2</td>
      <td>OpenAI CLIP-L</td>
      <td>68.86 </td>
      <td>- </td>
      <td>68.88 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>64.25 </td>
      <td>- </td>
      <td>64.27 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>64.14 </td>
      <td>- </td>
      <td>64.16 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>58.38 </td>
      <td>- </td>
      <td>58.41 </td>
   </tr>
</table>

</div>

## Zero-shot Video Action Recognition Evaluation


<div align="center">
<table style="text-align:center">
   <tr align="center">
      <td rowspan=1>Dataset</td>
      <td rowspan=1>Model</td>
      <td rowspan=1>acc@1</td>
      <td rowspan=1>acc@5</td>
      <td rowspan=1>mean(acc@1, acc@5)</td>
   </tr>
   <tr align="center">
      <td rowspan=4>UCF101</td>
      <td>OpenAI CLIP-L</td>
      <td>76.39 </td>
      <td>94.86 </td>
      <td>85.63 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>78.16</td>
      <td>95.02</td>
      <td>86.59 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>77.73 </td>
      <td>94.98 </td>
      <td>86.36 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>76.05 </td>
      <td>93.64 </td>
      <td>84.84 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>Kinetics400</td>
      <td>OpenAI CLIP-L</td>
      <td>52.88 </td>
      <td>76.06 </td>
      <td>64.47 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>51.63 </td>
      <td>74.49 </td>
      <td>63.06 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>50.35 </td>
      <td>73.03 </td>
      <td>61.69 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>54.04 </td>
      <td>76.42</td>
      <td>65.23 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>Kinetics600</td>
      <td>OpenAI CLIP-L</td>
      <td>52.41 </td>
      <td>76 </td>
      <td>64.21 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>52.25</td>
      <td>74.92</td>
      <td>63.58 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>50.79 </td>
      <td>73.53 </td>
      <td>62.16 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>52.76 </td>
      <td>75.99</td>
      <td>64.38 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>Kinetics700</td>
      <td>OpenAI CLIP-L</td>
      <td>45.73 </td>
      <td>69.63 </td>
      <td>57.68 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>44.64 </td>
      <td>67.54 </td>
      <td>56.09 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>43.6 </td>
      <td>66.39 </td>
      <td>54.99 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>46.65 </td>
      <td>70.16 </td>
      <td>58.4 </td>
   </tr>
</table>

</div>


## Zero-shot Retrieval Evaluation

<div align="center">

<table style="text-align:center">
   <tr align="center">
      <td rowspan=2>Dataset</td>
      <td rowspan=2>Model</td>
      <td colspan=3>Text-to-Image Retrival</td>
      <td colspan=3>Image-to-Text Retrival</td>
   </tr>
   <tr align="center">
      <td>R@1</td>
      <td>R@5</td>
      <td>R@10</td>
      <td>R@1</td>
      <td>R@5</td>
      <td>R@10</td>
   </tr>
   <tr align="center">
      <td rowspan=4>Flickr30k</td>
      <td>OpenAI CLIP-L</td>
      <td>65.18 </td>
      <td>87.28 </td>
      <td>92 </td>
      <td>85.2 </td>
      <td>97.3 </td>
      <td>99 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>77.78</td>
      <td>94.14</td>
      <td>96.62</td>
      <td>90.8</td>
      <td>99.3</td>
      <td>99.7</td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>76.52 </td>
      <td>93.62 </td>
      <td>96.28 </td>
      <td>90.8 </td>
      <td>99.1 </td>
      <td>99.8 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>72.64 </td>
      <td>91.6 </td>
      <td>95.12 </td>
      <td>88.3 </td>
      <td>98.3 </td>
      <td>99.3 </td>
   </tr>
   <tr align="center">
      <td rowspan=4>MSCOCO</td>
      <td>OpenAI CLIP-L</td>
      <td>36.51 </td>
      <td>61.01 </td>
      <td>71.11 </td>
      <td>56.34 </td>
      <td>79.32 </td>
      <td>86.66 </td>
   </tr>
   <tr align="center">
      <td>Open CLIP-H</td>
      <td>49.47</td>
      <td>73.4</td>
      <td>81.53</td>
      <td>65.96</td>
      <td>86.06</td>
      <td>91.9</td>
   </tr>
   <tr align="center">
      <td>Open CLIP-g</td>
      <td>47.99 </td>
      <td>72.37 </td>
      <td>80.75 </td>
      <td>64.96 </td>
      <td>85.3 </td>
      <td>91.46 </td>
   </tr>
   <tr align="center">
      <td><b>EVA CLIP-g</b></td>
      <td>44.07 </td>
      <td>68.5 </td>
      <td>77.33 </td>
      <td>61.76 </td>
      <td>83.28 </td>
      <td>89.96 </td>
   </tr>
</table>
</div>



The zero-shot retrieval performance of EVA-CLIP is relatively inferior to the Open CLIP-H / -g counterpart. We speculate there are two main reasons: 
- The size / capacity of the language tower in EVA-CLIP is much smaller / weaker than Open CLIP-H and Open CLIP-g, *i.e.*, `124M` *v.s.* `354M`. Meanwhile, retrieval tasks depend more on the capacity of the language branch compared with classification tasks.
- Retrieval tasks seem benefit more from the training dataset size (LAION-2B used by Open CLIP), while we only leverage LAION-400M for EVA-CLIP training. 

Nevertheless, it is hard to make a head-to-head comparison between different CLIP models. In the future, we will further scale up the language encoder & training data to improve the retrieval performance.