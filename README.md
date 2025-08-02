Below is a complete, polished, and integrated `README.md` file for your repository, based on the provided content. It’s structured to be clear, professional, and concise, while incorporating all key elements such as the paper citation, attention mechanism categories, plug-and-play modules, dynamic networks, vision transformers, and a section for contributions. I've also added a placeholder for your personal contribution and ensured the content is streamlined to avoid redundancy. The file is formatted in Markdown, ready to be added to your repository.


# Awesome Attention Mechanisms in Computer Vision


[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repository is a comprehensive resource for attention mechanisms in computer vision, inspired by the paper *[Attention Mechanisms in Computer Vision: A Survey](https://arxiv.org/abs/2111.07624)* by Meng-Hao Guo et al. It includes a curated list of papers, code implementations, and plug-and-play modules, covering channel, spatial, temporal, branch, and hybrid attention mechanisms, as well as vision transformers. Contributions are welcome!

🔥 *Papers with citations > 200 are marked with a fire emoji.*

---

## Table of Contents
- [Introduction](#introduction)
- [Paper Citation](#paper-citation)
- [Attention Mechanisms](#attention-mechanisms)
  - [Channel Attention](#channel-attention)
  - [Spatial Attention](#spatial-attention)
  - [Temporal Attention](#temporal-attention)
  - [Branch Attention](#branch-attention)
  - [Channel & Spatial Attention](#channel--spatial-attention)
  - [Spatial & Temporal Attention](#spatial--temporal-attention)
- [Plug-and-Play Modules](#plug-and-play-modules)
- [Dynamic Networks](#dynamic-networks)
- [Vision Transformers](#vision-transformers)
- [Contributing](#contributing)

---

## Introduction

This repository compiles resources on attention mechanisms in computer vision, including foundational papers, state-of-the-art models, and practical implementations. It builds on the survey paper *[Attention Mechanisms in Computer Vision: A Survey](https://arxiv.org/abs/2111.07624)*, published in *Computational Visual Media* (2022). The repository includes code implementations based on [Jittor](https://github.com/Jittor/jittor) and a curated list of plug-and-play attention modules and vision transformers. For a detailed overview in Chinese, refer to this [blog post](https://mp.weixin.qq.com/s/0iOZ45NTK9qSWJQlcI3_kQ).

We aim to keep this repository updated with the latest advancements. Contributions, including new papers, code, or corrections, are highly encouraged!

![Attention Mechanism Overview](https://github.com/MenghaoGuo/Awesome-Vision-Attentions/blob/main/imgs/fuse.png)

---

## Paper Citation

If you find this repository helpful, please cite the survey paper:

```bibtex
@article{guo2022attention,
  title={Attention mechanisms in computer vision: A survey},
  author={Guo, Meng-Hao and Xu, Tian-Xing and Liu, Jiang-Jiang and Liu, Zheng-Ning and Jiang, Peng-Tao and Mu, Tai-Jiang and Zhang, Song-Hai and Martin, Ralph R and Cheng, Ming-Ming and Hu, Shi-Min},
  journal={Computational Visual Media},
  pages={1--38},
  year={2022},
  publisher={Springer}
}
```

---

## Attention Mechanisms

Below is a categorized list of influential papers on attention mechanisms in computer vision, organized by type. Each entry includes the paper title, publication details, PDF link, and associated code (if available). Highly cited papers (>200 citations) are marked with 🔥.

### Channel Attention
- **Squeeze-and-Excitation Networks** (CVPR 2018, PAMI 2019)  
  [PDF](https://arxiv.org/pdf/1709.01507), [PAMI](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8701503), [Code](https://github.com/hujie-frank/SENet) 🔥  
- **Image Super-Resolution Using Very Deep Residual Channel Attention Networks** (ECCV 2018)  
  [PDF](https://arxiv.org/pdf/1807.02758) 🔥  
- **Context Encoding for Semantic Segmentation** (CVPR 2018)  
  [PDF](https://arxiv.org/pdf/1803.08904) 🔥  
- **Spatio-Temporal Channel Correlation Networks for Action Classification** (ECCV 2018)  
  [PDF](https://arxiv.org/pdf/1806.07754)  
- **Global Second-Order Pooling Convolutional Networks** (CVPR 2019)  
  [PDF](https://arxiv.org/pdf/1811.12006), [Code](https://github.com/ZilinGao/Global-Second-order-Pooling-Convolutional-Networks)  
- **SRM: A Style-Based Recalibration Module for Convolutional Neural Networks** (ICCV 2019)  
  [PDF](https://arxiv.org/pdf/1903.10829), [Code](https://github.com/hyunjaelee410/style-based-recalibration-module)  
- **You Look Twice: GaterNet for Dynamic Filter Selection in CNNs** (CVPR 2019)  
  [PDF](https://arxiv.org/pdf/1811.11205)  
- **Second-Order Attention Network for Single Image Super-Resolution** (CVPR 2019)  
  [PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.pdf) 🔥  
- **DIANet: Dense-and-Implicit Attention Network** (AAAI 2020)  
  [PDF](https://arxiv.org/pdf/1905.10671.pdf), [Code](https://github.com/gbup-group/DIANet)  
- **SpSequenceNet: Semantic Segmentation Network on 4D Point Clouds** (CVPR 2020)  
  [PDF](https://openaccess.thecvf.com/content_CVPR_2020/html/Shi_SpSequenceNet_Semantic_Segmentation_Network_on_4D_Point_Clouds_CVPR_2020_paper.html)  
- **ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks** (CVPR 2020)  
  [PDF](https://arxiv.org/pdf/1910.03151), [Code](https://github.com/BangguWu/ECANet) 🔥  
- **Gated Channel Transformation for Visual Recognition** (CVPR 2020)  
  [PDF](https://arxiv.org/pdf/1909.11519)  
- **FcaNet: Frequency Channel Attention Networks** (ICCV 2021)  
  [PDF](https://arxiv.org/pdf/2012.11879), [Code](https://github.com/cfzd/FcaNet)

### Spatial Attention
- **Recurrent Models of Visual Attention** (NeurIPS 2014)  
  [PDF](https://arxiv.org/pdf/1406.6247) 🔥  
- **Show, Attend and Tell: Neural Image Caption Generation with Visual Attention** (PMLR 2015)  
  [PDF](https://arxiv.org/pdf/1502.03044) 🔥  
- **DRAW: A Recurrent Neural Network for Image Generation** (ICML 2015)  
  [PDF](https://arxiv.org/pdf/1502.04623) 🔥  
- **Spatial Transformer Networks** (NeurIPS 2015)  
  [PDF](https://arxiv.org/pdf/1506.02025) 🔥  
- **Multiple Object Recognition with Visual Attention** (ICLR 2015)  
  [PDF](https://arxiv.org/pdf/1412.7755) 🔥  
- **Action Recognition Using Visual Attention** (arXiv 2015)  
  [PDF](https://arxiv.org/pdf/1511.04119) 🔥  
- **VideoLSTM Convolves, Attends and Flows for Action Recognition** (arXiv 2016)  
  [PDF](https://arxiv.org/pdf/1607.01794) 🔥  
- **Look Closer to See Better: Recurrent Attention Convolutional Neural Network** (CVPR 2017)  
  [PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Look_Closer_to_CVPR_2017_paper.pdf), [Code](https://github.com/Jianlong-Fu/Recurrent-Attention-CNN) 🔥  
- **Learning Multi-Attention Convolutional Neural Network for Fine-Grained Image Recognition** (ICCV 2017)  
  [PDF](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zheng_Learning_Multi-Attention_Convolutional_ICCV_2017_paper.pdf) 🔥  
- **Diversified Visual Attention Networks for Fine-Grained Object Classification** (TMM 2017)  
  [PDF](https://arxiv.org/pdf/1606.08572) 🔥  
- **Non-Local Neural Networks** (CVPR 2018)  
  [PDF](https://arxiv.org/pdf/1711.07971), [Code](https://github.com/AlexHex7/Non-local_pytorch) 🔥  
- **Relation Networks for Object Detection** (CVPR 2018)  
  [PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Relation_Networks_for_CVPR_2018_paper.pdf) 🔥  
- **A2-Nets: Double Attention Networks** (NeurIPS 2018)  
  [PDF](https://arxiv.org/pdf/1810.11579), [Code](https://github.com/nguyenvo09/Double-Attention-Network) 🔥  
- **Attention-Aware Compositional Network for Person Re-Identification** (CVPR 2018)  
  [PDF](https://arxiv.org/pdf/1805.03344) 🔥  
- **Tell Me Where to Look: Guided Attention Inference Network** (CVPR 2018)  
  [PDF](https://arxiv.org/pdf/1802.10171) 🔥  
- **PSANet: Point-Wise Spatial Attention Network for Scene Parsing** (ECCV 2018)  
  [PDF](https://openaccess.thecvf.com/content_ECCV_2018/html/Hengshuang_Zhao_PSANet_Point-wise_Spatial_ECCV_2018_paper.html) 🔥  
- **Self-Attention Generative Adversarial Networks** (ICML 2019)  
  [PDF](https://arxiv.org/pdf/1805.08318) 🔥  
- **Attention Augmented Convolutional Networks** (ICCV 2019)  
  [PDF](https://arxiv.org/pdf/1904.09925), [Code](https://github.com/leaderj1001/Attention-Augmented-Conv2d) 🔥  
- **GCNet: Non-Local Networks Meet Squeeze-Excitation Networks and Beyond** (ICCVW 2019)  
  [PDF](https://arxiv.org/pdf/1904.11492), [Code](https://github.com/xvjiarui/GCNet) 🔥  
- **Asymmetric Non-Local Neural Networks for Semantic Segmentation** (ICCV 2019)  
  [PDF](https://arxiv.org/pdf/1908.07678), [Code](https://github.com/MendelXu/ANN) 🔥  
- **End-to-End Object Detection with Transformers** (ECCV 2020)  
  [PDF](https://arxiv.org/pdf/2005.12872) 🔥  
- **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** (ICLR 2021)  
  [PDF](https://arxiv.org/pdf/2010.11929), [Code](https://github.com/lucidrains/vit-pytorch) 🔥  
- **OcNet: Object Context Network for Scene Parsing** (IJCV 2021)  
  [PDF](https://arxiv.org/pdf/1809.00916), [Code](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR) 🔥  

### Temporal Attention
- **Jointly Attentive Spatial-Temporal Pooling Networks for Video-Based Person Re-Identification** (ICCV 2017)  
  [PDF](https://arxiv.org/pdf/1708.02286.pdf) 🔥  
- **Video Person Re-Identification with Competitive Snippet-Similarity Aggregation** (CVPR 2018)  
  [PDF](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1036.pdf)  
- **SCAN: Self-and-Collaborative Attention Network for Video Person Re-Identification** (TIP 2019)  
  [PDF](https://arxiv.org/pdf/1807.05688.pdf)  

### Branch Attention
- **Training Very Deep Networks** (NeurIPS 2015)  
  [PDF](https://arxiv.org/pdf/1507.06228.pdf) 🔥  
- **Selective Kernel Networks** (CVPR 2019)  
  [PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Selective_Kernel_Networks_CVPR_2019_paper.pdf), [Code](https://github.com/implus/SKNet) 🔥  
- **CondConv: Conditionally Parameterized Convolutions for Efficient Inference** (NeurIPS 2019)  
  [PDF](https://arxiv.org/pdf/1904.04971.pdf), [Code](https://github.com/d-li14/condconv.pytorch)  
- **Dynamic Convolution: Attention over Convolution Kernels** (CVPR 2020)  
  [PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Dynamic_Convolution_Attention_Over_Convolution_Kernels_CVPR_2020_paper.pdf), [Code](https://github.com/kaijieshi7/Dynamic-convolution-Pytorch)  
- **ResNeSt: Split-Attention Networks** (arXiv 2020)  
  [PDF](https://arxiv.org/pdf/2004.08955.pdf), [Code](https://github.com/zhanghang1989/ResNeSt) 🔥  

### Channel & Spatial Attention
- **Residual Attention Network for Image Classification** (CVPR 2017)  
  [PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Residual_Attention_Network_CVPR_2017_paper.pdf) 🔥  
- **SCA-CNN: Spatial and Channel-Wise Attention in Convolutional Networks** (CVPR 2017)  
  [PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_SCA-CNN_Spatial_and_CVPR_2017_paper.pdf) 🔥  
- **CBAM: Convolutional Block Attention Module** (ECCV 2018)  
  [PDF](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf), [Code](https://github.com/Jongchan/attention-module) 🔥  
- **Harmonious Attention Network for Person Re-Identification** (CVPR 2018)  
  [PDF](https://arxiv.org/pdf/1802.08122.pdf) 🔥  
- **BAM: Bottleneck Attention Module** (BMVC 2018)  
  [PDF](http://bmvc2018.org/contents/papers/0092.pdf), [Code](https://github.com/Jongchan/attention-module) 🔥  
- **Dual Attention Network for Scene Segmentation** (CVPR 2019)  
  [PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fu_Dual_Attention_Network_for_Scene_Segmentation_CVPR_2019_paper.pdf), [Code](https://github.com/junfu1115/DANet) 🔥  
- **Coordinate Attention for Efficient Mobile Network Design** (CVPR 2021)  
  [PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Hou_Coordinate_Attention_for_Efficient_Mobile_Network_Design_CVPR_2021_paper.pdf), [Code](https://github.com/Andrew-Qibin/CoordAttention)  
- **SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks** (ICML 2021)  
  [PDF](http://proceedings.mlr.press/v139/yang21o/yang21o.pdf), [Code](https://github.com/ZjjConan/SimAM)  

### Spatial & Temporal Attention
- **An End-to-End Spatio-Temporal Attention Model for Human Action Recognition** (AAAI 2017)  
  [PDF](https://arxiv.org/pdf/1611.06067.pdf) 🔥  
- **Diversity Regularized Spatiotemporal Attention for Video-Based Person Re-Identification** (arXiv 2018)  
  [PDF](https://arxiv.org/abs/1803.11002) 🔥  
- **Interpretable Spatio-Temporal Attention for Video Action Recognition** (ICCVW 2019)  
  [PDF](https://openaccess.thecvf.com/content_ICCVW_2019/papers/HVU/Meng_Interpretable_Spatio-Temporal_Attention_for_Video_Action_Recognition_ICCVW_2019_paper.pdf)  
- **GTA: Global Temporal Attention for Video Action Understanding** (arXiv 2020)  
  [PDF](https://arxiv.org/pdf/2012.08510.pdf)  

---

## Plug-and-Play Modules

This section lists plug-and-play attention modules that can be easily integrated into convolutional neural networks.

| Title | Publish | Code |
|-------|---------|------|
| **ACNet: Strengthening the Kernel Skeletons for Powerful CNN** | ICCV 2019 | [ACNet](https://github.com/DingXiaoH/ACNet) |
| **DeepLab: Semantic Image Segmentation with Atrous Convolution** | TPAMI 2018 | [ASPP](https://github.com/kazuto1011/deeplab-pytorch) |
| **MixConv: Mixed Depthwise Convolutional Kernels** | BMVC 2019 | [MixConv](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet) |
| **Pyramid Scene Parsing Network** | CVPR 2017 | [PSP](https://github.com/hszhao/PSPNet) |
| **Receptive Field Block Net for Accurate Object Detection** | ECC Dense | [RFB](https://github.com/GOATmessi7/RFBNet) |
| **Strip Pooling: Rethinking Spatial Pooling for Scene Parsing** | CVPR 2020 | [SPNet](https://github.com/Andrew-Qibin/SPNet) |
| **EfficientNet: Rethinking Model Scaling for CNNs** | ICML 2019 | [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch) |
| **PP-NAS: Searching for Plug-and-Play Blocks on CNN** | ICCVW 2021 | [PP-NAS](https://github.com/sbl1996/PP-NAS) |

---

## Dynamic Networks

Dynamic networks adaptively adjust their architecture or parameters during inference for efficiency and performance.

| Title | Publish | Code |
|-------|---------|------|
| **CondConv: Conditionally Parameterized Convolutions** | NeurIPS 2019 | [CondConv](https://github.com/d-li14/condconv.pytorch) |
| **Dynamic Convolution: Attention over Convolution Kernels** | CVPR 2020 | [DynamicConv](https://github.com/kaijieshi7/Dynamic-convolution-Pytorch) |
| **WeightNet: Revisiting the Design Space of Weight Network** | ECCV 2020 | [WeightNet](https://github.com/megvii-model/WeightNet) |
| **SkipNet: Learning Dynamic Routing in Convolutional Networks** | ECCV 2018 | [SkipNet](https://github.com/ucbdrive/skipnet) |
| **Dynamic Group Convolution for Accelerating CNNs** | ECCV 2020 | [DGC](https://github.com/zhuogege1943/dgc) |

---

## Vision Transformers

Vision transformers leverage self-attention for image recognition tasks, achieving state-of-the-art performance.

| Title | Publish | Code | Main Idea |
|-------|---------|------|-----------|
| **An Image is Worth 16x16 Words: Transformers for Image Recognition** | ICLR 2021 | [ViT](https://github.com/lucidrains/vit-pytorch) | Vision Transformer |
| **Swin Transformer: Hierarchical Vision Transformer** | ICCV 2021 | [SwinT](https://github.com/microsoft/Swin-Transformer) | Shifted Windows |
| **CvT: Introducing Convolutions to Vision Transformers** | ICCV 2021 | [CvT](https://github.com/microsoft/CvT) | Convolution Projection |
| **ConViT: Improving Vision Transformers with Soft Convolutional Biases** | CoRR 2021 | [ConViT](https://github.com/facebookresearch/convit) | GPSA |
| **DeiT: Data-Efficient Image Transformers** | ICML 2021 | [DeiT](https://github.com/facebookresearch/deit) | Distillation |
| **ResT: An Efficient Transformer for Visual Recognition** | CoRR 2021 | [ResT](https://github.com/wofmanaf/ResT) | Efficient Design |
| **Segment Anything** | CoRR 2023 | [SAM](https://segment-anything.com/) | Universal Segmentation |
| **ConvNeXt V2: Co-designing and Scaling ConvNets** | CoRR 2023 | [ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2) | Masked Autoencoders |

---

## Contributing

We welcome contributions to keep this repository up-to-date! If you know of additional papers, code implementations, or improvements, please:
1. Submit an [issue](https://github.com/[YourUsername]/Awesome-Vision-Attentions/issues) to suggest new resources or report errors.
2. Create a [pull request](https://github.com/[YourUsername]/Awesome-Vision-Attentions/pulls) with your additions or updates.

**Your Contribution**: [Add your contribution here, e.g., "Added new attention mechanism paper on X" or "Contributed code for Y module"].  
Special thanks to [@dedekinds](https://github.com/dedekinds) for identifying issues with the DIANet description.

---

This `README.md` is designed to be modular and easy to extend. Replace `[YourUsername]` with your actual GitHub username when adding to your repository. You can insert your specific contribution in the "Your Contribution" placeholder. Let me know if you need further tweaks or assistance with integrating this into your repository!
