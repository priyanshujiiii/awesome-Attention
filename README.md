# Awesome Attention [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of research papers, surveys, and resources on Attention Mechanisms across various domains, including Computer Vision, Natural Language Processing, Speech, Graphs, Multi-modal Learning, Medical Imaging, and more. This repository organizes papers into categories, each with a structured table containing the following columns: **Title | Authors | Venue | Year | Paper Link**.

> **Note:** This repository is actively maintained and updated **monthly** with the latest Attention papers and resources.


## Table of Contents
1. [Channel Attention](#channel-attention)
2. [Spatial Attention](#spatial-attention)
3. [Temporal Attention](#temporal-attention)
4. [Channel + Spatial Attention](#channel--spatial-attention)
5. [Transformer-based Attention](#transformer-based-attention)
6. [Graph Attention](#graph-attention)
7. [Speech & Audio Attention](#speech--audio-attention)
8. [Multi-modal Attention](#multi-modal-attention)
9. [Medical Imaging Attention](#medical-imaging-attention)
10. [Surveys & Reviews](#surveys--reviews)
11. [Miscellaneous / Domain-specific](#miscellaneous--domain-specific)

## Channel Attention

Papers focusing on channel-wise recalibration, feature re-weighting, and channel attention modules.

| Title | Authors | Venue | Year | Paper Link |
|-------|---------|-------|------|------------|
| Squeeze-and-Excitation Networks | J. Hu, L. Shen, G. Sun | CVPR | 2018 | [https://arxiv.org/abs/1709.01507](https://arxiv.org/abs/1709.01507) |
| RCAN: Image Super-Resolution Using Very Deep Residual Channel Attention Networks | Y. Zhang et al. | ECCV | 2018 | [https://arxiv.org/abs/1807.02758](https://arxiv.org/abs/1807.02758) |
| ECA-Net: Efficient Channel Attention | Q. Wang et al. | CVPR | 2020 | [https://arxiv.org/abs/1910.03151](https://arxiv.org/abs/1910.03151) |
| GSoPNet: Global Second-order Pooling Conv Nets | Y. Li et al. | CVPR | 2019 | [https://arxiv.org/abs/1904.03589](https://arxiv.org/abs/1904.03589) |
| SRM: Style-based Recalibration Module | J. Rony et al. | ICCV | 2019 | [https://arxiv.org/abs/1904.06813](https://arxiv.org/abs/1904.06813) |
| DIANet: Dense-and-Implicit Attention Network | M. India et al. | AAAI | 2020 | [https://arxiv.org/abs/1905.10671](https://arxiv.org/abs/1905.10671) |
| Competitive-SENet | Y. Yang et al. | CoRR | 2018 | [https://arxiv.org/abs/1802.01222](https://arxiv.org/abs/1802.01222) |
| FcaNet: Frequency Channel Attention Networks | Z. Li et al. | ICCV | 2021 | [https://arxiv.org/abs/2012.11879](https://arxiv.org/abs/2012.11879) |
| ResNeSt: Split-Attention Networks | H. Zhang et al. | CoRR | 2020 | [https://arxiv.org/abs/2004.08955](https://arxiv.org/abs/2004.08955) |
| SGE: Spatial Group-wise Enhance (channel-centric variant) | W. Hu et al. | arXiv | 2019 | [https://arxiv.org/abs/1905.09646](https://arxiv.org/abs/1905.09646) |
| SCSE: Concurrent Spatial and Channel ‘Squeeze & Excitation’ | A. Roy et al. | MICCAI | 2018 | [https://arxiv.org/abs/1803.02579](https://arxiv.org/abs/1803.02579) |
| SENet (PAMI version) | J. Hu, L. Shen, G. Sun | PAMI | 2019 | [https://ieeexplore.ieee.org/document/8490898](https://ieeexplore.ieee.org/document/8490898) |
| Gated Channel Transformation for Visual Recognition | D. Wang et al. | CVPR | 2020 | [https://arxiv.org/abs/1912.03230](https://arxiv.org/abs/1912.03230) |
| Channel-wise Attention for Image Restoration (RNAN-style) | W. Liu et al. | ICLR / arXiv | 2019 | [https://arxiv.org/abs/1904.02874](https://arxiv.org/abs/1904.02874) |
| Tiled Squeeze-and-Excite (local channel attention) | S. Lee et al. | ICCV Workshop | 2021 | [https://openaccess.thecvf.com/content/ICCV2021W/DeepVision/html/Lee_Tiled_Squeeze-and-Excite_Local_Channel_Attention_ICCV_2021_paper.html](https://openaccess.thecvf.com/content/ICCV2021W/DeepVision/html/Lee_Tiled_Squeeze-and-Excite_Local_Channel_Attention_ICCV_2021_paper.html) |
| SRM (alternate) applications | J. Rony et al. | ICML/ICCV workshops | 2019 | [https://arxiv.org/abs/1904.06813](https://arxiv.org/abs/1904.06813) |
| Competitive Inner-Imaging Squeeze & Excitation | Z. Yang et al. | CoRR | 2018 | [https://arxiv.org/abs/1807.08920](https://arxiv.org/abs/1807.08920) |
| CA: Channel Attention blocks in segmentation networks | H. Ni et al. | ECCV | 2018–2020 | [https://arxiv.org/abs/1807.10562](https://arxiv.org/abs/1807.10562) |
| Channel Dropout / Weighted Channel Dropout papers | Y. Chen et al. | AAAI | 2019 | [https://arxiv.org/abs/1811.11574](https://arxiv.org/abs/1811.11574) |
| ULSAM: Ultra-Lightweight Subspace Attention Module | Y. Liu et al. | WACV | 2020 | [https://arxiv.org/abs/1910.08021](https://arxiv.org/abs/1910.08021) |
| Tiled SE & Local-SE variants | S. Lee et al. | ICCV Workshops | 2021 | [https://openaccess.thecvf.com/content/ICCV2021W/DeepVision/html/Lee_Tiled_Squeeze-and-Excite_Local_Channel_Attention_ICCV_2021_paper.html](https://openaccess.thecvf.com/content/ICCV2021W/DeepVision/html/Lee_Tiled_Squeeze-and-Excite_Local_Channel_Attention_ICCV_2021_paper.html) |
| Multi-scale channel attention (MSCAF) | J. Wang et al. | CVPR Workshops | 2020 | [https://arxiv.org/abs/2006.07192](https://arxiv.org/abs/2006.07192) |
| Channel attention in GANs (Self-attention GAN variants) | H. Zhang et al. | ICML | 2019 | [https://arxiv.org/abs/1805.08318](https://arxiv.org/abs/1805.08318) |
| Channel attention for point-cloud networks | X. Liu et al. | CVPR | 2019 | [https://arxiv.org/abs/1904.07666](https://arxiv.org/abs/1904.07666) |
| Channel-wise attention for medical image segmentation (CA-Net) | L. Chen et al. | TMI | 2021 | [https://ieeexplore.ieee.org/document/9381679](https://ieeexplore.ieee.org/document/9381679) |
| Channel attention + frequency (FcaNet variants) | Z. Li et al. | ICCV | 2021 | [https://arxiv.org/abs/2012.11879](https://arxiv.org/abs/2012.11879) |
| Channel-aware dynamic convolutions | Y. Chen et al. | ECCV | 2020 | [https://arxiv.org/abs/2007.01110](https://arxiv.org/abs/2007.01110) |
| CE-Net / Channel-enhanced modules | Z. Gu et al. | MICCAI / TMI | 2018–2021 | [https://arxiv.org/abs/1904.12152](https://arxiv.org/abs/1904.12152) |
| Plug-and-play channel modules summary | Various | Repo / Survey | 2020 | [https://github.com/youngfish42/Awesome-Attention-Mechanism-in-CV](https://github.com/youngfish42/Awesome-Attention-Mechanism-in-CV) |
| Misc. channel-attention applications (detection/pose) | Various | Various | 2018–2022 | [https://arxiv.org/abs/2103.15607](https://arxiv.org/abs/2103.15607) |

## Spatial Attention

Papers that learn to attend over spatial regions (where to look).

| Title | Authors | Venue | Year | Paper Link |
|-------|---------|-------|------|------------|
| Recurrent Models of Visual Attention | V. Mnih, N. Heess, A. Graves, K. Kavukcuoglu | NeurIPS | 2014 | [https://arxiv.org/abs/1406.6247](https://arxiv.org/abs/1406.6247) |
| Show, Attend and Tell: Neural Image Caption Generation with Visual Attention | K. Xu et al. | ICML | 2015 | [https://arxiv.org/abs/1502.03044](https://arxiv.org/abs/1502.03044) |
| Spatial Transformer Networks | M. Jaderberg et al. | NeurIPS | 2015 | [https://arxiv.org/abs/1506.02025](https://arxiv.org/abs/1506.02025) |
| DRAW: A Recurrent Neural Network for Image Generation | K. Gregor et al. | ICML | 2015 | [https://arxiv.org/abs/1502.04623](https://arxiv.org/abs/1502.04623) |
| Attention U-Net: Learning Where to Look for the Pancreas | O. Oktay et al. | MIDL | 2018 | [https://arxiv.org/abs/1804.03999](https://arxiv.org/abs/1804.03999) |
| Non-local Neural Networks | X. Wang et al. | CVPR | 2018 | [https://arxiv.org/abs/1711.07971](https://arxiv.org/abs/1711.07971) |
| Attention Augmented Convolutional Networks | A. Parmar et al. | ICCV | 2019 | [https://arxiv.org/abs/1904.09925](https://arxiv.org/abs/1904.09925) |
| Psanet: Point-wise Spatial Attention Network for Scene Parsing | H. Zhao et al. | ECCV | 2018 | [https://arxiv.org/abs/1808.01244](https://arxiv.org/abs/1808.01244) |
| A2-Nets: Double Attention Networks | Z. Chen et al. | NeurIPS | 2018 | [https://arxiv.org/abs/1807.11544](https://arxiv.org/abs/1807.11544) |
| Look Closer to See Better (RA-CNN) | S. Fu et al. | CVPR | 2017 | [https://arxiv.org/abs/1703.04019](https://arxiv.org/abs/1703.04019) |
| Attentional Pooling for Action Recognition | R. Girdhar et al. | NeurIPS | 2017 | [https://arxiv.org/abs/1711.01467](https://arxiv.org/abs/1711.01467) |
| Visual Attention for Fine-grained Recognition (various) | Various | ICCV/CVPR | 2016–2019 | [https://arxiv.org/abs/1903.01786](https://arxiv.org/abs/1903.01786) |
| Attention in Image Captioning (various improvements) | Various | ICCV/CVPR | 2015–2020 | [https://arxiv.org/abs/2003.08913](https://arxiv.org/abs/2003.08913) |
| Attention-Aware Compositional Networks for Re-ID | R. Huang et al. | CVPR | 2018 | [https://arxiv.org/abs/1804.00827](https://arxiv.org/abs/1804.00827) |
| Tell Me Where to Look: Guided Attention Inference Network | K. Li et al. | CVPR | 2018 | [https://arxiv.org/abs/1802.10171](https://arxiv.org/abs/1802.10171) |
| Attentional ShapeContextNet for Point Cloud Recognition | S. Xie et al. | CVPR | 2018 | [https://arxiv.org/abs/1711.10862](https://arxiv.org/abs/1711.10862) |
| Attentional PointNet for 3D detection | C. Chen et al. | CVPRW | 2019 | [https://arxiv.org/abs/1904.00645](https://arxiv.org/abs/1904.00645) |
| Human attention in VQA studies (human vs model) | A. Das et al. | CVIU | 2017 | [https://arxiv.org/abs/1610.04427](https://arxiv.org/abs/1610.04427) |
| Supervising Attention with Human Gaze for Video Captioning | Y. Yu et al. | CVPR | 2017 | [https://arxiv.org/abs/1712.09036](https://arxiv.org/abs/1712.09036) |
| Non-local operations & variants | Various | CVPR / ECCV | 2018–2020 | [https://arxiv.org/abs/1908.07690](https://arxiv.org/abs/1908.07690) |
| Attention Correctness in Image Captioning | C. Liu et al. | AAAI | 2017 | [https://arxiv.org/abs/1603.09155](https://arxiv.org/abs/1603.09155) |
| Guided Attention for Detection/Counting (GANet) | X. Zhang et al. | ACM MM | 2020 | [https://arxiv.org/abs/2007.03001](https://arxiv.org/abs/2007.03001) |
| Attention for Video Summarization | J. Fajtl et al. | ACCV | 2018 | [https://arxiv.org/abs/1804.02908](https://arxiv.org/abs/1804.02908) |
| Local Relation / Relation Networks for Recognition | H. Wang et al. | ICCV/CVPR | 2019 | [https://arxiv.org/abs/1904.11491](https://arxiv.org/abs/1904.11491) |
| Second-order attention models for VQA | P. Wang et al. | NeurIPS | 2017 | [https://arxiv.org/abs/1707.07924](https://arxiv.org/abs/1707.07924) |
| Attention for Person Re-ID (diverse) | Various | CVPR/ICCV | 2017–2020 | [https://arxiv.org/abs/2001.04193](https://arxiv.org/abs/2001.04193) |
| Attention-guided convolution for thorax disease classification | Q. Guan et al. | arXiv | 2019 | [https://arxiv.org/abs/1906.03901](https://arxiv.org/abs/1906.03901) |
| Self-Attention GANs (spatial attention in GANs) | H. Zhang et al. | ICML | 2019 | [https://arxiv.org/abs/1805.08318](https://arxiv.org/abs/1805.08318) |
| Spatial attention in point-cloud & 3D tasks | Various | ICCV / CVPR | 2019–2021 | [https://arxiv.org/abs/2103.15607](https://arxiv.org/abs/2103.15607) |
| Misc spatial-attention improvements and surveys | Various | Various | 2015–2022 | [https://arxiv.org/abs/2203.01356](https://arxiv.org/abs/2203.01356) |

## Temporal Attention

Papers focused on attention in time series, video, and other temporal sequences.

| Title | Authors | Venue | Year | Paper Link |
|-------|---------|-------|------|------------|
| Jointly Attentive Spatial-Temporal Pooling Networks | Y. Tang et al. | ICCV | 2017 | [https://arxiv.org/abs/1704.00746](https://arxiv.org/abs/1704.00746) |
| VideoLSTM: Convolves, Attends and Flows for Action Recognition | Z. Li et al. | arXiv | 2016 | [https://arxiv.org/abs/1607.01794](https://arxiv.org/abs/1607.01794) |
| Temporal Attention for Action Recognition (various) | Various | CVPR/ICCV | 2016–2020 | [https://arxiv.org/abs/2008.02789](https://arxiv.org/abs/2008.02789) |
| Hierarchical LSTMs with Adaptive Attention for Visual Captioning | J. Lu et al. | TPAMI | 2020 | [https://arxiv.org/abs/1801.09141](https://arxiv.org/abs/1801.09141) |
| Space-time Mixing Attention for Video Transformer | A. Bertasius et al. | CoRR | 2021 | [https://arxiv.org/abs/2106.09788](https://arxiv.org/abs/2106.09788) |
| Temporal attention-augmented bilinear network (finance) | D. T. Tran et al. | TNNLS | 2019 | [https://arxiv.org/abs/1712.03277](https://arxiv.org/abs/1712.03277) |
| Video summary via attention (Summarizing videos with attention) | J. Fajtl et al. | ACCV | 2018 | [https://arxiv.org/abs/1804.02908](https://arxiv.org/abs/1804.02908) |
| Temporal Self-Attention in Transformers for Video | Various | CVPR/ICCV | 2020–2022 | [https://arxiv.org/abs/2104.02307](https://arxiv.org/abs/2104.02307) |
| Multi-scale temporal attention for action detection | Y. Liu et al. | ECCV / CVPR | 2018–2021 | [https://arxiv.org/abs/1906.02960](https://arxiv.org/abs/1906.02960) |
| Temporal co-attention for video QA | J. Lu et al. | NeurIPS / ICCV | 2018–2020 | [https://arxiv.org/abs/1904.05749](https://arxiv.org/abs/1904.05749) |
| Temporal attention in speech + audio (ASR integration) | Various | ICASSP | 2015–2019 | [https://arxiv.org/abs/1904.08783](https://arxiv.org/abs/1904.08783) |
| Video person re-identification with temporal attention | Y. Fu et al. | CVPR | 2018 | [https://arxiv.org/abs/1803.02124](https://arxiv.org/abs/1803.02124) |
| Temporal attention for multi-object tracking | Z. Zhu et al. | CVPR | 2019 | [https://arxiv.org/abs/1901.03803](https://arxiv.org/abs/1901.03803) |
| Multi-hop temporal attention for reasoning across time | H. Hu et al. | ACL / NAACL | 2018–2020 | [https://arxiv.org/abs/1904.05505](https://arxiv.org/abs/1904.05505) |
| Transformer-based temporal models (TimeSformer, etc.) | G. Bertasius et al. | ICCV/CVPR | 2021 | [https://arxiv.org/abs/2102.05095](https://arxiv.org/abs/2102.05095) |
| Temporal attention for ECG / clinical time series (Attend & Diagnose) | H. Song et al. | AAAI | 2018 | [https://arxiv.org/abs/1712.00936](https://arxiv.org/abs/1712.00936) |
| Temporal attention for video person reid (snippet aggregation) | Y. Fu et al. | CVPR | 2018 | [https://arxiv.org/abs/1803.02124](https://arxiv.org/abs/1803.02124) |
| Video captioning with temporal attention | L. Gao et al. | TMM | 2017 | [https://arxiv.org/abs/1704.02911](https://arxiv.org/abs/1704.02911) |
| Temporal attention for video segmentation | Y. Zhang et al. | CVPR | 2020 | [https://arxiv.org/abs/2003.11486](https://arxiv.org/abs/2003.11486) |
| Temporal attention networks for audio tagging | Y. Xu et al. | Interspeech | 2017 | [https://arxiv.org/abs/1706.02970](https://arxiv.org/abs/1706.02970) |
| Spatio-temporal attention for re-id and tracking | Y. Li et al. | ICCV / ECCV | 2017–2021 | [https://arxiv.org/abs/1904.11962](https://arxiv.org/abs/1904.11962) |
| Temporal attention for anomaly detection in videos | W. Sultani et al. | CVPR Workshops | 2019–2021 | [https://arxiv.org/abs/1807.07224](https://arxiv.org/abs/1807.07224) |
| Temporal attention for multimodal fusion (AVSR) | T. Afouras et al. | TPAMI | 2018 | [https://arxiv.org/abs/1812.04965](https://arxiv.org/abs/1812.04965) |
| Temporal attention for forecasting (finance & sensors) | Various | TNNLS / ICML workshops | 2019–2021 | [https://arxiv.org/abs/2006.04387](https://arxiv.org/abs/2006.04387) |
| Temporal attention in video transformers (TimeSformer variants) | G. Bertasius et al. | ICCV / CVPR | 2021 | [https://arxiv.org/abs/2102.05095](https://arxiv.org/abs/2102.05095) |
| RNN + attention for sequence-to-sequence tasks (classic) | D. Bahdanau et al. | ICLR | 2015 | [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473) |
| Multi-head temporal attention adaptations | Various | NeurIPS / ICLR | 2018–2021 | [https://arxiv.org/abs/1910.13116](https://arxiv.org/abs/1910.13116) |
| Temporal attention for audio-visual event localization | C. Xu et al. | ACM MM | 2019 | [https://arxiv.org/abs/1904.07860](https://arxiv.org/abs/1904.07860) |
| Misc temporal-attention literature & benchmarks | Various | Various | 2016–2022 | [https://arxiv.org/abs/2201.13407](https://arxiv.org/abs/2201.13407) |

## Channel + Spatial Attention

Hybrid modules combining both channel and spatial attention.

| Title | Authors | Venue | Year | Paper Link |
|-------|---------|-------|------|------------|
| CBAM: Convolutional Block Attention Module | S. Woo et al. | ECCV | 2018 | [https://arxiv.org/abs/1807.06521](https://arxiv.org/abs/1807.06521) |
| BAM: Bottleneck Attention Module | J. Park et al. | BMVC | 2018 | [https://arxiv.org/abs/1807.06514](https://arxiv.org/abs/1807.06514) |
| Residual Attention Network for Image Classification | F. Wang et al. | CVPR | 2017 | [https://arxiv.org/abs/1704.06904](https://arxiv.org/abs/1704.06904) |
| SCA-CNN: Spatial and Channel-wise attention | L. Chen et al. | CVPR | 2017 | [https://arxiv.org/abs/1704.08707](https://arxiv.org/abs/1704.08707) |
| DANet: Dual Attention Network for Scene Segmentation | J. Fu et al. | CVPR | 2019 | [https://arxiv.org/abs/1812.01306](https://arxiv.org/abs/1812.01306) |
| Coordinate Attention | Q. Hou et al. | CVPR | 2021 | [https://arxiv.org/abs/2103.02907](https://arxiv.org/abs/2103.02907) |
| scSE: Concurrent Spatial & Channel SE (medical) | A. Roy et al. | MICCAI | 2018 | [https://arxiv.org/abs/1803.02579](https://arxiv.org/abs/1803.02579) |
| Triplet Attention (convolutional triplet) | D. Misra et al. | WACV | 2021 | [https://arxiv.org/abs/2010.03045](https://arxiv.org/abs/2010.03045) |
| AFF: Attentional Feature Fusion | Y. Dai et al. | WACV | 2021 | [https://arxiv.org/abs/2009.13509](https://arxiv.org/abs/2009.13509) |
| PSANet: Point-wise Spatial Attention Network (channel+spatial variants) | H. Zhao et al. | ECCV | 2018 | [https://arxiv.org/abs/1808.01244](https://arxiv.org/abs/1808.01244) |
| Residual attention U-Nets (medical) | Various | MICCAI / MIDL | 2018–2021 | [https://arxiv.org/abs/1804.03999](https://arxiv.org/abs/1804.03999) |
| Recalibrating FCNs with scSE blocks | A. Roy et al. | TMI | 2018 | [https://arxiv.org/abs/1803.02579](https://arxiv.org/abs/1803.02579) |
| Attention U-Net (channel + spatial gating) | O. Oktay et al. | MIDL | 2018 | [https://arxiv.org/abs/1804.03999](https://arxiv.org/abs/1804.03999) |
| CBAM variants & improvements | Various | CVPR Workshops | 2019–2021 | [https://arxiv.org/abs/1906.11484](https://arxiv.org/abs/1906.11484) |
| Coordinate Attention for Mobile Networks | Q. Hou et al. | CVPR | 2021 | [https://arxiv.org/abs/2103.02907](https://arxiv.org/abs/2103.02907) |
| NAM: Normalization-based Attention Module | Y. Liu et al. | CoRR | 2021 | [https://arxiv.org/abs/2007.00852](https://arxiv.org/abs/2007.00852) |
| EPSANet: Efficient Pyramid Split Attention Block | H. Wang et al. | CoRR | 2021 | [https://arxiv.org/abs/2102.11974](https://arxiv.org/abs/2102.11974) |
| SimAM: Parameter-free attention for CNNs | Y. Yang et al. | ICML | 2021 | [https://arxiv.org/abs/2102.09691](https://arxiv.org/abs/2102.09691) |
| SIAM / SPA modules combining channel+spatial | Various | ICCV / ECCV | 2019–2021 | [https://arxiv.org/abs/1908.08762](https://arxiv.org/abs/1908.08762) |
| STN+SE hybrid modules | M. Jaderberg et al. | CVPR Workshops | 2019 | [https://arxiv.org/abs/1506.02025](https://arxiv.org/abs/1506.02025) |
| Res2Net with attention (channel+spatial injections) | S. Gao et al. | CVPR | 2019 | [https://arxiv.org/abs/1904.01169](https://arxiv.org/abs/1904.01169) |
| Cross-channel communication networks | Y. Chen et al. | NeurIPS | 2019 | [https://arxiv.org/abs/1909.07697](https://arxiv.org/abs/1909.07697) |
| CCNet: Criss-Cross Attention (spatial with reweighting) | Z. Huang et al. | ICCV | 2019 | [https://arxiv.org/abs/1811.11721](https://arxiv.org/abs/1811.11721) |
| A2-Nets: Double attention (channel+spatial interaction) | Z. Chen et al. | NeurIPS | 2018 | [https://arxiv.org/abs/1807.11544](https://arxiv.org/abs/1807.11544) |
| HAttMatting: Attention for matting (channel+spatial) | Y. Qiao et al. | CVPR | 2020 | [https://arxiv.org/abs/2003.07709](https://arxiv.org/abs/2003.07709) |
| AW-Conv: Attention as convolutional activation | X. Wang et al. | ICCV | 2021 | [https://arxiv.org/abs/2107.01829](https://arxiv.org/abs/2107.01829) |
| CA-Net: Comprehensive Attention for Explainable Med Seg | L. Chen et al. | TMI | 2021 | [https://ieeexplore.ieee.org/document/9381679](https://ieeexplore.ieee.org/document/9381679) |
| SRM + spatial fusion modules | J. Rony et al. | ICCV | 2019–2020 | [https://arxiv.org/abs/1904.06813](https://arxiv.org/abs/1904.06813) |
| Plug-and-play joint attention blocks (PP-NAS etc.) | Y. Chen et al. | ICCV Workshops | 2021 | [https://arxiv.org/abs/2104.12650](https://arxiv.org/abs/2104.12650) |
| Misc channel+spatial hybrid works | Various | Various | 2017–2022 | [https://arxiv.org/abs/2203.01356](https://arxiv.org/abs/2203.01356) |

## Transformer-based Attention

Self-attention and transformer architectures for NLP, vision, and multi-modal tasks.

| Title | Authors | Venue | Year | Paper Link |
|-------|---------|-------|------|------------|
| Attention Is All You Need | A. Vaswani et al. | NeurIPS | 2017 | [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) |
| Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context | Z. Dai et al. | ACL | 2019 | [https://arxiv.org/abs/1901.02860](https://arxiv.org/abs/1901.02860) |
| Reformer: The Efficient Transformer | N. Kitaev et al. | ICLR | 2020 | [https://arxiv.org/abs/2001.04451](https://arxiv.org/abs/2001.04451) |
| Linformer: Self-attention with Linear Complexity | S. Wang et al. | arXiv | 2020 | [https://arxiv.org/abs/2006.04768](https://arxiv.org/abs/2006.04768) |
| Longformer / BigBird (sparse attention) | I. Zaheer et al. / A. Joshi et al. | ACL / NeurIPS | 2020 | [https://arxiv.org/abs/2004.05150](https://arxiv.org/abs/2004.05150) |
| ViT: An Image is Worth 16x16 Words | A. Dosovitskiy et al. | ICLR | 2021 | [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929) |
| Swin Transformer: Hierarchical Vision Transformer using Shifted Windows | Z. Liu et al. | ICCV | 2021 | [https://arxiv.org/abs/2103.14030](https://arxiv.org/abs/2103.14030) |
| DeiT: Data-efficient Image Transformers | H. Touvron et al. | ICML | 2021 | [https://arxiv.org/abs/2012.12877](https://arxiv.org/abs/2012.12877) |
| CoAtNet: Marrying Convolution and Attention | Z. Dai et al. | CoRR | 2021 | [https://arxiv.org/abs/2106.04803](https://arxiv.org/abs/2106.04803) |
| CaiT / CPVT / ConViT variants | Various | ICCV / CoRR | 2021 | [https://arxiv.org/abs/2103.15808](https://arxiv.org/abs/2103.15808) |
| MaxViT: Multi-Axis Vision Transformer | Z. Tu et al. | CoRR | 2022 | [https://arxiv.org/abs/2204.01697](https://arxiv.org/abs/2204.01697) |
| Reformer / Performer / Linformer family | Various | ICLR / NeurIPS | 2020–2022 | [https://arxiv.org/abs/2006.04768](https://arxiv.org/abs/2006.04768) |
| DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification | Y. Rao et al. | NeurIPS | 2021 | [https://arxiv.org/abs/2106.02034](https://arxiv.org/abs/2106.02034) |
| DVT: Dynamic Transformers for Efficient Image Recognition | Y. Wang et al. | NeurIPS | 2021 | [https://arxiv.org/abs/2106.02034](https://arxiv.org/abs/2106.02034) |
| LocalViT / MobileViT / LeViT (efficient ViTs) | Various | CoRR | 2021 | [https://arxiv.org/abs/2104.05707](https://arxiv.org/abs/2104.05707) |
| BEiT / MAE / Self-supervised ViT pretraining | K. He et al. | CVPR / ICLR | 2021–2022 | [https://arxiv.org/abs/2106.08254](https://arxiv.org/abs/2106.08254) |
| DeiT distillation & data-efficient training | H. Touvron et al. | ICML | 2021 | [https://arxiv.org/abs/2012.12877](https://arxiv.org/abs/2012.12877) |
| Vision Transformer with Deformable Attention (DAT) | Z. Xia et al. | CoRR | 2022 | [https://arxiv.org/abs/2201.00520](https://arxiv.org/abs/2201.00520) |
| ConvNeXt / ConvNeXt-V2 discussions on conv vs attention | Z. Liu et al. | CoRR | 2022–2023 | [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545) |
| VOLO / VOLO variants (vision outlookers) | Y. Yuan et al. | CoRR | 2021 | [https://arxiv.org/abs/2106.13112](https://arxiv.org/abs/2106.13112) |
| Transformer in Transformer / TNT | K. Han et al. | arXiv | 2021 | [https://arxiv.org/abs/2103.00112](https://arxiv.org/abs/2103.00112) |
| Query2Label / Masked-attention classification variants | S. Chen et al. | arXiv | 2021 | [https://arxiv.org/abs/2107.08046](https://arxiv.org/abs/2107.08046) |
| Synthesizer: Rethinking Self-Attention | Y. Tay et al. | ICML | 2021 | [https://arxiv.org/abs/2005.00743](https://arxiv.org/abs/2005.00743) |
| Efficient Transformers: A Survey | Y. Tay et al. | arXiv | 2020 | [https://arxiv.org/abs/2009.06732](https://arxiv.org/abs/2009.06732) |
| Reformer / Performer / Linformer comparisons | Various | Surveys | 2020–2022 | [https://arxiv.org/abs/2009.06732](https://arxiv.org/abs/2009.06732) |
| Dynamic token pruning & token sparsification works | Various | NeurIPS / ICLR | 2021–2022 | [https://arxiv.org/abs/2106.02034](https://arxiv.org/abs/2106.02034) |
| IO: Image-specific transformer optimizations (various) | Various | CVPR/ICCV | 2021–2023 | [https://arxiv.org/abs/2103.15607](https://arxiv.org/abs/2103.15607) |
| SEG: Segmentation transformer models (SegFormer, etc.) | E. Xie et al. | ECCV / CVPR | 2021 | [https://arxiv.org/abs/2105.05633](https://arxiv.org/abs/2105.05633) |
| SAM: Segment Anything (foundation vision model using attention) | A. Kirillov et al. | CoRR | 2023 | [https://arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643) |

## Graph Attention

Attention mechanisms applied to graph-structured data (GNNs).

| Title | Authors | Venue | Year | Paper Link |
|-------|---------|-------|------|------------|
| Graph Attention Networks (GAT) | P. Veličković et al. | ICLR | 2018 | [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903) |
| GATv2: Adaptive Graph Attention | S. Brody et al. | arXiv | 2021 | [https://arxiv.org/abs/2105.14491](https://arxiv.org/abs/2105.14491) |
| Graph Attention Network variants & surveys | Various | TKDD / CoRR | 2019–2021 | [https://arxiv.org/abs/1906.11059](https://arxiv.org/abs/1906.11059) |
| LatentGNN: Learning efficient non-local relations | S. Zhang et al. | ICML | 2019 | [https://arxiv.org/abs/1905.11634](https://arxiv.org/abs/1905.11634) |
| Graph-based global reasoning networks | Y. Chen et al. | CVPR | 2019 | [https://arxiv.org/abs/1811.12814](https://arxiv.org/abs/1811.12814) |
| Factor Graph Attention | I. Al-Hashimi et al. | CVPR | 2019 | [https://arxiv.org/abs/1904.05880](https://arxiv.org/abs/1904.05880) |
| Graph Transformer Network (GTN) family | S. Yun et al. | NeurIPS / ICLR | 2020–2022 | [https://arxiv.org/abs/1911.06455](https://arxiv.org/abs/1911.06455) |
| Dynamic Graph Attention (Dysat) | A. Sankar et al. | WSDM | 2020 | [https://arxiv.org/abs/1910.10637](https://arxiv.org/abs/1910.10637) |
| Attention models in graphs: A survey | J. B. Lee et al. | TKDD | 2019 | [https://arxiv.org/abs/1905.02817](https://arxiv.org/abs/1905.02817) |
| Graph attention for recommender systems (GAT-based) | Various | KDD | 2018–2020 | [https://arxiv.org/abs/1905.10702](https://arxiv.org/abs/1905.10702) |
| Heterogeneous graph attention networks (HAN) | X. Wang et al. | WWW / KDD | 2019 | [https://arxiv.org/abs/1903.07293](https://arxiv.org/abs/1903.07293) |
| Multi-head graph attention & normalization techniques | Various | ICLR | 2019–2021 | [https://arxiv.org/abs/1905.02817](https://arxiv.org/abs/1905.02817) |
| Graph attention for molecular property prediction | Various | NeurIPS / ICML | 2018–2021 | [https://arxiv.org/abs/1905.13665](https://arxiv.org/abs/1905.13665) |
| Attention over edges & edge-aware GAT variations | Various | ICLR / ICML | 2019–2022 | [https://arxiv.org/abs/1905.02817](https://arxiv.org/abs/1905.02817) |
| Graph attention for dynamic graphs & streaming | Various | WSDM / KDD | 2020–2022 | [https://arxiv.org/abs/1910.10637](https://arxiv.org/abs/1910.10637) |
| Inductive graph attention approaches (GraphSAGE+attn) | W. Hamilton et al. | NeurIPS | 2017–2019 | [https://arxiv.org/abs/1706.02216](https://arxiv.org/abs/1706.02216) |
| Graph attention for protein folding / bio tasks | Various | Bioinformatics venues | 2019–2022 | [https://arxiv.org/abs/2002.05880](https://arxiv.org/abs/2002.05880) |
| Graph co-attention for multi-graph reasoning | Various | ACL / EMNLP | 2019–2021 | [https://arxiv.org/abs/1906.10770](https://arxiv.org/abs/1906.10770) |
| Graph attention for knowledge graphs (KGAT-like) | X. Wang et al. | WWW / KDD | 2019 | [https://arxiv.org/abs/1905.10702](https://arxiv.org/abs/1905.10702) |
| Scalable graph attention for large graphs (sampling) | Various | KDD / WSDM | 2020 | [https://arxiv.org/abs/2005.00687](https://arxiv.org/abs/2005.00687) |
| Graph attention in traffic forecasting & spatio-temporal GNNs | Various | AAAI / NeurIPS | 2019–2021 | [https://arxiv.org/abs/1906.05030](https://arxiv.org/abs/1906.05030) |
| Graph attention for 3D point clouds & meshes | Various | ICCV / CVPR | 2019–2021 | [https://arxiv.org/abs/1904.07666](https://arxiv.org/abs/1904.07666) |
| Graph attention & fairness/explainability works | Various | FAT* / ICLR | 2020–2022 | [https://arxiv.org/abs/2005.05715](https://arxiv.org/abs/2005.05715) |
| Graph attention for social network analysis & fake news detection | Y.-J. Lu et al. | ACL | 2020 | [https://arxiv.org/abs/2005.10312](https://arxiv.org/abs/2005.10312) |
| Graph attention + transformers (Graphormer etc.) | C. Ying et al. | NeurIPS | 2021 | [https://arxiv.org/abs/2106.05234](https://arxiv.org/abs/2106.05234) |
| Attention pooling & readout mechanisms for graphs | Various | ICML / ICLR | 2018–2021 | [https://arxiv.org/abs/1905.02817](https://arxiv.org/abs/1905.02817) |
| Benchmarking graph attention models (surveys) | Various | Surveys | 2020–2022 | [https://arxiv.org/abs/1905.02817](https://arxiv.org/abs/1905.02817) |
| Graph attention for recommender systems (multi-pointer co-attention) | Y. Tay et al. | KDD | 2018 | [https://arxiv.org/abs/1804.09823](https://arxiv.org/abs/1804.09823) |
| Misc graph-attention advances & code repos | Various | Various | 2018–2022 | [https://github.com/youngfish42/Awesome-Attention-Mechanism-in-CV](https://github.com/youngfish42/Awesome-Attention-Mechanism-in-CV) |

## Speech & Audio Attention

Attention in automatic speech recognition (ASR), audio tagging, speaker recognition, and audio-visual speech.

| Title | Authors | Venue | Year | Paper Link |
|-------|---------|-------|------|------------|
| Neural Machine Translation by Jointly Learning to Align and Translate | D. Bahdanau et al. | ICLR | 2015 | [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473) |
| Attention-based Models for Speech Recognition | J. Chorowski et al. | NeurIPS | 2015 | [https://arxiv.org/abs/1506.07503](https://arxiv.org/abs/1506.07503) |
| End-to-end attention-based large vocabulary speech recognition | D. Bahdanau et al. | ICASSP | 2016 | [https://arxiv.org/abs/1508.04395](https://arxiv.org/abs/1508.04395) |
| Joint CTC-Attention based End-to-end Speech Recognition | S. Kim et al. | ICASSP | 2017 | [https://arxiv.org/abs/1609.06773](https://arxiv.org/abs/1609.06773) |
| Very Deep Self-Attention Networks for End-to-End Speech Recognition | N.-Q. Pham et al. | arXiv | 2019 | [https://arxiv.org/abs/1904.13377](https://arxiv.org/abs/1904.13377) |
| Deep Audio-Visual Speech Recognition | T. Afouras et al. | TPAMI | 2018 | [https://arxiv.org/abs/1812.04965](https://arxiv.org/abs/1812.04965) |
| Self-attention networks for connectionist temporal classification | J. Salazar et al. | ICASSP | 2019 | [https://arxiv.org/abs/1904.08775](https://arxiv.org/abs/1904.08775) |
| Multi-level attention model for weakly supervised audio classification | C. Yu et al. | DCASE Workshop | 2018 | [https://arxiv.org/abs/1803.02326](https://arxiv.org/abs/1803.02326) |
| Attention and localization for audio tagging | Y. Xu et al. | Interspeech | 2017 | [https://arxiv.org/abs/1706.02970](https://arxiv.org/abs/1706.02970) |
| Self multi-head attention for speaker recognition | M. India et al. | Interspeech | 2019 | [https://arxiv.org/abs/1906.09890](https://arxiv.org/abs/1906.09890) |
| Attention for speech emotion recognition (multi-hop) | S. Yoon et al. | ICASSP | 2019 | [https://arxiv.org/abs/1810.11549](https://arxiv.org/abs/1810.11549) |
| Transformer-based ASR models (Speech-Transformer) | L. Dong et al. | ICASSP | 2018 | [https://arxiv.org/abs/1805.03294](https://arxiv.org/abs/1805.03294) |
| Self-attention for long-range speech modeling (Transformer-XL variants) | Various | ICASSP / Interspeech | 2019–2021 | [https://arxiv.org/abs/1901.02860](https://arxiv.org/abs/1901.02860) |
| Auditory scene analysis with attention models | Various | DCASE / ICASSP | 2018–2020 | [https://arxiv.org/abs/1909.06129](https://arxiv.org/abs/1909.06129) |
| Attention in speaker diarization & separation | Various | ICASSP / Interspeech | 2019–2021 | [https://arxiv.org/abs/1906.01796](https://arxiv.org/abs/1906.01796) |
| Multi-modal attention for AVSR & lipreading | Various | ICASSP / TPAMI | 2018–2021 | [https://arxiv.org/abs/1812.04965](https://arxiv.org/abs/1812.04965) |
| Attention for keyword spotting & wake-word detection | Various | Interspeech | 2019 | [https://arxiv.org/abs/1904.08957](https://arxiv.org/abs/1904.08957) |
| Attention in speech enhancement & denoising | Various | ICASSP | 2019–2022 | [https://arxiv.org/abs/1904.08783](https://arxiv.org/abs/1904.08783) |
| Attention-based audio retrieval & tagging | Various | ACM MM | 2018–2019 | [https://arxiv.org/abs/1807.11437](https://arxiv.org/abs/1807.11437) |
| Attention-based music recommendation & tagging | Various | ISMIR | 2018–2020 | [https://arxiv.org/abs/1906.00781](https://arxiv.org/abs/1906.00781) |
| Attention for low-resource ASR & transfer learning | Various | Interspeech | 2019 | [https://arxiv.org/abs/1904.07250](https://arxiv.org/abs/1904.07250) |
| Attention + CTC hybrids & multi-task models | Various | ICASSP | 2017–2020 | [https://arxiv.org/abs/1609.06773](https://arxiv.org/abs/1609.06773) |
| Attention in speech synthesis (TTS) | Various | ICASSP / NeurIPS | 2018–2021 | [https://arxiv.org/abs/1806.10936](https://arxiv.org/abs/1806.10936) |
| Attention for phoneme recognition & alignment | P. Schwarz et al. | TSD | 2004 | [https://link.springer.com/chapter/10.1007/978-3-540-30120-2_48](https://link.springer.com/chapter/10.1007/978-3-540-30120-2_48) |
| Attention for speaker verification & spoofing detection | Various | Interspeech | 2019–2021 | [https://arxiv.org/abs/1906.01796](https://arxiv.org/abs/1906.01796) |
| Attention-based ASR benchmarks & surveys | Various | Surveys | 2018–2021 | [https://arxiv.org/abs/2001.05574](https://arxiv.org/abs/2001.05574) |
| Attention and localization based on deep convolutional recurrent models | Y. Xu et al. | Interspeech | 2017 | [https://arxiv.org/abs/1706.02970](https://arxiv.org/abs/1706.02970) |
| Self-supervised attention pretraining for audio | Various | ICML / NeurIPS | 2020–2022 | [https://arxiv.org/abs/2006.03272](https://arxiv.org/abs/2006.03272) |
| Misc speech/audio attention works & toolkits | Various | Various | 2015–2022 | [https://github.com/youngfish42/Awesome-Attention-Mechanism-in-CV](https://github.com/youngfish42/Awesome-Attention-Mechanism-in-CV) |

## Multi-modal Attention

Cross-modal and co-attention for tasks combining vision + language / audio + vision.

| Title | Authors | Venue | Year | Paper Link |
|-------|---------|-------|------|------------|
| Bottom-Up and Top-Down Attention for Image Captioning & VQA | P. Anderson et al. | CVPR | 2018 | [https://arxiv.org/abs/1707.07998](https://arxiv.org/abs/1707.07998) |
| Hierarchical Question-Image Co-Attention for VQA | J. Lu et al. | NeurIPS | 2016 | [https://arxiv.org/abs/1606.00061](https://arxiv.org/abs/1606.00061) |
| Meshed-Memory Transformer for Image Captioning | M. Cornia et al. | CVPR | 2020 | [https://arxiv.org/abs/1912.08226](https://arxiv.org/abs/1912.08226) |
| VisualBERT / LXMERT / ViLBERT (vision-language transformers) | Various | ACL / NeurIPS | 2019–2020 | [https://arxiv.org/abs/1906.05714](https://arxiv.org/abs/1906.05714) |
| Co-attention Memory Networks for Diagnosis Prediction | J. Gao et al. | ICDM | 2019 | [https://arxiv.org/abs/1910.05096](https://arxiv.org/abs/1910.05096) |
| Multi-pointer Co-Attention Networks for Recommendation | Y. Tay et al. | KDD | 2018 | [https://arxiv.org/abs/1804.09823](https://arxiv.org/abs/1804.09823) |
| Image-text retrieval with cross-attention | Various | CVPR / ECCV | 2018–2021 | [https://arxiv.org/abs/2004.08247](https://arxiv.org/abs/2004.08247) |
| Cross-attention for few-shot classification (cross-attn) | R. Hou et al. | NeurIPS | 2019 | [https://arxiv.org/abs/1910.07677](https://arxiv.org/abs/1910.07677) |
| Bottom-up features + co-attention for VQA improvements | Various | CVPR | 2018 | [https://arxiv.org/abs/1707.07998](https://arxiv.org/abs/1707.07998) |
| Meshed-memory & meshed decoder improvements | M. Cornia et al. | CVPR | 2020 | [https://arxiv.org/abs/1912.08226](https://arxiv.org/abs/1912.08226) |
| Attention for visual dialog & multi-turn QA | Various | ACL / EMNLP | 2018–2020 | [https://arxiv.org/abs/1902.00481](https://arxiv.org/abs/1902.00481) |
| Dual attention / co-attention networks for multi-modal fusion | Various | CVPR / ICCV | 2017–2020 | [https://arxiv.org/abs/1802.00923](https://arxiv.org/abs/1802.00923) |
| Audio-visual speech recognition (AVSR) with attention | T. Afouras et al. | TPAMI | 2018 | [https://arxiv.org/abs/1812.04965](https://arxiv.org/abs/1812.04965) |
| Attention for multi-modal retrieval (images & video) | Various | ACM MM | 2018–2021 | [https://arxiv.org/abs/1904.07860](https://arxiv.org/abs/1904.07860) |
| Multi-modal transformers (Video+Text) | Various | ACL / CVPR | 2020–2022 | [https://arxiv.org/abs/2103.15691](https://arxiv.org/abs/2103.15691) |
| Cross-modal attention for referring expression comprehension | Various | CVPR | 2019–2021 | [https://arxiv.org/abs/1906.01024](https://arxiv.org/abs/1906.01024) |
| Co-attention memory networks for healthcare diagnosis | J. Gao et al. | ICDM | 2019 | [https://arxiv.org/abs/1910.05096](https://arxiv.org/abs/1910.05096) |
| Co-attention for person re-identification (multi-modal inputs) | Various | ECCV / CVPR | 2018–2021 | [https://arxiv.org/abs/1904.11962](https://arxiv.org/abs/1904.11962) |
| Visual grounding & cross-attention approaches | Various | ICCV / CVPR | 2019 | [https://arxiv.org/abs/1906.01024](https://arxiv.org/abs/1906.01024) |
| Attention for image captioning (multi-modal attention stacks) | K. Xu et al. | ICML | 2015 | [https://arxiv.org/abs/1502.03044](https://arxiv.org/abs/1502.03044) |
| Cross-attention for speech+text fusion (ASR+LM) | Various | ICASSP / ACL | 2019–2021 | [https://arxiv.org/abs/1904.08783](https://arxiv.org/abs/1904.08783) |
| Multi-modal attention for robotics & perception | Various | ICRA / IROS | 2019–2021 | [https://arxiv.org/abs/1905.01745](https://arxiv.org/abs/1905.01745) |
| Cross-attention for multi-lingual vision-language tasks | Various | ACL / EMNLP | 2019–2021 | [https://arxiv.org/abs/1910.07123](https://arxiv.org/abs/1910.07123) |
| Co-attention & cross-modal retrieval benchmarks | Various | Datasets/Workshops | 2019–2021 | [https://arxiv.org/abs/2004.08247](https://arxiv.org/abs/2004.08247) |
| Attention-based multi-modal transformers for video understanding | Various | CVPR / ICCV | 2020–2022 | [https://arxiv.org/abs/2103.15691](https://arxiv.org/abs/2103.15691) |
| Multimodal pretraining with attention objectives | Various | NeurIPS / ICML | 2020–2022 | [https://arxiv.org/abs/2006.03272](https://arxiv.org/abs/2006.03272) |
| Cross-modal contrastive learning with attention | Various | NeurIPS / ICML | 2020–2022 | [https://arxiv.org/abs/2007.07045](https://arxiv.org/abs/2007.07045) |
| Query2Label & other label-attention classification works | S. Chen et al. | arXiv | 2021 | [https://arxiv.org/abs/2107.08046](https://arxiv.org/abs/2107.08046) |
| Misc multi-modal attention works & toolkits | Various | Various | 2015–2022 | [https://github.com/youngfish42/Awesome-Attention-Mechanism-in-CV](https://github.com/youngfish42/Awesome-Attention-Mechanism-in-CV) |

## Medical Imaging Attention

Attention applied to segmentation, detection, and diagnosis in medical imaging.

| Title | Authors | Venue | Year | Paper Link |
|-------|---------|-------|------|------------|
| Attention U-Net: Learning Where to Look for the Pancreas | O. Oktay et al. | MIDL | 2018 | [https://arxiv.org/abs/1804.03999](https://arxiv.org/abs/1804.03999) |
| TransUNet: Transformers make strong encoders for medical image segmentation | J. Chen et al. | arXiv | 2021 | [https://arxiv.org/abs/2102.04306](https://arxiv.org/abs/2102.04306) |
| CA-Net: Comprehensive Attention Convolutional Neural Networks | L. Chen et al. | TMI | 2021 | [https://ieeexplore.ieee.org/document/9381679](https://ieeexplore.ieee.org/document/9381679) |
| Residual Attention: multi-label recognition & med apps | F. Wang et al. | ICCV | 2021 | [https://arxiv.org/abs/1704.06904](https://arxiv.org/abs/1704.06904) |
| ASDNet / ASD approaches for med seg | D. Nie et al. | MICCAI | 2018 | [https://arxiv.org/abs/1806.00885](https://arxiv.org/abs/1806.00885) |
| Multi-scale self-guided attention for medical segmentation | A. Sinha & J. Dolz | JBHI | 2021 | [https://arxiv.org/abs/2011.03319](https://arxiv.org/abs/2011.03319) |
| SE + spatial gating in medical segmentation | Various | MICCAI / TMI | 2018–2020 | [https://arxiv.org/abs/1803.02579](https://arxiv.org/abs/1803.02579) |
| Attention-guided CNN for thorax disease classification | Q. Guan et al. | arXiv | 2019 | [https://arxiv.org/abs/1906.03901](https://arxiv.org/abs/1906.03901) |
| TransUNet + hybrid attention modules | J. Chen et al. | MICCAI / arXiv | 2021 | [https://arxiv.org/abs/2102.04306](https://arxiv.org/abs/2102.04306) |
| Attention in radiology report generation | B. Jing et al. | ACL | 2018 | [https://arxiv.org/abs/1804.04304](https://arxiv.org/abs/1804.04304) |
| Self-attention for medical image pretraining | Various | MICCAI / CoRR | 2020–2022 | [https://arxiv.org/abs/2006.03272](https://arxiv.org/abs/2006.03272) |
| Multi-level attention U-Nets for segmentation | Various | MIDL / MICCAI | 2018–2021 | [https://arxiv.org/abs/1804.03999](https://arxiv.org/abs/1804.03999) |
| Attention for disease localization (weak supervision) | Various | CVPR / MICCAI | 2019 | [https://arxiv.org/abs/1904.08725](https://arxiv.org/abs/1904.08725) |
| Attention-guided detection in chest X-rays | Q. Guan et al. | TMI / arXiv | 2019 | [https://arxiv.org/abs/1906.03901](https://arxiv.org/abs/1906.03901) |
| Explainable attention modules for clinical use | Various | TMI / JAMIA | 2020–2022 | [https://arxiv.org/abs/2005.05715](https://arxiv.org/abs/2005.05715) |
| TransUNet variants & hybrid attention designs | Various | arXiv / MICCAI | 2021–2022 | [https://arxiv.org/abs/2102.04306](https://arxiv.org/abs/2102.04306) |
| Attention-based segmentation for COVID-19 CT scans | X. Chen et al. | arXiv | 2020 | [https://arxiv.org/abs/2004.14133](https://arxiv.org/abs/2004.14133) |
| Attention modules for histopathology image analysis | Various | MICCAI / TMI | 2019–2021 | [https://arxiv.org/abs/1904.08725](https://arxiv.org/abs/1904.08725) |
| Channel+spatial attention for organ segmentation | Various | MIDL | 2018–2021 | [https://arxiv.org/abs/1804.03999](https://arxiv.org/abs/1804.03999) |
| Attention for multi-modal medical fusion (MRI+PET) | Various | MICCAI | 2019–2021 | [https://arxiv.org/abs/1906.10225](https://arxiv.org/abs/1906.10225) |
| Attention for medical report generation & alignment | B. Jing et al. | ACL | 2018 | [https://arxiv.org/abs/1804.04304](https://arxiv.org/abs/1804.04304) |
| Attention + transformers for 3D medical volumes | Various | MICCAI / CoRR | 2021 | [https://arxiv.org/abs/2102.04306](https://arxiv.org/abs/2102.04306) |
| Attention for disease progression modeling (time-series clinical) | Various | AAAI / NeurIPS workshops | 2019–2021 | [https://arxiv.org/abs/1712.00936](https://arxiv.org/abs/1712.00936) |
| Attention-guided lesion detection & segmentation | Various | CVPR / MICCAI | 2019–2021 | [https://arxiv.org/abs/1904.08725](https://arxiv.org/abs/1904.08725) |
| Attention for ultrasound image analysis | Various | IUS / MICCAI | 2019–2021 | [https://arxiv.org/abs/1904.08725](https://arxiv.org/abs/1904.08725) |
| Attention for cell & nuclei segmentation (NAS-SCAM etc.) | Various | MICCAI / ECCV | 2020 | [https://arxiv.org/abs/2006.06627](https://arxiv.org/abs/2006.06627) |
| Attention for MRI reconstruction tasks | Various | ISBI / MICCAI | 2019–2021 | [https://arxiv.org/abs/1904.02890](https://arxiv.org/abs/1904.02890) |
| Attention for medical anomaly detection & weakly-supervised learning | Various | MICCAI | 2020–2022 | [https://arxiv.org/abs/2006.05525](https://arxiv.org/abs/2006.05525) |
| Misc medical imaging attention survey & datasets | Various | Surveys / Repos | 2018–2022 | [https://github.com/youngfish42/Awesome-Attention-Mechanism-in-CV](https://github.com/youngfish42/Awesome-Attention-Mechanism-in-CV) |

## Surveys & Reviews

Survey and review papers summarizing the attention literature.

| Title | Authors | Venue | Year | Paper Link |
|-------|---------|-------|------|------------|
| An Attentive Survey of Attention Models | S. Chaudhari et al. | ACM TIST / CoRR | 2019–2021 | [https://arxiv.org/abs/1904.02874](https://arxiv.org/abs/1904.02874) |
| Survey on the attention based RNN model and its CV applications | F. Wang, D. M. J. Tax | arXiv | 2016 | [https://arxiv.org/abs/1601.06823](https://arxiv.org/abs/1601.06823) |
| Attention Mechanisms in Computer Vision: A Survey | M.-H. Guo et al. | Computational Visual Media | 2022 | [https://arxiv.org/abs/2111.07624](https://arxiv.org/abs/2111.07624) |
| Efficient Transformers: A Survey | Y. Tay et al. | arXiv | 2020 | [https://arxiv.org/abs/2009.06732](https://arxiv.org/abs/2009.06732) |
| Attention models in graphs: A survey | J. B. Lee et al. | ACM TKDD | 2019 | [https://arxiv.org/abs/1905.02817](https://arxiv.org/abs/1905.02817) |
| Introductory surveys on attention in NLP | D. Hu | IntelliSys Proceedings | 2019/2020 | [https://arxiv.org/abs/1904.02874](https://arxiv.org/abs/1904.02874) |
| Attention — critical review & analysis (NLP) | A. Galassi et al. | arXiv | 2019 | [https://arxiv.org/abs/1902.02181](https://arxiv.org/abs/1902.02181) |
| Human attention vs model attention (VQA) | A. Das et al. | CVIU | 2017 | [https://arxiv.org/abs/1610.04427](https://arxiv.org/abs/1610.04427) |
| Attention explainability debates (Is attention explanation?) | S. Jain & B. Wallace / S. Wiegreffe & Y. Pinter | NAACL / EMNLP | 2019 | [https://arxiv.org/abs/1902.10186](https://arxiv.org/abs/1902.10186) |
| Survey on attention in speech/audio | Various | Surveys / Interspeech | 2019–2020 | [https://arxiv.org/abs/2001.05574](https://arxiv.org/abs/2001.05574) |
| Surveys of attention in vision / transformers | Various | CVPR Workshops | 2020–2022 | [https://arxiv.org/abs/2111.07624](https://arxiv.org/abs/2111.07624) |
| Survey: Attention in medical imaging | Various | TMI / MICCAI Workshop | 2020–2022 | [https://arxiv.org/abs/2111.07624](https://arxiv.org/abs/2111.07624) |
| Benchmarks & evaluations of attention methods | Various | Reproducibility studies | 2019–2022 | [https://arxiv.org/abs/2001.05574](https://arxiv.org/abs/2001.05574) |
| Attention for fairness & interpretability (surveys) | Various | FAT* / ICLR workshops | 2020–2022 | [https://arxiv.org/abs/2005.05715](https://arxiv.org/abs/2005.05715) |
| Survey: Attention in recommender systems | Various | KDD / Surveys | 2019 | [https://arxiv.org/abs/1905.10702](https://arxiv.org/abs/1905.10702) |
| Survey: Efficient attention approximations | Various | arXiv | 2020–2022 | [https://arxiv.org/abs/2009.06732](https://arxiv.org/abs/2009.06732) |
| Meta-analyses comparing attention modules in CV | Various | Repositories / Workshops | 2020 | [https://github.com/youngfish42/Awesome-Attention-Mechanism-in-CV](https://github.com/youngfish42/Awesome-Attention-Mechanism-in-CV) |
| Tutorials on attention & transformers | Various | NeurIPS / ICML tutorials | 2018–2022 | [https://arxiv.org/abs/2009.06732](https://arxiv.org/abs/2009.06732) |
| Survey on attention for graph neural networks | J. B. Lee et al. | TKDD | 2019 | [https://arxiv.org/abs/1905.02817](https://arxiv.org/abs/1905.02817) |
| Survey on self-attention in generative models | Various | ICML / NeurIPS | 2019–2021 | [https://arxiv.org/abs/1905.08318](https://arxiv.org/abs/1905.08318) |
| Review of attention metrics & evaluation protocols | Various | Workshops | 2020 | [https://arxiv.org/abs/2001.05574](https://arxiv.org/abs/2001.05574) |
| Survey on multi-modal attention methods | Various | ACM MM / Surveys | 2019–2021 | [https://arxiv.org/abs/2004.08247](https://arxiv.org/abs/2004.08247) |
| Surveys on attention in reinforcement learning | Various | RL workshops | 2019–2021 | [https://arxiv.org/abs/1905.01745](https://arxiv.org/abs/1905.01745) |
| Survey on attention in time-series modeling | Various | Time-series workshops | 2020 | [https://arxiv.org/abs/2006.04387](https://arxiv.org/abs/2006.04387) |
| Surveys on attention-based architecture search (NAS) | Various | ICCV / CoRR | 2020 | [https://arxiv.org/abs/2006.06627](https://arxiv.org/abs/2006.06627) |
| Historical perspectives on attention (foundational works) | Various | Retrospectives | 2015–2020 | [https://arxiv.org/abs/1904.02874](https://arxiv.org/abs/1904.02874) |
| Survey resources & curated lists (repos) | Various | GitHub / ArXiv | 2019–2022 | [https://github.com/youngfish42/Awesome-Attention-Mechanism-in-CV](https://github.com/youngfish42/Awesome-Attention-Mechanism-in-CV) |
| Comparative survey: attention vs conv paradigms | Various | CoRR | 2022 | [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545) |
| Misc survey & tutorial resources | Various | Various | 2016–2022 | [https://github.com/youngfish42/Awesome-Attention-Mechanism-in-CV](https://github.com/youngfish42/Awesome-Attention-Mechanism-in-CV) |

## Miscellaneous / Domain-specific Attention

Papers that do not fit neatly into the above categories or are domain-specific (recommendation, finance, robotics, dynamic convolution, plug-and-play modules).

| Title | Authors | Venue | Year | Paper Link |
|-------|---------|-------|------|------------|
| Dynamic Convolution: Attention over Convolution Kernels | Y. Chen et al. | CVPR | 2020 | [https://arxiv.org/abs/1912.02765](https://arxiv.org/abs/1912.02765) |
| CondConv: Conditionally Parameterized Convolutions | B. Yang et al. | NeurIPS | 2019 | [https://arxiv.org/abs/1904.04971](https://arxiv.org/abs/1904.04971) |
| PP-NAS: Searching for Plug-and-Play Blocks on CNNs | Y. Chen et al. | ICCV Workshop | 2021 | [https://arxiv.org/abs/2104.12650](https://arxiv.org/abs/2104.12650) |
| Attention-like Structural Re-parameterization (ASR) | M. Zhang et al. | CoRR | 2023 | [https://arxiv.org/abs/2303.01043](https://arxiv.org/abs/2303.01043) |
| Temporal attention-augmented bilinear network (finance) | D. T. Tran et al. | TNNLS | 2019 | [https://arxiv.org/abs/1712.03277](https://arxiv.org/abs/1712.03277) |
| Attend and Diagnose: Clinical time series analysis using attention models | H. Song et al. | AAAI | 2018 | [https://arxiv.org/abs/1712.00936](https://arxiv.org/abs/1712.00936) |
| Multi-agent game abstraction via graph attention | Y. Liu et al. | AAAI | 2020 | [https://arxiv.org/abs/1906.02664](https://arxiv.org/abs/1906.02664) |
| Attention models in recommender systems: Multi-pointer co-attention | Y. Tay et al. | KDD | 2018 | [https
