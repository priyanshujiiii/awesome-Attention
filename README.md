# Awesome Attention [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of research papers, surveys, and resources on **Attention Mechanisms** across different domains — Computer Vision, NLP, Speech, Graphs, Multi-modal Learning, Medical Imaging, and more.  
This repository organizes papers into categories, each with a structured table:

**Title | Authors | Venue | Year | Paper Link**

---

## 📑 Table of Contents
1. [Channel Attention](channel_attention.md)
2. [Spatial Attention](spatial_attention.md)
3. [Temporal Attention](temporal_attention.md)
4. [Channel + Spatial Attention](channel_spatial_attention.md)
5. [Transformer-based Attention](transformer_attention.md)
6. [Graph Attention](graph_attention.md)
7. [Speech & Audio Attention](speech_audio_attention.md)
8. [Multi-modal Attention](multi_modal_attention.md)
9. [Medical Imaging Attention](medical_attention.md)
10. [Surveys & Reviews](surveys.md)
11. [Miscellaneous / Domain-specific](misc_attention.md)

---

# Channel Attention — Awesome Attention

Papers that focus on channel-wise recalibration, feature re-weighting and channel attention modules.

> Table columns: **Title | Authors | Venue | Year | Paper Link**

| Title | Authors | Venue | Year | Paper Link |
|---|---|---:|---:|---|
| Squeeze-and-Excitation Networks | J. Hu, L. Shen, G. Sun | CVPR | 2018 | []() |
| RCAN: Image Super-Resolution Using Very Deep Residual Channel Attention Networks | Y. Zhang et al. | ECCV | 2018 | []() |
| ECA-Net: Efficient Channel Attention | Q. Wang et al. | CVPR | 2020 | []() |
| GSoPNet: Global Second-order Pooling Conv Nets | Y. Li et al. | CVPR | 2019 | []() |
| SRM: Style-based Recalibration Module | J. Rony et al. | ICCV | 2019 | []() |
| DIANet: Dense-and-Implicit Attention Network | M. India et al. | AAAI | 2020 | []() |
| Competitive-SENet | — | CoRR | 2018 | []() |
| FcaNet: Frequency Channel Attention Networks | Z. Li et al. | ICCV | 2021 | []() |
| ResNeSt: Split-Attention Networks | H. Zhang et al. | CoRR | 2020 | []() |
| SGE: Spatial Group-wise Enhance (channel-centric variant) | W. Hu et al. | arXiv | 2019 | []() |
| SCSE: Concurrent Spatial and Channel ‘Squeeze & Excitation’ | — | MICCAI | 2018 | []() |
| SENet (PAMI version) | J. Hu, L. Shen, G. Sun | PAMI | 2019 | []() |
| Gated Channel Transformation for Visual Recognition | — | CVPR | 2020 | []() |
| Channel-wise Attention for Image Restoration (RNAN-style) | — | ICLR / arXiv | 2019 | []() |
| Tiled Squeeze-and-Excite (local channel attention) | — | ICCV Workshop | 2021 | []() |
| SRM (alternate) applications | — | ICML/ICCV workshops | 2019 | []() |
| Competitive Inner-Imaging Squeeze & Excitation | — | CoRR | 2018 | []() |
| CA: Channel Attention blocks in segmentation networks | — | ECCV | 2018–2020 | []() |
| Channel Dropout / Weighted Channel Dropout papers | — | AAAI | 2019 | []() |
| ULSAM: Ultra-Lightweight Subspace Attention Module | — | WACV | 2020 | []() |
| Tiled SE & Local-SE variants | — | ICCV Workshops | 2021 | []() |
| Multi-scale channel attention (MSCAF) | — | CVPR Workshops | 2020 | []() |
| Channel attention in GANs (Self-attention GAN variants) | H. Zhang et al. | ICML | 2019 | []() |
| Channel attention for point-cloud networks | — | CVPR | 2019 | []() |
| Channel-wise attention for medical image segmentation (CA-Net) | L. et al. | TMI | 2021 | []() |
| Channel attention + frequency (FcaNet variants) | — | ICCV | 2021 | []() |
| Channel-aware dynamic convolutions | — | ECCV | 2020 | []() |
| CE-Net / Channel-enhanced modules | — | MICCAI / TMI | 2018–2021 | []() |
| Plug-and-play channel modules summary | — | Repo / Survey | 2020 | []() |
| Misc. channel-attention applications (detection/pose) | — | Various | 2018–2022 | []() |

---
# Spatial Attention — Awesome Attention

Papers that learn to attend over spatial regions (where to look).

| Title | Authors | Venue | Year | Paper Link |
|---|---|---:|---:|---|
| Recurrent Models of Visual Attention | V. Mnih, N. Heess, A. Graves, K. Kavukcuoglu | NeurIPS | 2014 | []() |
| Show, Attend and Tell: Neural Image Caption Generation with Visual Attention | K. Xu et al. | ICML | 2015 | []() |
| Spatial Transformer Networks | M. Jaderberg et al. | NeurIPS | 2015 | []() |
| DRAW: A Recurrent Neural Network for Image Generation | K. Gregor et al. | ICML | 2015 | []() |
| Attention U-Net: Learning Where to Look for the Pancreas | O. Oktay et al. | MIDL | 2018 | []() |
| Non-local Neural Networks | X. Wang et al. | CVPR | 2018 | []() |
| Attention Augmented Convolutional Networks | A. Parmar et al. | ICCV | 2019 | []() |
| Psanet: Point-wise Spatial Attention Network for Scene Parsing | — | ECCV | 2018 | []() |
| A2-Nets: Double Attention Networks | Z. et al. | NeurIPS | 2018 | []() |
| Look Closer to See Better (RA-CNN) | S. Fu et al. | CVPR | 2017 | []() |
| Attentional Pooling for Action Recognition | R. et al. | NeurIPS | 2017 | []() |
| Visual Attention for Fine-grained Recognition (various) | — | ICCV/CVPR | 2016–2019 | []() |
| Attention in Image Captioning (various improvements) | — | ICCV/CVPR | 2015–2020 | []() |
| Attention-Aware Compositional Networks for Re-ID | — | CVPR | 2018 | []() |
| Tell Me Where to Look: Guided Attention Inference Network | — | CVPR | 2018 | []() |
| Attentional ShapeContextNet for Point Cloud Recognition | — | CVPR | 2018 | []() |
| Attentional PointNet for 3D detection | — | CVPRW | 2019 | []() |
| Human attention in VQA studies (human vs model) | A. Das et al. | CVIU | 2017 | []() |
| Supervising Attention with Human Gaze for Video Captioning | Y. Yu et al. | CVPR | 2017 | []() |
| Non-local operations & variants | — | CVPR / ECCV | 2018–2020 | []() |
| Attention Correctness in Image Captioning | C. Liu et al. | AAAI | 2017 | []() |
| Guided Attention for Detection/Counting (GANet) | — | ACM MM | 2020 | []() |
| Attention for Video Summarization | J. Fajtl et al. | ACCV | 2018 | []() |
| Local Relation / Relation Networks for Recognition | — | ICCV/CVPR | 2019 | []() |
| Second-order attention models for VQA | — | NeurIPS | 2017 | []() |
| Attention for Person Re-ID (diverse) | — | CVPR/ICCV | 2017–2020 | []() |
| Attention-guided convolution for thorax disease classification | — | arXiv | 2019 | []() |
| Self-Attention GANs (spatial attention in GANs) | H. Zhang et al. | ICML | 2019 | []() |
| Spatial attention in point-cloud & 3D tasks | — | ICCV / CVPR | 2019–2021 | []() |
| Misc spatial-attention improvements and surveys | — | Various | 2015–2022 | []() |
---

# Temporal Attention — Awesome Attention

Papers focused on attention in time series, video, and other temporal sequences.

| Title | Authors | Venue | Year | Paper Link |
|---|---|---:|---:|---|
| Jointly Attentive Spatial-Temporal Pooling Networks | — | ICCV | 2017 | []() |
| VideoLSTM: Convolves, Attends and Flows for Action Recognition | — | arXiv | 2016 | []() |
| Temporal Attention for Action Recognition (various) | — | CVPR/ICCV | 2016–2020 | []() |
| Hierarchical LSTMs with Adaptive Attention for Visual Captioning | — | TPAMI | 2020 | []() |
| Space-time Mixing Attention for Video Transformer | — | CoRR | 2021 | []() |
| Temporal attention-augmented bilinear network (finance) | D. T. Tran et al. | TNNLS | 2019 | []() |
| Video summary via attention (Summarizing videos with attention) | J. Fajtl et al. | ACCV | 2018 | []() |
| Temporal Self-Attention in Transformers for Video | — | CVPR/ICCV | 2020–2022 | []() |
| Multi-scale temporal attention for action detection | — | ECCV / CVPR | 2018–2021 | []() |
| Temporal co-attention for video QA | — | NeurIPS / ICCV | 2018–2020 | []() |
| Temporal attention in speech + audio (ASR integration) | — | ICASSP | 2015–2019 | []() |
| Video person re-identification with temporal attention | — | CVPR | 2018 | []() |
| Temporal attention for multi-object tracking | — | CVPR | 2019 | []() |
| Multi-hop temporal attention for reasoning across time | — | ACL / NAACL | 2018–2020 | []() |
| Transformer-based temporal models (TimeSformer, etc.) | — | ICCV/CVPR | 2021 | []() |
| Temporal attention for ECG / clinical time series (Attend & Diagnose) | H. Song et al. | AAAI | 2018 | []() |
| Temporal attention for video person reid (snippet aggregation) | — | CVPR | 2018 | []() |
| Video captioning with temporal attention | L. Gao et al. | TMM | 2017 | []() |
| Temporal attention for video segmentation | — | CVPR | 2020 | []() |
| Temporal attention networks for audio tagging | Y. Xu et al. | Interspeech | 2017 | []() |
| Spatio-temporal attention for re-id and tracking | — | ICCV / ECCV | 2017–2021 | []() |
| Temporal attention for anomaly detection in videos | — | CVPR Workshops | 2019–2021 | []() |
| Temporal attention for multimodal fusion (AVSR) | T. Afouras et al. | TPAMI | 2018 | []() |
| Temporal attention for forecasting (finance & sensors) | — | TNNLS / ICML workshops | 2019–2021 | []() |
| Temporal attention in video transformers (TimeSformer variants) | — | ICCV / CVPR | 2021 | []() |
| RNN + attention for sequence-to-sequence tasks (classic) | D. Bahdanau et al. | ICLR | 2015 | []() |
| Multi-head temporal attention adaptations | — | NeurIPS / ICLR | 2018–2021 | []() |
| Temporal attention for audio-visual event localization | — | ACM MM | 2019 | []() |
| Misc temporal-attention literature & benchmarks | — | Various | 2016–2022 | []() |

---

# Channel + Spatial Attention — Awesome Attention

Hybrid modules that combine both channel and spatial attention.

| Title | Authors | Venue | Year | Paper Link |
|---|---|---:|---:|---|
| CBAM: Convolutional Block Attention Module | S. Woo et al. | ECCV | 2018 | []() |
| BAM: Bottleneck Attention Module | J. Park et al. | BMVC | 2018 | []() |
| Residual Attention Network for Image Classification | F. Wang et al. | CVPR | 2017 | []() |
| SCA-CNN: Spatial and Channel-wise attention | — | CVPR | 2017 | []() |
| DANet: Dual Attention Network for Scene Segmentation | J. Fu et al. | CVPR | 2019 | []() |
| Coordinate Attention | Q. Hou et al. | CVPR | 2021 | []() |
| scSE: Concurrent Spatial & Channel SE (medical) | — | MICCAI | 2018 | []() |
| Triplet Attention (convolutional triplet) | — | WACV | 2021 | []() |
| AFF: Attentional Feature Fusion | — | WACV | 2021 | []() |
| PSANet: Point-wise Spatial Attention Network (channel+spatial variants) | — | ECCV | 2018 | []() |
| Residual attention U-Nets (medical) | — | MICCAI / MIDL | 2018–2021 | []() |
| Recalibrating FCNs with scSE blocks | — | TMI | 2018 | []() |
| Attention U-Net (channel + spatial gating) | O. Oktay et al. | MIDL | 2018 | []() |
| CBAM variants & improvements | — | CVPR Workshops | 2019–2021 | []() |
| Coordinate Attention for Mobile Networks | — | CVPR | 2021 | []() |
| NAM: Normalization-based Attention Module | — | CoRR | 2021 | []() |
| EPSANet: Efficient Pyramid Split Attention Block | — | CoRR | 2021 | []() |
| SimAM: Parameter-free attention for CNNs | — | ICML | 2021 | []() |
| SIAM / SPA modules combining channel+spatial | — | ICCV / ECCV | 2019–2021 | []() |
| STN+SE hybrid modules | — | CVPR Workshops | 2019 | []() |
| Res2Net with attention (channel+spatial injections) | — | CVPR | 2019 | []() |
| Cross-channel communication networks | — | NeurIPS | 2019 | []() |
| CCNet: Criss-Cross Attention (spatial with reweighting) | — | ICCV | 2019 | []() |
| A2-Nets: Double attention (channel+spatial interaction) | — | NeurIPS | 2018 | []() |
| HAttMatting: Attention for matting (channel+spatial) | — | CVPR | 2020 | []() |
| AW-Conv: Attention as convolutional activation | — | ICCV | 2021 | []() |
| CA-Net: Comprehensive Attention for Explainable Med Seg | L. et al. | TMI | 2021 | []() |
| SRM + spatial fusion modules | — | ICCV | 2019–2020 | []() |
| Plug-and-play joint attention blocks (PP-NAS etc.) | — | ICCV Workshops | 2021 | []() |
| Misc channel+spatial hybrid works | — | Various | 2017–2022 | []() |

---

# Transformer-based Attention — Awesome Attention

Self-attention and transformer architectures for NLP, vision, and multi-modal tasks.

| Title | Authors | Venue | Year | Paper Link |
|---|---|---:|---:|---|
| Attention Is All You Need | A. Vaswani et al. | NeurIPS | 2017 | []() |
| Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context | Z. Dai et al. | ACL | 2019 | []() |
| Reformer: The Efficient Transformer | N. Kitaev et al. | ICLR | 2020 | []() |
| Linformer: Self-attention with Linear Complexity | S. Wang et al. | arXiv | 2020 | []() |
| Longformer / BigBird (sparse attention) | I. Zaheer et al. / A. Joshi et al. | ACL / NeurIPS | 2020 | []() |
| ViT: An Image is Worth 16x16 Words | A. Dosovitskiy et al. | ICLR | 2021 | []() |
| Swin Transformer: Hierarchical Vision Transformer using Shifted Windows | Z. Liu et al. | ICCV | 2021 | []() |
| DeiT: Data-efficient Image Transformers | Touvron et al. | ICML | 2021 | []() |
| CoAtNet: Marrying Convolution and Attention | — | CoRR | 2021 | []() |
| CaiT / CPVT / ConViT variants | — | ICCV / CoRR | 2021 | []() |
| MaxViT: Multi-Axis Vision Transformer | — | CoRR | 2022 | []() |
| Reformer / Performer / Linformer family | Various | ICLR / NeurIPS | 2020–2022 | []() |
| DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification | — | NeurIPS | 2021 | []() |
| DVT: Dynamic Transformers for Efficient Image Recognition | — | NeurIPS | 2021 | []() |
| LocalViT / MobileViT / LeViT (efficient ViTs) | — | CoRR | 2021 | []() |
| BEiT / MAE / Self-supervised ViT pretraining | — | CVPR / ICLR | 2021–2022 | []() |
| DeiT distillation & data-efficient training | Touvron et al. | ICML | 2021 | []() |
| Vision Transformer with Deformable Attention (DAT) | — | CoRR | 2022 | []() |
| ConvNeXt / ConvNeXt-V2 discussions on conv vs attention | — | CoRR | 2022–2023 | []() |
| VOLO / VOLO variants (vision outlookers) | — | CoRR | 2021 | []() |
| Transformer in Transformer / TNT | — | arXiv | 2021 | []() |
| Query2Label / Masked-attention classification variants | — | arXiv | 2021 | []() |
| Synthesizer: Rethinking Self-Attention | Y. Tay et al. | ICML | 2021 | []() |
| Efficient Transformers: A Survey | Y. Tay et al. | arXiv | 2020 | []() |
| Reformer / Performer / Linformer comparisons | — | Surveys | 2020–2022 | []() |
| Dynamic token pruning & token sparsification works | — | NeurIPS / ICLR | 2021–2022 | []() |
| IO: Image-specific transformer optimizations (various) | — | CVPR/ICCV | 2021–2023 | []() |
| SEG: Segmentation transformer models (SegFormer, etc.) | — | ECCV / CVPR | 2021 | []() |
| SAM: Segment Anything (foundation vision model using attention) | Meta AI | CoRR | 2023 | []() |

---

# Graph Attention — Awesome Attention

Attention mechanisms applied to graph-structured data (GNNs).

| Title | Authors | Venue | Year | Paper Link |
|---|---|---:|---:|---|
| Graph Attention Networks (GAT) | P. Veličković et al. | ICLR | 2018 | []() |
| GATv2: Adaptive Graph Attention | — | arXiv | 2021 | []() |
| Graph Attention Network variants & surveys | — | TKDD / CoRR | 2019–2021 | []() |
| LatentGNN: Learning efficient non-local relations | S. Zhang et al. | ICML | 2019 | []() |
| Graph-based global reasoning networks | — | CVPR | 2019 | []() |
| Factor Graph Attention | — | CVPR | 2019 | []() |
| Graph Transformer Network (GTN) family | — | NeurIPS / ICLR | 2020–2022 | []() |
| Dynamic Graph Attention (Dysat) | A. Sankar et al. | WSDM | 2020 | []() |
| Attention models in graphs: A survey | J. B. Lee et al. | TKDD | 2019 | []() |
| Graph attention for recommender systems (GAT-based) | — | KDD | 2018–2020 | []() |
| Heterogeneous graph attention networks (HAN) | — | WWW / KDD | 2019 | []() |
| Multi-head graph attention & normalization techniques | — | ICLR | 2019–2021 | []() |
| Graph attention for molecular property prediction | — | NeurIPS / ICML | 2018–2021 | []() |
| Attention over edges & edge-aware GAT variations | — | ICLR / ICML | 2019–2022 | []() |
| Graph attention for dynamic graphs & streaming | — | WSDM / KDD | 2020–2022 | []() |
| Inductive graph attention approaches (GraphSAGE+attn) | — | NeurIPS | 2017–2019 | []() |
| Graph attention for protein folding / bio tasks | — | Bioinformatics venues | 2019–2022 | []() |
| Graph co-attention for multi-graph reasoning | — | ACL / EMNLP | 2019–2021 | []() |
| Graph attention for knowledge graphs (KGAT-like) | — | WWW / KDD | 2019 | []() |
| Scalable graph attention for large graphs (sampling) | — | KDD / WSDM | 2020 | []() |
| Graph attention in traffic forecasting & spatio-temporal GNNs | — | AAAI / NeurIPS | 2019–2021 | []() |
| Graph attention for 3D point clouds & meshes | — | ICCV / CVPR | 2019–2021 | []() |
| Graph attention & fairness/explainability works | — | FAT* / ICLR | 2020–2022 | []() |
| Graph attention for social network analysis & fake news detection | Y.-J. Lu et al. | ACL | 2020 | []() |
| Graph attention + transformers (Graphormer etc.) | — | NeurIPS | 2021 | []() |
| Attention pooling & readout mechanisms for graphs | — | ICML / ICLR | 2018–2021 | []() |
| Benchmarking graph attention models (surveys) | — | Surveys | 2020–2022 | []() |
| Graph attention for recommender systems (multi-pointer co-attention) | Y. Tay et al. | KDD | 2018 | []() |
| Misc graph-attention advances & code repos | — | Various | 2018–2022 | []() |

---

# Speech & Audio Attention — Awesome Attention

Attention in ASR, audio tagging, speaker recognition, and audio-visual speech.

| Title | Authors | Venue | Year | Paper Link |
|---|---|---:|---:|---|
| Neural Machine Translation by Jointly Learning to Align and Translate | D. Bahdanau et al. | ICLR | 2015 | []() |
| Attention-based Models for Speech Recognition | J. Chorowski et al. | NeurIPS | 2015 | []() |
| End-to-end attention-based large vocabulary speech recognition | D. Bahdanau et al. | ICASSP | 2016 | []() |
| Joint CTC-Attention based End-to-end Speech Recognition | S. Kim et al. | ICASSP | 2017 | []() |
| Very Deep Self-Attention Networks for End-to-End Speech Recognition | N.-Q. Pham et al. | arXiv | 2019 | []() |
| Deep Audio-Visual Speech Recognition | T. Afouras et al. | TPAMI / 2018 | 2018 | []() |
| Self-attention networks for connectionist temporal classification | J. Salazar et al. | ICASSP | 2019 | []() |
| Multi-level attention model for weakly supervised audio classification | C. Yu et al. | DCASE Workshop | 2018 | []() |
| Attention and localization for audio tagging | Y. Xu et al. | Interspeech | 2017 | []() |
| Self multi-head attention for speaker recognition | M. India et al. | Interspeech | 2019 | []() |
| Attention for speech emotion recognition (multi-hop) | S. Yoon et al. | ICASSP | 2019 | []() |
| Transformer-based ASR models (Speech-Transformer) | L. Dong et al. | ICASSP | 2018 | []() |
| Self-attention for long-range speech modeling (Transformer-XL variants) | — | ICASSP / Interspeech | 2019–2021 | []() |
| Auditory scene analysis with attention models | — | DCASE / ICASSP | 2018–2020 | []() |
| Attention in speaker diarization & separation | — | ICASSP / Interspeech | 2019–2021 | []() |
| Multi-modal attention for AVSR & lipreading | — | ICASSP / TPAMI | 2018–2021 | []() |
| Attention for keyword spotting & wake-word detection | — | Interspeech | 2019 | []() |
| Attention in speech enhancement & denoising | — | ICASSP | 2019–2022 | []() |
| Attention-based audio retrieval & tagging | — | ACM MM | 2018–2019 | []() |
| Attention-based music recommendation & tagging | — | ISMIR | 2018–2020 | []() |
| Attention for low-resource ASR & transfer learning | — | Interspeech | 2019 | []() |
| Attention + CTC hybrids & multi-task models | — | ICASSP | 2017–2020 | []() |
| Attention in speech synthesis (TTS) | — | ICASSP / NeurIPS | 2018–2021 | []() |
| Attention for phoneme recognition & alignment | P. Schwarz et al. | TSD | 2004 | []() |
| Attention for speaker verification & spoofing detection | — | Interspeech | 2019–2021 | []() |
| Attention-based ASR benchmarks & surveys | — | Surveys | 2018–2021 | []() |
| Attention and localization based on deep convolutional recurrent models | Y. Xu et al. | Interspeech | 2017 | []() |
| Self-supervised attention pretraining for audio | — | ICML / NeurIPS | 2020–2022 | []() |
| Misc speech/audio attention works & toolkits | — | Various | 2015–2022 | []() |

---

# Multi-modal Attention — Awesome Attention

Cross-modal and co-attention for tasks combining vision + language / audio + vision.

| Title | Authors | Venue | Year | Paper Link |
|---|---|---:|---:|---|
| Bottom-Up and Top-Down Attention for Image Captioning & VQA | P. Anderson et al. | CVPR | 2018 | []() |
| Hierarchical Question-Image Co-Attention for VQA | J. Lu et al. | NeurIPS | 2016 | []() |
| Meshed-Memory Transformer for Image Captioning | M. Cornia et al. | CVPR | 2020 | []() |
| VisualBERT / LXMERT / ViLBERT (vision-language transformers) | — | ACL / NeurIPS | 2019–2020 | []() |
| Co-attention Memory Networks for Diagnosis Prediction | J. Gao et al. | ICDM | 2019 | []() |
| Multi-pointer Co-Attention Networks for Recommendation | Y. Tay et al. | KDD | 2018 | []() |
| Image-text retrieval with cross-attention | — | CVPR / ECCV | 2018–2021 | []() |
| Cross-attention for few-shot classification (cross-attn) | R. Hou et al. | NeurIPS | 2019 | []() |
| Bottom-up features + co-attention for VQA improvements | — | CVPR | 2018 | []() |
| Meshed-memory & meshed decoder improvements | — | CVPR | 2020 | []() |
| Attention for visual dialog & multi-turn QA | — | ACL / EMNLP | 2018–2020 | []() |
| Dual attention / co-attention networks for multi-modal fusion | — | CVPR / ICCV | 2017–2020 | []() |
| Audio-visual speech recognition (AVSR) with attention | T. Afouras et al. | TPAMI | 2018 | []() |
| Attention for multi-modal retrieval (images & video) | — | ACM MM | 2018–2021 | []() |
| Multi-modal transformers (Video+Text) | — | ACL / CVPR | 2020–2022 | []() |
| Cross-modal attention for referring expression comprehension | — | CVPR | 2019–2021 | []() |
| Co-attention memory networks for healthcare diagnosis | J. Gao et al. | ICDM | 2019 | []() |
| Co-attention for person re-identification (multi-modal inputs) | — | ECCV / CVPR | 2018–2021 | []() |
| Visual grounding & cross-attention approaches | — | ICCV / CVPR | 2019 | []() |
| Attention for image captioning (multi-modal attention stacks) | K. Xu et al. | ICML | 2015 | []() |
| Cross-attention for speech+text fusion (ASR+LM) | — | ICASSP / ACL | 2019–2021 | []() |
| Multi-modal attention for robotics & perception | — | ICRA / IROS | 2019–2021 | []() |
| Cross-attention for multi-lingual vision-language tasks | — | ACL / EMNLP | 2019–2021 | []() |
| Co-attention & cross-modal retrieval benchmarks | — | Datasets/Workshops | 2019–2021 | []() |
| Attention-based multi-modal transformers for video understanding | — | CVPR / ICCV | 2020–2022 | []() |
| Multimodal pretraining with attention objectives | — | NeurIPS / ICML | 2020–2022 | []() |
| Cross-modal contrastive learning with attention | — | NeurIPS / ICML | 2020–2022 | []() |
| Query2Label & other label-attention classification works | — | arXiv | 2021 | []() |
| Misc multi-modal attention works & toolkits | — | Various | 2015–2022 | []() |

---









