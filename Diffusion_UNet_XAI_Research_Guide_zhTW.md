# Diffusion U-Net 內部表徵與 XAI：台灣降雨降尺度研究筆記與實作規劃

> 本文件整理「如何針對 Diffusion Model 的 U-Net 內部特徵做可解釋性分析（XAI / Mechanistic Interpretability）」的完整研究脈絡，並直接對應目前的台灣降雨 conditional diffusion 專案。
>
> 目標不是只在模型輸入端做 SHAP、Permutation Importance 或 Grad-CAM，而是進一步研究：
>
> **Diffusion U-Net 的 Encoder、Middle/Bottleneck、Decoder、Attention、個別 channels，以及不同 denoising timesteps 之間，到底形成了什麼可解釋的內部表徵？**

---

## 目錄

1. [研究動機](#1-研究動機)
2. [目前專案與 XAI 的切入點](#2-目前專案與-xai-的切入點)
3. [核心研究問題](#3-核心研究問題)
4. [最重要的相關論文總覽](#4-最重要的相關論文總覽)
5. [論文一：Elucidating the Representation of Images Within an Unconditional Diffusion Model Denoiser](#5-論文一elucidating-the-representation-of-images-within-an-unconditional-diffusion-model-denoiser)
6. [論文二：Not All Diffusion Model Activations Have Been Evaluated as Discriminative Features](#6-論文二not-all-diffusion-model-activations-have-been-evaluated-as-discriminative-features)
7. [論文三：Revelio](#7-論文三revelio)
8. [論文四：Diffusion Hyperfeatures](#8-論文四diffusion-hyperfeatures)
9. [論文五：Emergent Correspondence from Image Diffusion / DIFT](#9-論文五emergent-correspondence-from-image-diffusion--dift)
10. [論文六：Plug-and-Play Diffusion Features](#10-論文六plug-and-play-diffusion-features)
11. [論文七：DAAM](#11-論文七daam)
12. [論文八：Mechanistic Interpretability via Cross-Attention Interventions](#12-論文八mechanistic-interpretability-via-cross-attention-interventions)
13. [論文九：Unveiling Concept Attribution in Diffusion Models](#13-論文九unveiling-concept-attribution-in-diffusion-models)
14. [Weather XAI：Mechanistic Interpretability Tool for AI Weather Models](#14-weather-xaimechanistic-interpretability-tool-for-ai-weather-models)
15. [Weather XAI：Towards Mechanistic Understanding in a Data-Driven Weather Model](#15-weather-xaitowards-mechanistic-understanding-in-a-data-driven-weather-model)
16. [降水 Diffusion：How Far Can We Downscale?](#16-降水-diffusionhow-far-can-we-downscale)
17. [如何把這些文獻接到我的專案](#17-如何把這些文獻接到我的專案)
18. [第一版 XAI 方法：Activation Hook](#18-第一版-xai-方法activation-hook)
19. [Layer-wise 分析](#19-layer-wise-分析)
20. [Channel-wise 分析](#20-channel-wise-分析)
21. [PCA / Cosine Similarity / Clustering](#21-pca--cosine-similarity--clustering)
22. [Diffusion Timestep 分析](#22-diffusion-timestep-分析)
23. [Spatial Feature Map 分析](#23-spatial-feature-map-分析)
24. [Attention 分析](#24-attention-分析)
25. [Feature Probing](#25-feature-probing)
26. [Sparse Autoencoder](#26-sparse-autoencoder)
27. [Causal Intervention / Ablation](#27-causal-intervention--ablation)
28. [固定 Seed 為什麼非常重要](#28-固定-seed-為什麼非常重要)
29. [完整研究 Pipeline](#29-完整研究-pipeline)
30. [建議實驗設計](#30-建議實驗設計)
31. [建議圖表](#31-建議圖表)
32. [研究貢獻可以怎麼寫](#32-研究貢獻可以怎麼寫)
33. [目前方法的限制與不能過度解釋的地方](#33-目前方法的限制與不能過度解釋的地方)
34. [建議實作資料夾](#34-建議實作資料夾)
35. [最小可行版本（MVP）](#35-最小可行版本mvp)
36. [進階版本](#36-進階版本)
37. [論文 Related Work 可以怎麼組織](#37-論文-related-work-可以怎麼組織)
38. [參考文獻](#38-參考文獻)

---

# 1. 研究動機

Diffusion Model 雖然可以產生品質很好的高解析度輸出，但其生成過程通常被視為 black box。

對目前的 precipitation downscaling 專案來說，我們已經知道模型的外部流程：

```text
14×9 coarse atmospheric variables
        ↓
resize / normalize
        ↓
conditional diffusion
        ↓
EDMPrecond
        ↓
U-Net
        ↓
iterative denoising
        ↓
high-resolution precipitation residual
        ↓
reconstruct precipitation
```

但我們仍然不知道：

```text
U-Net Encoder 到底學到了什麼？
Middle/Bottleneck 到底儲存了什麼？
Decoder 是什麼時候開始形成局部 precipitation correction？
不同 diffusion timestep 裡，模型看的「特徵」是否不同？
某些 latent channels 是否和特定大氣條件或降雨型態有穩定關係？
```

因此本研究真正想回答的是：

> **Conditional diffusion U-Net 在 precipitation downscaling 過程中，內部到底如何組織、保存、轉換與使用氣象資訊？**

這比「哪個 input variable 重要」更深入。

---

# 2. 目前專案與 XAI 的切入點

目前專案核心可概念化成：

```text
Dataset
│
├─ q700
├─ t2m
├─ u
├─ v
├─ msl
├─ tp
└─ land mask
     ↓
[B, 7, 112, 72]

Diffusion noisy residual
[B, 1, 112, 72]
     ↓
concat
     ↓
[B, 8, 112, 72]

EDMPrecond
     ↓
UNet
     │
     ├─ Encoder
     │   128 × 112 × 72
     │   256 × 56 × 36
     │   384 × 28 × 18
     │   512 × 14 × 9
     │
     ├─ Bottleneck
     │   512 × 14 × 9
     │   + self-attention
     │
     └─ Decoder
         384 × 28 × 18
         256 × 56 × 36
         128 × 112 × 72
             ↓
         1 × 112 × 72
```

因此，XAI 可以直接從以下位置取 internal activations：

```text
Encoder block outputs
Middle / bottleneck block outputs
Decoder block outputs
Attention Q / K / V
Attention score / output
Individual feature channels
Different diffusion timestep outputs
```

這也是這份研究最重要的特色：

> **不是只看 model input/output，而是把 diffusion U-Net 本身當作分析對象。**

---

# 3. 核心研究問題

可以先定義以下 Research Questions。

## RQ1：不同 U-Net layer 學到的資訊是否不同？

例如：

```text
Encoder early layers
→ 是否偏局部、低階 spatial pattern？

Middle / bottleneck
→ 是否形成比較 compact、stable 的 weather representation？

Decoder
→ 是否逐漸形成 precipitation correction 的空間細節？
```

---

## RQ2：不同 diffusion timestep 學到的 representation 是否不同？

Diffusion sampling：

```text
high sigma
    ↓
medium sigma
    ↓
low sigma
    ↓
near-clean output
```

可以問：

> 大尺度氣象結構是否在高 noise 階段先形成，而細尺度降水 pattern 在低 noise 階段才形成？

這必須實驗驗證，不能直接當成結論。

---

## RQ3：是否存在「選擇性 latent channels」？

例如某個 bottleneck channel：

```text
channel 183
```

是否只在某些特定 precipitation regimes 強烈 activation？

例如：

- heavy rain
- monsoon
- typhoon-associated events
- orographic rainfall
- dry cases
- strong moisture transport cases

---

## RQ4：latent representation 是否和 physical variables 有可解釋關係？

例如：

\[
corr(\phi_c, q700)
\]

\[
corr(\phi_c, tp)
\]

\[
corr(\phi_c, rainfall\ intensity)
\]

\[
corr(\phi_c, terrain-related\ regional\ statistic)
\]

這不代表 causal relationship，但可以作為物理解讀的第一步。

---

## RQ5：這些 latent features 是否真的「影響」模型 prediction？

這是最關鍵的 mechanistic interpretability 問題。

如果把某個 channel：

\[
A_c
\]

改成：

\[
A_c'=0
\]

並保持：

```text
same date
same input
same model
same seed
same sampling settings
```

最後 precipitation prediction 顯著改變，才能有比較強的 causal evidence。

---

# 4. 最重要的相關論文總覽

| 優先級 | Paper | 年份 / Venue | 主要分析對象 | 對本研究最重要的啟發 |
|---|---|---|---|---|
| ★★★★★ | Elucidating the representation of images within an unconditional diffusion model denoiser | 2025 arXiv | U-Net middle block channels | sparse / selective internal channels |
| ★★★★★ | Not All Diffusion Model Activations Have Been Evaluated as Discriminative Features | NeurIPS 2024 | diffusion activations、Q/K 等 | 不要只看 block output，要系統比較 activation |
| ★★★★★ | Revelio | ICCV 2025 | layers × timesteps × sparse features | k-SAE / monosemantic feature |
| ★★★★★ | Diffusion Hyperfeatures | NeurIPS 2023 | layer × timestep | multi-scale + multi-time feature aggregation |
| ★★★★☆ | Emergent Correspondence from Image Diffusion (DIFT) | NeurIPS 2023 | intermediate diffusion features | diffusion U-Net features 可當 dense representation |
| ★★★★☆ | Plug-and-Play Diffusion Features | CVPR 2023 | ResBlock features + self-attention | PCA / feature visualization / feature intervention |
| ★★★☆☆ | DAAM | ACL 2023 Best Paper | cross-attention | attribution map 的建立與驗證 |
| ★★★★☆ | Mechanistic Interpretability via Cross-Attention Interventions | ACL Findings 2026 | attention + fixed-seed intervention | causal validation / fixed seed |
| ★★★★☆ | Unveiling Concept Attribution in Diffusion Models | 2024 | model components | positive / negative component contribution |
| ★★★★★ | Mechanistic Interpretability Tool for AI Weather Models | 2026 | weather latent representations | PCA / cosine similarity / physical interpretation |
| ★★★★★ | Towards mechanistic understanding in a data-driven weather model | 2025 | GraphCast latent features + SAE | weather feature discovery + causal intervention |
| ★★★★★ | How far can we downscale? | 2026 | precipitation diffusion | 直接鄰近：降水 downscaling + physical interpretability |

---

# 5. 論文一：Elucidating the Representation of Images Within an Unconditional Diffusion Model Denoiser

**Zahra Kadkhodaie, Stéphane Mallat, Eero Simoncelli, 2025**

Paper：<https://arxiv.org/abs/2506.01912>

## 核心問題

這篇不是用 diffusion model 做生成後再分析結果，而是直接研究：

> **一個只為 denoising 而訓練的 U-Net，內部到底形成了什麼 representation？**

這和本研究非常接近。

## 核心發現

作者分析 U-Net middle block，發現：

1. 個別影像只會強烈啟動部分 channels。
2. middle block representation 有明顯 sparsity。
3. 對每個 channel 做 spatial average 後，可得到有意義的 image representation。
4. 某些 channels 對特定視覺 pattern 有選擇性。

## 數學形式

假設 middle block：

\[
A \in \mathbb{R}^{C\times H\times W}
\]

對每個 channel：

\[
\phi_c
=
\frac{1}{HW}
\sum_{i=1}^{H}
\sum_{j=1}^{W}
A_{c,i,j}
\]

最後得到：

\[
\phi
=
[\phi_1,\phi_2,\dots,\phi_C]
\in
\mathbb{R}^{C}
\]

## 對我的模型

我的 bottleneck：

\[
A
\in
\mathbb{R}^{512\times14\times9}
\]

可以做：

\[
[512,14,9]
\rightarrow
[512]
\]

每一天、每個 member、每個 timestep 都可以有一個 512 維 representation。

例如：

```text
date = 2018-08-23
member = 5
step = 20
layer = bottleneck

activation
[512, 14, 9]

spatial mean
        ↓
[512]
```

接著可以找：

```text
Top activated channels
Top activating dates
Heavy-rain vs light-rain activation difference
Seasonal difference
```

## 本研究可借用的方法

```text
U-Net bottleneck activation
        ↓
spatial average
        ↓
channel activation vector
        ↓
rank channels
        ↓
找 top activating events
        ↓
分析事件共同物理特徵
```

這是非常適合第一版的 XAI。

---

# 6. 論文二：Not All Diffusion Model Activations Have Been Evaluated as Discriminative Features

**Benyuan Meng et al., NeurIPS 2024**

Paper：<https://proceedings.neurips.cc/paper_files/paper/2024/hash/633780c1344d0c95e4d2dd3431fe08d9-Abstract-Conference.html>

Code：<https://github.com/Darkbblue/generic-diffusion-feature>

## 核心問題

Diffusion backbone 裡有非常多 activation：

```text
block output
residual feature
attention query
attention key
attention value
attention output
...
```

過去研究常常只挑固定幾個 block。

這篇強調：

> **不是所有 activation 都有被公平比較，也不能先假設某個 layer 一定最有用。**

## 對我的研究最重要的意義

不能直接說：

```text
bottleneck 最深
→ 所以一定最可解釋
```

應該做 systematic layer comparison。

例如：

| Layer | Shape |
|---|---|
| Encoder L0 | 128 × 112 × 72 |
| Encoder L1 | 256 × 56 × 36 |
| Encoder L2 | 384 × 28 × 18 |
| Encoder L3 | 512 × 14 × 9 |
| Middle | 512 × 14 × 9 |
| Decoder L2 | 384 × 28 × 18 |
| Decoder L1 | 256 × 56 × 36 |
| Decoder L0 | 128 × 112 × 72 |

再比較：

```text
heavy-rain discrimination
season discrimination
physical correlation
feature sparsity
spatial interpretability
causal ablation effect
```

---

# 7. 論文三：Revelio

**Dahye Kim, Xavier Thomas, Deepti Ghadiyaram, ICCV 2025**

Paper：<https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Revelio_Interpreting_and_leveraging_semantic_information_in_diffusion_models_ICCV_2025_paper.html>

Project：<https://revelio-diffusion.github.io/revelio/>

Code：<https://github.com/revelio-diffusion/revelio>

## 核心問題

Revelio 明確研究：

> diffusion model 的不同 layers 與 denoising timesteps 裡，semantic information 怎麼被 representation？

這幾乎直接對應：

\[
layer \times timestep
\]

的 XAI。

## 最重要方法：k-Sparse Autoencoder

直接看 U-Net channel 有個問題：

\[
channel\ c
\]

可能同時包含很多不同概念，也就是 polysemantic feature。

Sparse Autoencoder 嘗試把 dense internal representation：

\[
A
\]

轉成比較 sparse 的 latent features：

\[
z
\]

使每次只有少數 latent units active。

概念：

```text
U-Net activation
     ↓
k-Sparse Autoencoder
     ↓
Sparse latent features
     ↓
Top activating cases
     ↓
human / quantitative interpretation
```

## 對我的專案

第一階段不用立刻做 SAE。

可以先：

```text
raw channel analysis
PCA
clustering
probing
```

若發現 channel 很混亂，再升級：

```text
U-Net feature
     ↓
k-SAE
     ↓
interpretable sparse features
```

---

# 8. 論文四：Diffusion Hyperfeatures

**Grace Luo, Lisa Dunlap, Dong Huk Park, Aleksander Holynski, Trevor Darrell, NeurIPS 2023**

Paper：<https://proceedings.neurips.cc/paper_files/paper/2023/hash/942032b61720a3fd64897efe46237c81-Abstract-Conference.html>

Code：<https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures>

Project：<https://diffusion-hyperfeatures.github.io/>

## 最重要觀念

Diffusion representation 不是只有 layer，還有 timestep。

真正的 feature space 可以寫成：

\[
F_{l,t}
\]

其中：

- \(l\)：U-Net layer
- \(t\)：diffusion timestep

## 對我的模型

目前 inference 約 40 EDM steps。

可以抽：

```text
step 0
step 5
step 10
step 20
step 30
step 39
```

每一步再抽：

```text
Encoder
Middle
Decoder
```

得到：

```text
                 t0   t5   t10   t20   t30   t39
Encoder L0        F    F     F     F     F     F
Encoder L1        F    F     F     F     F     F
Middle            F    F     F     F     F     F
Decoder L1        F    F     F     F     F     F
Decoder L0        F    F     F     F     F     F
```

這就是本研究很重要的二維分析空間。

---

# 9. 論文五：Emergent Correspondence from Image Diffusion / DIFT

**Luming Tang et al., NeurIPS 2023**

Paper：<https://proceedings.neurips.cc/paper_files/paper/2023/file/0503f5dce343a1d06d16ba103dd52db1-Paper-Conference.pdf>

## 核心觀念

Diffusion U-Net 的 intermediate feature maps，即使沒有額外 supervision，也可以形成有用的 dense feature descriptors。

## 對我的專案

例如 decoder：

\[
F
\in
\mathbb{R}^{384\times28\times18}
\]

某個 spatial location：

\[
F(:,i,j)
\in
\mathbb{R}^{384}
\]

可以視為這個位置的 latent descriptor。

兩個位置的 similarity：

\[
sim(a,b)
=
\frac{
F_a\cdot F_b
}{
\|F_a\|\|F_b\|
}
\]

可以研究：

```text
不同地區 latent similarity
不同日期相似 precipitation pattern 的 latent similarity
同一地點不同 weather regime 的 representation change
```

---

# 10. 論文六：Plug-and-Play Diffusion Features

**Narek Tumanyan et al., CVPR 2023**

Paper：<https://openaccess.thecvf.com/content/CVPR2023/html/Tumanyan_Plug-and-Play_Diffusion_Features_for_Text-Driven_Image-to-Image_Translation_CVPR_2023_paper.html>

Code：<https://github.com/MichalGeyer/plug-and-play>

## 值得學的部分

這篇直接使用：

```text
ResBlock spatial features
Self-attention
PCA visualization
Feature injection / manipulation
```

所以對我的研究最有價值的是：

1. intermediate feature extraction
2. PCA visualization
3. attention visualization
4. intervention 的思維

## PCA Feature Map

假設：

\[
A=[384,28,18]
\]

reshape：

\[
384\times504
\]

轉置：

\[
504\times384
\]

PCA：

\[
384D\rightarrow3D
\]

得到：

\[
504\times3
\]

reshape：

\[
28\times18\times3
\]

可以把：

```text
PC1 → R
PC2 → G
PC3 → B
```

形成 latent feature map visualization。

對氣象資料而言，不應把 RGB 本身過度解讀，但它可以幫助看這一層是否自然形成 spatial regions / boundaries / structures。

---

# 11. 論文七：DAAM

**Raphael Tang et al., ACL 2023 Best Paper**

Paper：<https://aclanthology.org/2023.acl-long.310/>

Code：<https://github.com/castorini/daam>

## 核心方法

DAAM 聚合 Stable Diffusion denoising U-Net 裡的 cross-attention：

```text
text token
    ↓
cross attention
    ↓
spatial attribution map
```

## 為什麼不能直接照搬到我的模型？

我的 condition 是：

```text
q700
t2m
u
v
msl
tp
mask
```

以 image channels 的方式 concat。

不是：

```text
text token
→ cross-attention
```

因此我不能直接說：

```text
q700 attention map
t2m attention map
```

因為目前 architecture 沒有這種 cross-attention。

## 但它值得借用的觀念

DAAM 告訴我們：

> diffusion denoising network 本身可以被當成 attribution / explanation 的分析對象。

真正適合我的方法不是 text cross-attention，而是：

```text
activation analysis
channel analysis
attention internal analysis
feature intervention
```

---

# 12. 論文八：Mechanistic Interpretability via Cross-Attention Interventions

**Maisha Maliha, Dean F. Hougen, Findings of ACL 2026**

Paper：<https://aclanthology.org/2026.findings-acl.1265/>

## 最重要的地方不是 Cross-Attention，而是 Causal Intervention

作者：

1. 記錄 denoising 中的 internal activation。
2. 建立 spatial grounding。
3. 做 controlled intervention。
4. **保持 sampling seed 固定。**
5. 比較 intervention 前後結果。

這對 diffusion XAI 非常重要。

## 對我的專案

可以改成：

```text
Original:
channel 183 正常

Intervention:
channel 183 = 0

其他條件完全固定：
- same date
- same inputs
- same checkpoint
- same member
- same seed
- same sampler settings
```

比較：

\[
\Delta P
=
P_{\text{original}}
-
P_{\text{ablated}}
\]

這比 heatmap 看起來很亮更有 causal evidence。

---

# 13. 論文九：Unveiling Concept Attribution in Diffusion Models

**Quang H. Nguyen, Hoang Phan, Khoa D. Doan, 2024**

Paper：<https://arxiv.org/abs/2412.02542>

Code：<https://github.com/mail-research/CAD-attribution4diffusion>

## 核心觀念

它問：

> 模型中的不同 components 如何一起造成某個 concept？

而不是只問：

> 哪一層有概念？

作者分析：

```text
positive contributing components
negative contributing components
```

## 對我的專案

可以轉成：

```text
某次 heavy-rain correction
        ↓
哪些 blocks 增強 precipitation correction？
哪些 blocks 抑制 correction？
```

這比只找最大 activation 更完整。

---

# 14. Weather XAI：Mechanistic Interpretability Tool for AI Weather Models

**Kirsten I. Tempest, Matthias Beylich, George C. Craig, 2026**

Paper：<https://arxiv.org/abs/2604.20467>

## 為什麼非常重要？

這篇不是 diffusion，而是 AI weather model / GraphCast。

但它提供了一個非常適合本研究的 weather interpretability 框架：

```text
internal latent representation
       ↓
organize / extract
       ↓
PCA
cosine similarity
       ↓
latent directions
       ↓
meteorological interpretation
```

作者以 mid-latitude synoptic-scale waves 與 specific humidity 做初步案例。

## 對我的研究的意義

Computer Vision diffusion papers 可以支持：

> U-Net internal activations 是有用的 representations。

這篇則支持：

> Weather model internal latent representations 可以進一步和 meteorological phenomena 做連結。

所以本研究的 bridge 可以寫成：

\[
\boxed{
Diffusion\ internal\ representation
+
Weather\ mechanistic\ interpretability
}
\]

---

# 15. Weather XAI：Towards Mechanistic Understanding in a Data-Driven Weather Model

**Theodore MacMillan, Nicholas T. Ouellette, 2025**

Paper：<https://arxiv.org/abs/2512.24440>

## 核心方法

作者將 mechanistic interpretability / Sparse Autoencoder 類思維用在 GraphCast。

找到與以下現象有關的 latent features：

- tropical cyclones
- atmospheric rivers
- diurnal / seasonal behavior
- large-scale precipitation
- geography
- sea ice

並進一步進行 intervention。

## 對我的研究最重要的啟發

真正理想的 weather XAI 不應停在：

```text
這個 channel 很亮
```

而應該走：

```text
feature detection
    ↓
physical association
    ↓
feature intervention
    ↓
prediction change
    ↓
physical consistency check
```

---

# 16. 降水 Diffusion：How Far Can We Downscale?

**How far can we downscale? Resolution limits and physical interpretability of diffusion models for African precipitation, 2026**

目前可取得版本：<https://jangholee.com/files/2026_LeeShamekh_MLE.pdf>

## 為什麼是直接鄰近工作？

它同時包含：

```text
precipitation
downscaling
conditional diffusion
U-Net
physical interpretability
```

非常接近本研究 application。

## 它主要怎麼做 interpretability？

目前其 physical interpretability 主要使用：

\[
\boxed{perturbation-based\ feature\ importance}
\]

也就是看：

> 在不同 atmospheric contexts 下，模型對不同 predictor 的依賴如何改變？

這比較接近 Input-level explanation，而不是 Internal U-Net representation explanation。

## 因此可以形成研究 gap

該類工作已開始回答：

> diffusion precipitation downscaling 依賴哪些 atmospheric predictors？

但仍可進一步問：

> **這些 atmospheric signals 在 denoising U-Net 裡是如何被 representation、轉換與逐步使用的？**

也就是：

```text
Input importance
        ↓
本研究再往模型內部走
        ↓
Layer
Channel
Timestep
Attention
Causal internal intervention
```

---

# 17. 如何把這些文獻接到我的專案

目前模型的 internal representation 可以寫：

\[
F_{d,m,t,l}
\]

其中：

- \(d\) = date / event
- \(m\) = ensemble member
- \(t\) = diffusion timestep
- \(l\) = U-Net layer

而：

\[
F_{d,m,t,l}
\in
\mathbb{R}^{C_l\times H_l\times W_l}
\]

例如：

```text
date   = 2018-08-23
member = 7
step   = 20
layer  = bottleneck

F = [512, 14, 9]
```

這就是所有後續 XAI analysis 的基本資料單位。

---

# 18. 第一版 XAI 方法：Activation Hook

最直接的方法是利用 PyTorch forward hook。

概念：

```python
features = {}

def make_hook(name):
    def hook(module, inputs, output):
        features[name] = output.detach().cpu()
    return hook
```

再：

```python
handle = target_module.register_forward_hook(
    make_hook("bottleneck")
)
```

## 建議先 hook 的 layers

### Encoder

```text
72x112_block1
36x56_block1
18x28_block1
9x14_block1
```

### Middle / Bottleneck

```text
9x14_in0
9x14_in1
```

### Decoder

```text
18x28_block2
36x56_block2
72x112_block2
```

## 為什麼不要第一版每個 block 都存？

因為 diffusion inference：

```text
40 steps
×
30 members
×
很多 layers
```

資料量會非常大。

例如只看 bottleneck：

\[
512\times14\times9
=
64,512
\]

floats / call。

如果所有 layer、所有 step、所有 member、所有 dates 全存，很容易爆 storage。

因此 MVP 應該先 sparse sampling：

```text
selected dates
selected members
selected timesteps
selected layers
```

---

# 19. Layer-wise 分析

比較：

```text
Encoder early
Encoder deep
Middle
Decoder deep
Decoder late
```

可以量化：

## 1. Activation Mean

\[
\mu_l
=
Mean(|F_l|)
\]

## 2. Activation Sparsity

例如：

\[
S_l
=
\frac{
\#(|F_l|<\epsilon)
}{
\#F_l
}
\]

## 3. Feature Variance

\[
Var(F_l)
\]

## 4. Event separability

用 layer feature 做 simple probe：

```text
heavy rain / normal rain
season A / season B
region A / region B
```

---

# 20. Channel-wise 分析

假設 bottleneck：

\[
A\in\mathbb{R}^{512\times14\times9}
\]

spatial average：

\[
\phi_c
=
Mean_{H,W}(A_c)
\]

得到：

\[
\phi\in\mathbb{R}^{512}
\]

## 分析方法

### Top activated channels

```text
channel 183
channel 41
channel 309
...
```

### Top activating dates

對 channel 183：

```text
rank all dates by φ_183
```

得到：

```text
Top 1  2018-08-23
Top 2  ...
Top 3  ...
```

再檢查這些 case 是否共享物理 pattern。

## 注意

不能因為 top cases 都是大雨就直接說：

> channel 183 = heavy-rain neuron。

比較正確：

> channel 183 的 activation 與 heavy-rain cases 有穩定 association。

還要進一步 intervention 才能談 causal role。

---

# 21. PCA / Cosine Similarity / Clustering

## PCA

如果有 N 個 dates：

\[
\Phi
\in
\mathbb{R}^{N\times512}
\]

做：

\[
PCA(512\rightarrow2)
\]

得到：

\[
[N,2]
\]

可以畫 PC1 vs PC2，並用：

```text
rainfall intensity
season
event class
region
```

上色。

## Cosine Similarity

\[
sim(a,b)
=
\frac{
\phi_a\cdot\phi_b
}{
\|\phi_a\|\|\phi_b\|
}
\]

問：

> 類似 precipitation regimes 是否在 latent space 也相似？

## Clustering

例如：

```text
KMeans
HDBSCAN
hierarchical clustering
```

但 cluster interpretation 要非常小心。

流程：

```text
latent cluster
    ↓
collect cases
    ↓
看 rainfall / atmospheric variables
    ↓
找共同 physical pattern
```

---

# 22. Diffusion Timestep 分析

這是本研究和一般 CNN XAI 最大的差異之一。

Diffusion 有：

\[
t=0,\dots,T
\]

或者用 \(\sigma\) 表示 noise level。

## 建議先選幾個代表 timestep

不要第一版全部 40 steps。

例如：

```text
high noise:
step 0

early:
step 5

middle:
step 15

late:
step 30

near-clean:
step 39
```

## 分析 representation trajectory

同一個 event：

\[
\phi_0,
\phi_5,
\phi_{15},
\phi_{30},
\phi_{39}
\]

可以計算：

\[
sim(\phi_t,\phi_{t+1})
\]

或 PCA trajectory。

## Research Question

> precipitation-related physical representation 是在哪個 denoising 階段逐漸形成？

---

# 23. Spatial Feature Map 分析

對單一 channel：

\[
A_c
\in
\mathbb{R}^{H\times W}
\]

例如：

\[
14\times9
\]

可以 upsample：

\[
14\times9
\rightarrow
112\times72
\]

再跟：

```text
Ground Truth
Coarse
Prediction
Prediction - Coarse
Elevation
Mask
```

疊圖比較。

## 一個非常重要的量

模型真正 correction：

\[
\Delta P
=
P_{pred}
-
P_{coarse}
\]

可以看：

\[
corr(A_c,\Delta P)
\]

比直接跟 truth correlation 更貼近 residual model 的工作。

---

# 24. Attention 分析

目前 U-Net bottleneck 有 self-attention。

如果：

\[
C=512
\]

\[
H\times W=14\times9=126
\]

預設 64 channels/head：

\[
512/64=8
\]

heads。

每個 head attention matrix：

\[
126\times126
\]

## 可以研究

```text
每個 query location 最關注哪些位置？
attention 是否局部？
attention 是否跨區域？
不同 timestep attention 是否變化？
不同 rainfall events 是否產生不同 attention structure？
```

## 但不能直接過度解讀

Attention 不等於完整 causal explanation。

最好仍搭配：

```text
attention visualization
+
feature intervention
+
prediction change
```

---

# 25. Feature Probing

這是非常實用的 quantitative validation。

假設每個 event bottleneck representation：

\[
\phi\in\mathbb{R}^{512}
\]

可以訓練非常簡單的 probe：

```text
Logistic Regression
Linear Regression
small linear classifier
```

## Example 1：Heavy rain probe

Label：

```text
0 = normal
1 = heavy rainfall
```

如果：

```text
Encoder L0 feature → AUC 0.62
Encoder L2 feature → AUC 0.75
Middle feature     → AUC 0.90
Decoder L2 feature → AUC 0.85
```

可以說：

> middle representation contains more linearly accessible information associated with heavy-rain regime.

## Example 2：Continuous physical variable

Prediction target：

```text
domain mean q700
domain max precipitation
regional rainfall
```

使用 Linear Regression：

\[
\hat y=w^T\phi+b
\]

檢查：

\[
R^2
\]

## 注意

Probe 很強不代表模型真的使用這項資訊。

它只代表：

> information is decodable from representation.

所以仍要搭配 causal intervention。

---

# 26. Sparse Autoencoder

若 raw channel 很 polysemantic，進階可以用 SAE。

概念：

\[
F
\rightarrow
Encoder_{SAE}
\rightarrow
z_{sparse}
\rightarrow
Decoder_{SAE}
\rightarrow
\hat F
\]

希望：

\[
F\approx\hat F
\]

而 \(z\) 只有少數 feature active。

## 目的

從：

```text
512 raw U-Net channels
```

轉成：

```text
more sparse latent concepts
```

再對每個 sparse feature：

```text
top activating events
spatial maps
physical correlations
intervention
```

---

# 27. Causal Intervention / Ablation

這應該是整個 XAI 中最重要的 validation。

## Channel Ablation

原始：

\[
F_c
\]

介入：

\[
F_c'=0
\]

## Channel Scaling

\[
F_c'=\alpha F_c
\]

例如：

```text
α = 0
α = 0.5
α = 1
α = 2
```

看 prediction 是否有 monotonic response。

## Group Ablation

如果找到一群與 heavy rain 高度相關的 features：

\[
C^*
\]

一次：

\[
F_{C^*}=0
\]

## Layer Intervention

也可以對整個 layer 做：

```text
feature suppression
feature noise injection
feature replacement
```

但整層 intervention 可能過度破壞 network，解讀要更小心。

## 量化結果

### Prediction change

\[
\Delta P
=
P_{original}
-
P_{intervened}
\]

### MAE change

\[
\Delta MAE
=
MAE_{intervened}
-
MAE_{original}
\]

### Regional effect

\[
\Delta P_R
=
Mean_{i\in R}
(P_{original,i}-P_{intervened,i})
\]

### Extreme precipitation change

例如 \(\Delta P_{95}\) 或 threshold exceedance area change。

---

# 28. 固定 Seed 為什麼非常重要

Diffusion 本身是 stochastic。

如果：

```text
Original run
seed = 42

Ablation run
seed = 99
```

則：

\[
P_{original}-P_{ablated}
\]

混合了：

```text
feature intervention effect
+
random noise trajectory difference
```

無法乾淨解釋。

## 正確做法

保持：

```text
same date
same checkpoint
same condition
same seed
same ensemble member
same number of steps
same sigma schedule
same sampling parameters
```

只有 feature activation 被改變。

所以：

\[
\Delta P
\]

才比較接近 intervention effect。

這和近期 diffusion mechanistic interpretability 的 controlled intervention 思路一致。

---

# 29. 完整研究 Pipeline

```text
                    Trained Diffusion Model
                              │
                              ▼
                     Fixed Test Events
                              │
                              ▼
                    Run EDM Inference
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        Save Prediction               Save U-Net Features
                                            │
                                  ┌─────────┼──────────┐
                                  ▼         ▼          ▼
                               Layer     Channel    Timestep
                              analysis   analysis    analysis
                                  │         │          │
                                  └────┬────┴────┬─────┘
                                       ▼         ▼
                                      PCA      Spatial Map
                                       │
                                       ▼
                                 Physical Association
                                       │
                         ┌─────────────┼─────────────┐
                         ▼             ▼             ▼
                     q700 / tp      rainfall       terrain
                         │             │             │
                         └─────────────┼─────────────┘
                                       ▼
                               Candidate Features
                                       │
                                       ▼
                              Controlled Ablation
                              SAME SEED / SAME INPUT
                                       │
                                       ▼
                              Prediction Difference
                                       │
                                       ▼
                           Physical / Causal Validation
```

---

# 30. 建議實驗設計

## Experiment A：Layer-wise Representation

目的：

> 找哪一層最有 physical / precipitation information。

比較：

```text
Encoder L0
Encoder L1
Encoder L2
Encoder L3
Middle
Decoder L2
Decoder L1
Decoder L0
```

Metrics：

```text
PCA separation
probe AUC / R²
activation sparsity
physical correlation
```

## Experiment B：Timestep Evolution

固定：

```text
same date
same member
```

比較：

```text
step 0
5
15
30
39
```

觀察：

```text
feature trajectory
spatial organization
physical correlation evolution
```

## Experiment C：Top Activating Cases

針對 candidate channel：

```text
rank all events
```

取：

```text
Top 20
Bottom 20
```

比較：

```text
rainfall
q700
wind
season
regional pattern
```

## Experiment D：Causal Ablation

對 selected channels：

```text
original
vs
channel zero
vs
channel ×0.5
vs
channel ×2
```

固定 seed。

比較：

```text
prediction map
Δprediction
regional mean change
MAE change
extreme rainfall change
```

## Experiment E：Layer × Timestep Matrix

建立：

\[
Score(l,t)
\]

例如 heavy-rain probe AUC。

做 heatmap：

```text
           t0   t5  t15  t30  t39
Enc L0
Enc L1
Middle
Dec L1
Dec L0
```

這非常適合直接對應 Diffusion Hyperfeatures 的研究概念。

---

# 31. 建議圖表

## Figure 1：Model + XAI Hook Architecture

```text
Condition
  ↓
EDMPrecond
  ↓
U-Net

Encoder ──hook
Encoder ──hook
Middle  ──hook
Decoder ──hook
Decoder ──hook
```

## Figure 2：Layer-wise PCA

顯示 Encoder / Middle / Decoder representation 的變化。

## Figure 3：Timestep Evolution

同一事件：

```text
step 0
step 5
step 15
step 30
step 39
```

feature visualization。

## Figure 4：Top Activating Events

例如：

```text
Feature #183

Top-1 event
Top-2 event
...
Top-8 event
```

並排 precipitation / atmospheric fields。

## Figure 5：Physical Correlation

例如：

```text
feature activation
vs
q700
tp
rainfall intensity
```

## Figure 6：Causal Ablation

```text
Original Prediction
Ablated Prediction
Difference
Ground Truth
```

## Figure 7：Layer × Timestep Heatmap

顯示 probe score / physical correlation / causal effect。

## Figure 8：Feature-level Mechanistic Story

```text
Input atmospheric state
        ↓
U-Net internal feature
        ↓
feature activation
        ↓
spatial correction
        ↓
precipitation output
```

---

# 32. 研究貢獻可以怎麼寫

若實驗結果成立，可以形成以下 contribution。

## Contribution 1：Diffusion U-Net internal representation analysis for precipitation downscaling

> 系統分析 conditional precipitation diffusion model 在不同 U-Net layers 與 denoising timesteps 中的 internal representations。

## Contribution 2：Physical association of latent features

> 將 U-Net latent channels / sparse features 與 precipitation regimes 及 atmospheric predictors 做定量連結。

## Contribution 3：Causal validation

> 透過 fixed-seed feature interventions，確認 selected latent representations 對 precipitation correction 的 causal influence。

## Contribution 4：Layer × timestep interpretability framework

> 不只分析 static neural-network layer，而是把 diffusion timestep 本身納入 XAI，研究 model representation 隨 iterative denoising 演化的過程。

---

# 33. 目前方法的限制與不能過度解釋的地方

## 1. Correlation 不是 causation

\[
corr(feature,rainfall)
\]

高，不等於 feature 就是 rainfall neuron。

## 2. 一個 channel 可能 polysemantic

所以 channel 183 可能同時混合 moisture、terrain、season、rainfall、noise level。

這也是為什麼 SAE 值得作為進階方法。

## 3. PCA component 不是天然物理變數

即使 PC1 看起來像 mountain pattern，也只能說 visually / statistically associated。

不能直接宣稱 PC1 = terrain physics。

## 4. Attention 不等於完整 explanation

Attention map 可提供 model interaction clue，但最好仍需 intervention。

## 5. Ablation 可能是 out-of-distribution intervention

直接：

\[
channel=0
\]

可能產生 training distribution 沒見過的 activation state。

因此可增加：

```text
zero ablation
mean replacement
scaling intervention
matched-control channels
```

增加 robustness。

## 6. Diffusion stochasticity 必須被控制

任何 causal comparison 應保持 seed 與 sampling settings 一致。

---

# 34. 建議實作資料夾

```text
xai/
│
├── hooks/
│   ├── activation_hooks.py
│   ├── attention_hooks.py
│   └── layer_registry.py
│
├── extraction/
│   ├── extract_features.py
│   ├── extract_timestep_features.py
│   └── feature_cache.py
│
├── analysis/
│   ├── channel_activation.py
│   ├── spatial_feature_maps.py
│   ├── pca_analysis.py
│   ├── cosine_similarity.py
│   ├── clustering.py
│   ├── physical_correlation.py
│   └── probing.py
│
├── intervention/
│   ├── channel_ablation.py
│   ├── channel_scaling.py
│   ├── group_ablation.py
│   └── fixed_seed_compare.py
│
├── sae/
│   ├── train_sae.py
│   ├── analyze_sae_features.py
│   └── sae_intervention.py
│
├── visualization/
│   ├── plot_feature_map.py
│   ├── plot_pca.py
│   ├── plot_timestep_trajectory.py
│   ├── plot_ablation_difference.py
│   └── plot_layer_timestep_matrix.py
│
└── outputs/
    ├── features/
    ├── figures/
    ├── tables/
    └── interventions/
```

---

# 35. 最小可行版本（MVP）

如果先做最小版，建議只做以下五件事。

## Step 1：選三層

```text
Encoder deep
Middle
Decoder late
```

例如：

```text
9x14_block1
9x14_in0
72x112_block2
```

## Step 2：選五個 timestep

```text
0
5
15
30
39
```

## Step 3：選代表事件

例如：

```text
heavy rain
moderate rain
light / dry
different seasons
```

每類先 20–50 cases。

## Step 4：做三種 descriptive XAI

```text
channel spatial average
PCA
spatial feature map
```

## Step 5：做 fixed-seed ablation

挑：

```text
3–10 candidate channels
```

做：

```text
original
vs
zero ablation
```

比較：

```text
prediction map
Δprediction
ΔMAE
```

這已經可以形成一個有研究價值的第一版。

---

# 36. 進階版本

MVP 成功後再逐步加入：

```text
all layers
all selected timesteps
probe models
SAE
attention Q/K/V analysis
feature grouping
causal tracing
interactive visualization
```

---

# 37. 論文 Related Work 可以怎麼組織

建議分成四段。

## 37.1 Diffusion-based Downscaling

介紹：

```text
ClimateDiffuse
CorrDiff
precipitation diffusion downscaling
```

說明 diffusion 可生成 high-frequency / stochastic fine-scale structures。

## 37.2 Diffusion Features and Internal Representations

介紹：

```text
DIFT
Diffusion Hyperfeatures
Plug-and-Play Diffusion Features
Not All Diffusion Model Activations...
```

核心論點：

> diffusion denoisers 中的 intermediate features 本身具有 meaningful representations。

## 37.3 Mechanistic Interpretability of Diffusion Models

介紹：

```text
Elucidating the representation...
Revelio
DAAM
Cross-Attention Intervention
Concept Attribution
```

核心：

> 不只是利用 features，而是理解 features / components 的功能與 causal role。

## 37.4 Interpretability of AI Weather Models

介紹：

```text
Mechanistic Interpretability Tool for AI Weather Models
Towards mechanistic understanding in a data-driven weather model
African precipitation physical interpretability
```

再指出 gap：

> 現有 weather XAI 已開始研究 predictor importance 或 latent physical features，但對 **conditional precipitation diffusion U-Net 本身的 layer × channel × timestep representation，以及其 causal role**，仍有值得深入研究的空間。

---

# 38. 參考文獻

## Diffusion Internal Representation / Feature

### Kadkhodaie et al. (2025)

**Elucidating the representation of images within an unconditional diffusion model denoiser**

<https://arxiv.org/abs/2506.01912>

用途：U-Net middle block、channel sparsity、spatial-average representation、selective internal channels。

### Meng et al. (2024)

**Not All Diffusion Model Activations Have Been Evaluated as Discriminative Features**

NeurIPS 2024

<https://proceedings.neurips.cc/paper_files/paper/2024/hash/633780c1344d0c95e4d2dd3431fe08d9-Abstract-Conference.html>

Code：<https://github.com/Darkbblue/generic-diffusion-feature>

用途：systematic activation selection、block activations、attention Q/K、feature comparison。

### Kim et al. (2025)

**Revelio: Interpreting and leveraging semantic information in diffusion models**

ICCV 2025

<https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Revelio_Interpreting_and_leveraging_semantic_information_in_diffusion_models_ICCV_2025_paper.html>

Code：<https://github.com/revelio-diffusion/revelio>

用途：layer × timestep、k-Sparse Autoencoder、monosemantic features、lightweight probing。

### Luo et al. (2023)

**Diffusion Hyperfeatures: Searching Through Time and Space for Semantic Correspondence**

NeurIPS 2023

<https://proceedings.neurips.cc/paper_files/paper/2023/hash/942032b61720a3fd64897efe46237c81-Abstract-Conference.html>

Code：<https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures>

用途：multi-layer features、multi-timestep features、feature aggregation。

### Tang et al. (2023)

**Emergent Correspondence from Image Diffusion**

NeurIPS 2023

<https://proceedings.neurips.cc/paper_files/paper/2023/file/0503f5dce343a1d06d16ba103dd52db1-Paper-Conference.pdf>

用途：DIFT、diffusion intermediate features、dense feature descriptors。

### Tumanyan et al. (2023)

**Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation**

CVPR 2023

<https://openaccess.thecvf.com/content/CVPR2023/html/Tumanyan_Plug-and-Play_Diffusion_Features_for_Text-Driven_Image-to-Image_Translation_CVPR_2023_paper.html>

Code：<https://github.com/MichalGeyer/plug-and-play>

用途：ResBlock features、PCA visualization、self-attention、feature manipulation。

## Diffusion XAI / Mechanistic Interpretability

### Tang et al. (2023)

**What the DAAM: Interpreting Stable Diffusion Using Cross Attention**

ACL 2023 Best Paper

<https://aclanthology.org/2023.acl-long.310/>

用途：diffusion attribution、cross-attention aggregation、spatial attribution maps。

注意：本專案沒有 text cross-attention，因此不能直接套 DAAM，但 attribution 與 validation 思路值得參考。

### Maliha & Hougen (2026)

**Mechanistic Interpretability of Text-to-Image Diffusion Models via Cross-Attention Interventions**

Findings of ACL 2026

<https://aclanthology.org/2026.findings-acl.1265/>

用途：causal intervention、fixed-seed counterfactual comparison、timestep analysis、module/head specialization。

### Nguyen et al. (2024)

**Unveiling Concept Attribution in Diffusion Models**

<https://arxiv.org/abs/2412.02542>

Code：<https://github.com/mail-research/CAD-attribution4diffusion>

用途：component attribution、positive contribution、negative contribution、model editing / ablation。

## Weather Mechanistic Interpretability

### Tempest et al. (2026)

**Mechanistic Interpretability Tool for AI Weather Models**

<https://arxiv.org/abs/2604.20467>

用途：weather latent representations、PCA、cosine similarity、meteorological latent directions。

### MacMillan & Ouellette (2025)

**Towards mechanistic understanding in a data-driven weather model: internal activations reveal interpretable physical features**

<https://arxiv.org/abs/2512.24440>

用途：GraphCast、Sparse Autoencoder、tropical cyclone、atmospheric river、precipitation、causal intervention。

## Precipitation Diffusion + Physical Interpretability

### Lee & Shamekh et al. (2026)

**How far can we downscale? Resolution limits and physical interpretability of diffusion models for African precipitation**

可取得版本：<https://jangholee.com/files/2026_LeeShamekh_MLE.pdf>

用途：conditional diffusion、precipitation downscaling、physical interpretability、perturbation-based predictor importance。

與本研究差異：該工作較偏 input predictor perturbation；本研究希望進一步研究 internal U-Net representation。

---

# 最後：本研究最核心的一句話

> **本研究不是只想知道「哪個輸入氣象變數重要」，而是想進一步回答：在 conditional precipitation diffusion 的 iterative denoising 過程中，U-Net 的不同 layers、channels 與 timesteps 如何形成與使用可解釋的氣象表徵，並透過 fixed-seed causal feature intervention 驗證這些內部特徵是否真的影響 precipitation correction。**

可以把整個研究濃縮成：

```text
Diffusion U-Net
      ↓
Internal Activations
      ↓
Layer × Channel × Timestep
      ↓
PCA / Similarity / Spatial Map / Probing
      ↓
Physical Association
      ↓
Candidate Latent Features
      ↓
Fixed-Seed Causal Ablation
      ↓
Prediction Change
      ↓
Mechanistic Interpretation of
Precipitation Downscaling
```

---

# 建議閱讀順序

如果時間有限，優先：

1. **Elucidating the representation of images within an unconditional diffusion model denoiser**
2. **Not All Diffusion Model Activations Have Been Evaluated as Discriminative Features**
3. **Diffusion Hyperfeatures**
4. **Revelio**
5. **Mechanistic Interpretability Tool for AI Weather Models**
6. **Towards mechanistic understanding in a data-driven weather model**
7. **How far can we downscale?**
8. **Mechanistic Interpretability of Text-to-Image Diffusion Models via Cross-Attention Interventions**

這個順序剛好對應：

```text
先理解 diffusion feature
        ↓
再知道 feature 分布在 layer × timestep
        ↓
再研究 interpretable / sparse features
        ↓
再連到 weather physics
        ↓
最後用 causal intervention 做嚴格驗證
```

---

# 專案對應補充

目前專案 GitHub：

<https://github.com/409262308/IDW_loss/tree/dev>

在實作 XAI 時，核心修改位置會落在：

```text
Network.py
    ↓
UNet.forward()
UNetBlock.forward()
AttentionOp

Inference_Taiwan_Precip_Metrics_CSVold.py
    ↓
sample_model_eds_batch()
    ↓
每一個 diffusion timestep 呼叫 model(...) 的位置
```

建議原則：

> **盡量不要改原本 trained model 的數學，只在 inference 端加 non-invasive hooks；真正做 causal intervention 時再另外建立 intervention mode。**

如此可避免因為加入 XAI code 而改變原始 prediction pipeline。
