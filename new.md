''' text
有，而且我查了一輪後，**你真正要找的關鍵字不是只有「XAI + diffusion」**，而是下面這一整條研究線：

$$
\boxed{
\text{Diffusion internal representations}
}
$$

$$
\boxed{
\text{Diffusion features}
}
$$

$$
\boxed{
\text{U-Net activations}
}
$$

$$
\boxed{
\text{Mechanistic interpretability of diffusion models}
}
$$

這些論文不是只對 final prediction 做 SHAP，而是真的會把 **Diffusion denoising U-Net 裡的 encoder / middle block / decoder / attention / individual channels / different diffusion timesteps** 拿出來研究。

而且我認為其中有幾篇跟你現在的模型非常接近。

---

# 一、我先給你結論：最值得你讀的 8 篇

如果照「和你現在研究的直接相關程度」排序，我會排：

| 優先    | Paper                                                                                         | 它看 Diffusion 裡什麼？                                                                     |            跟你的適合度 |
| ----- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ----------------: |
| ⭐⭐⭐⭐⭐ | **Elucidating the representation of images within an unconditional diffusion model denoiser** | **U-Net middle block individual channels、activation、sparsity、feature representation** |         **10/10** |
| ⭐⭐⭐⭐⭐ | **Not All Diffusion Model Activations Have Been Evaluated as Discriminative Features**        | U-Net 各 block、內部 activations、Q/K 等                                                    |        **9.5/10** |
| ⭐⭐⭐⭐⭐ | **Revelio: Interpreting and leveraging semantic information in diffusion models**             | 不同 U-Net layers / timesteps + sparse features                                         |        **9.5/10** |
| ⭐⭐⭐⭐½ | **Diffusion Hyperfeatures**                                                                   | 不同 U-Net layers × diffusion timesteps 的 feature maps                                  |          **9/10** |
| ⭐⭐⭐⭐½ | **Emergent Correspondence from Image Diffusion (DIFT)**                                       | 直接抽 U-Net decoder intermediate feature maps                                           |          **9/10** |
| ⭐⭐⭐⭐  | **A Tale of Two Features**                                                                    | PCA 分析 U-Net decoder layer features                                                   |        **8.5/10** |
| ⭐⭐⭐⭐  | **Plug-and-Play Diffusion Features**                                                          | U-Net spatial features + self-attention                                               |        **8.5/10** |
| ⭐⭐⭐   | **DAAM / Mechanistic Interpretability via Cross-Attention**                                   | U-Net attention + causal intervention                                                 | **方法值得學，但不能直接照搬** |

另外大氣領域我找到兩條很值得你接起來：

* **Mechanistic Interpretability Tool for AI Weather Models (2026)**
* **Towards mechanistic understanding in a data-driven weather model (2025)**

它們不是 diffusion，但研究問題跟你非常適合：

> **「AI weather model 裡面的 latent channels 到底學到什麼物理現象？」**

還有一篇 2026 年非常接近你的 application：

> **How far can we downscale? Resolution limits and physical interpretability of diffusion models for African precipitation**

它就是 precipitation + conditional diffusion + U-Net + physical interpretability，但它目前的 interpretability 主力是 **perturbation-based predictor importance**，不是你說的「直接研究 U-Net internal features」。([數字對象識別器][1])

所以你的方向其實可以漂亮地把：

$$
\boxed{
\text{Diffusion-U-Net internal feature interpretability}
}
$$

跟：

$$
\boxed{
\text{physical interpretability for weather}
}
$$

接在一起。

---

# 二、最重要第一篇：這篇我認為你一定要讀

## 1. Elucidating the representation of images within an unconditional diffusion model denoiser

Kadkhodaie, Mallat, Simoncelli，2025。

[Elucidating the representation of images within an unconditional diffusion model denoiser](https://arxiv.org/abs/2506.01912?utm_source=chatgpt.com)

這篇幾乎正中你現在想做的事情。

它問：

> Diffusion U-Net trained only for denoising，**裡面的 hidden channels 到底代表什麼？**

而且不是 Stable Diffusion text cross-attention。

它研究的是：

$$
\boxed{\text{真正的 denoising U-Net}}
$$

這點跟你非常重要。

作者直接研究一個 ImageNet diffusion denoiser 裡：

```text
Encoder
   ↓
Middle block
   ↓
Decoder
```

的 hidden activations。([arXiv][2])

---

# 三、它到底做了什麼？

假設某一個 U-Net block output：

$$
A_j
\in
\mathbb R^{C\times H\times W}.
$$

這和你現在 Network 完全是同樣概念。

例如你的 bottleneck：

$$
A
\in
\mathbb R^{512\times14\times9}.
$$

作者首先對每個 channel 做 spatial average：

$$
\phi_c
=
\frac{1}{HW}
\sum_{i,j}
A_{c,i,j}.
$$

因此：

$$
[C,H,W]
\rightarrow
[C].
$$

例如：

$$
[512,14,9]
\rightarrow
[512].
$$

這個：

$$
\boxed{\phi\in\mathbb R^{512}}
$$

就變成「這張 image 在 U-Net middle block 的 internal representation」。

論文明確說，他們檢查各 block output 的 channel spatial averages，並發現 **middle block 的 representation 特別穩定，而且呈現 sparse channel activation**。([arXiv][2])

---

# 四、這跟你的模型可以直接怎麼對？

你的 bottleneck 大概：

```text
9x14_in0
9x14_in1

[B,512,14,9]
```

你可以直接做：

$$
A^{(d,t)}
\in
\mathbb R^{512\times14\times9}
$$

其中：

* \(d\)：date/event
* \(t\)：diffusion timestep / sigma

然後：

$$
\phi_c^{(d,t)}
=
Mean_{H,W}
(A_{c,H,W}^{(d,t)}).
$$

所以每一天、每個 diffusion step：

```text
512×14×9
      ↓
spatial mean
      ↓
512-dimensional vector
```

接下來研究：

```text
哪些 channels 最 active？
哪些 channels 只在 heavy rain active？
哪些 channels 對特定天氣型態 active？
哪些 channel 在高 sigma 就出現？
哪些到低 sigma 才出現？
```

這就已經是：

$$
\boxed{\text{Diffusion U-Net internal representation XAI}}
$$

而且是有直接 literature basis 的，不是你自己亂 invent。

---

# 五、這篇還發現「specialized channels」

這部分跟你尤其有潛力。

他們發現 middle block channels 可粗分成：

### Common channels

很多 image 都 activation。

可能表示比較 general 的：

* global structure
* brightness
* common visual properties

### Selective / specialized channels

只在少數 image 高 activation。

例如有些 channel 會特別對：

* periodic texture
* bird-like composition
* dog faces
* particular visual structures

有反應。([arXiv][2])

---

## 放到你的 precipitation case

你不能一開始就直接說：

> Channel 183 = orographic precipitation。

這樣不嚴謹。

但你可以測：

```text
Channel 183
      ↓
在哪些日期最 active？
      ↓
把 Top-20 dates 找出來
      ↓
這些 case 有什麼共同特徵？
```

例如可能最後觀察到：

```text
Top activation cases:
2018-08-23
2019-08-08
2020-05-22
...
```

這些 case 都具有：

```text
高 q700
東南風
中央山脈迎風側強降水
```

才可以形成 hypothesis：

> Channel 183 **可能與某類 moisture/orographic precipitation structure 有關。**

再進一步 intervention 才能確認。

這跟現在 mechanistic interpretability 的思路非常接近。

---

# 六、甚至它還研究「U-Net 哪裡開始 denoise」

這也很適合你。

作者量化不同：

```text
Encoder
Middle
Decoder
```

的 activation sparsity。

他們觀察：

> Encoder 比較像是在建立、展開 noisy input 的 hierarchical representation；到 middle block / decoder 後，channel representation 變得更 sparse，可能表示 network 開始保留 signal、抑制 noise。([arXiv][2])

所以你可以研究：

```text
你的 diffusion precipitation U-Net：

Encoder：
112×72
↓
56×36
↓
28×18
↓
14×9

到底在把什麼 information 編碼進去？

Middle：
14×9×512

是否形成最 compact / stable 的 weather representation？

Decoder：
14×9
↓
28×18
↓
56×36
↓
112×72

何時開始形成局部 rainfall correction？
```

這就比只畫 Grad-CAM 深很多。

---

# 七、第二篇：NeurIPS 2024，非常值得你讀

## Not All Diffusion Model Activations Have Been Evaluated as Discriminative Features

Meng et al., **NeurIPS 2024**。

[Not All Diffusion Model Activations Have Been Evaluated as Discriminative Features](https://proceedings.neurips.cc/paper_files/paper/2024/hash/633780c1344d0c95e4d2dd3431fe08d9-Abstract-Conference.html?utm_source=chatgpt.com)

這篇的核心問題是：

> Diffusion 裡這麼多 activation，到底哪些 layer / module / internal signal 才真正有 useful information？

他們不只看一般：

```text
Block output feature map
```

還指出過去很多研究漏掉了：

$$
Q,\quad K
$$

等 attention 內部 activations，也系統分析不同 architecture 裡的 internal signals。([NeurIPS 會議論文集][3])

---

# 八、它對你最大的啟發：不要一開始假設 Bottleneck 最好

你現在很容易想：

> 那就拿 `512×14×9` bottleneck 做 XAI。

可以作為第一版。

但科學上更好的問題應該是：

$$
\boxed{
\text{Which U-Net block contains the most physically interpretable precipitation representation?}
}
$$

所以你可以比較：

```text
Encoder:
128×112×72
256×56×36
384×28×18
512×14×9

Middle:
512×14×9

Decoder:
384×28×18
256×56×36
128×112×72
```

對每一層都抽 activation。

然後比較：

* precipitation regime separation
* terrain regime separation
* heavy/light rain separation
* cosine similarity
* PCA cluster quality
* physical variable correlation

這就是 NeurIPS 2024 那篇 paper 很值得借鑑的地方。

他們甚至有官方 feature extraction / feature visualization code。([GitHub][4])

---

# 九、第三篇：Revelio，和你的研究故事非常搭

## Revelio: Interpreting and leveraging semantic information in diffusion models

ICCV 2025。

作者研究：

> 不同 diffusion architectures 的不同 **layers + denoising timesteps** 到底怎麼 encode semantic information。

而且使用：

$$
\boxed{k\text{-Sparse Autoencoder}}
$$

去找比較「monosemantic / interpretable」的 features。([開放存取計算機視覺基金會][5])

---

# 十、為什麼 Sparse Autoencoder 對你很重要？

因為直接看：

```text
Channel 182
```

最大的問題是：

$$
\boxed{\text{一個 neural channel 很可能 polysemantic}}
$$

也就是：

> 它不是只代表「颱風」。

它可能同時混：

```text
terrain
humidity
season
rain band
wind direction
noise level
```

所以只看 individual channels 很容易過度解釋。

Sparse Autoencoder：

$$
A
\rightarrow
z_{sparse}
\rightarrow
\hat A
$$

會嘗試把 dense activation 分解成比較稀疏的 latent features。

你之後可能得到：

```text
SAE feature 37
SAE feature 112
SAE feature 205
...
```

再找：

> 哪些 precipitation cases 最 excite feature 37？

就比直接說：

> 原 U-Net channel 37 就是一個物理現象

更有 interpretability 基礎。

Revelio 就是在 diffusion feature interpretation 上走這條路。([開放存取計算機視覺基金會][5])

---

# 十一、第四篇：Diffusion Hyperfeatures

## Diffusion Hyperfeatures: Searching Through Time and Space for Semantic Correspondence

NeurIPS 2023。

[Diffusion Hyperfeatures](https://proceedings.neurips.cc/paper_files/paper/2023/hash/942032b61720a3fd64897efe46237c81-Abstract-Conference.html?utm_source=chatgpt.com)

這篇有一個對你超級重要的觀念：

> Diffusion feature 不是只存在「哪一層」。

還存在：

$$
\boxed{
Layer
\times
Diffusion\ timestep
}
$$

兩個 dimensions。([NeurIPS 會議論文集][6])

---

# 十二、這就是你的模型目前非常有價值的 XAI 方向

你的 inference：

$$
\sigma_{max}
\rightarrow
\sigma_{min}
$$

大約：

```text
step 0
step 1
...
step 39
```

而每一次都跑完整 U-Net。

所以你其實有：

$$
F_{l,t}
$$

其中：

* \(l\)：U-Net layer
* \(t\)：diffusion timestep

例如：

```text
                 t0     t5     t10    t20    t39
Encoder L1       F      F       F      F      F
Encoder L2       F      F       F      F      F
Middle           F      F       F      F      F
Decoder L2       F      F       F      F      F
Decoder L1       F      F       F      F      F
```

這是很漂亮的一個二維研究空間。

---

# 十三、你的問題就可以變成

### RQ1

> 不同 U-Net resolution level 學到什麼 precipitation information？

### RQ2

> 不同 diffusion noise levels 中，physical structure 何時形成？

例如：

```text
σ very high
↓
可能只看出 broad atmospheric regime

σ medium
↓
可能開始定位 rain band

σ low
↓
可能形成 local rainfall intensity / texture
```

這只是待驗證 hypothesis，不能先當結論。

但 Diffusion Hyperfeatures 正好提供：

> **features distributed across both layers and timesteps**

這個 literature foundation。([NeurIPS 會議論文集][6])

---

# 十四、第五篇：DIFT

## Emergent Correspondence from Image Diffusion

NeurIPS 2023。

這篇提出：

$$
\boxed{\text{DIffusion FeaTures — DIFT}}
$$

最核心其實很簡單：

> 直接從 diffusion U-Net intermediate blocks 抽 feature maps，看看它們是否本身包含 semantic / geometric information。

結果發現，即使 diffusion model 從來沒有被訓練做 correspondence，它的 U-Net feature maps 仍然可以提供很有用的 dense representation。([NeurIPS 會議論文集][7])

---

# 十五、DIFT 跟你的關係

你的：

```text
decoder block
[B,384,28,18]
```

可以直接視為：

$$
F\in\mathbb R^{384\times28\times18}.
$$

每個 spatial position：

$$
F(:,i,j)
$$

其實就是一個：

$$
384D
$$

descriptor。

所以可以問：

> 台灣兩個 spatial locations 的 internal diffusion feature 相不相似？

$$
sim(i,j,p,q)
=
\frac{
F_{ij}\cdot F_{pq}
}{
\|F_{ij}\|\|F_{pq}\|
}.
$$

甚至兩個不同 rainfall events：

$$
F^{day1}_{ij}
\quad vs \quad
F^{day2}_{pq}.
$$

可以看：

> 模型是否把不同日期但相似 precipitation pattern 的區域放到相似 latent representation。

這就是從 DIFT 可以借的想法。

---

# 十六、第六篇：A Tale of Two Features

## Stable Diffusion Complements DINO for Zero-Shot Semantic Correspondence

NeurIPS 2023。

這篇對你最值得看的不是 DINO。

而是他們**真的比較 U-Net decoder 不同 layers 的 features**。

他們用 PCA visualizations 發現：

> 不同 decoder depth 的 Stable Diffusion features 有不同性質；一些較早的 decoder features 更偏 structure/semantic information，而較後 layers 則更偏 texture/appearance。([sd-complements-dino.github.io][8])

---

# 十七、這個可以非常自然地改成你的 scientific question

你的 decoder：

```text
14×9 × 512
       ↓
28×18 × 384
       ↓
56×36 × 256
       ↓
112×72 × 128
```

你可以檢查：

### Deep decoder

$$
512\times14\times9
$$

是否比較表現：

* 大尺度 weather regime
* synoptic structure

### Middle decoder

$$
384\times28\times18
$$

是否形成：

* rain-band organization
* regional precipitation structures

### Final decoder

$$
128\times112\times72
$$

是否比較偏：

* local intensity
* fine structure
* coast / terrain detail

注意：

> 這些是你需要實驗驗證的假說，不可以直接把 computer vision paper 的 semantic/texture 結論原封不動套到 precipitation。

但它非常適合當 methodology inspiration。

---

# 十八、第七篇：Plug-and-Play Diffusion Features

## CVPR 2023

這篇研究：

$$
\boxed{
\text{U-Net spatial features}
+
\text{self-attention}
}
$$

而且他們真的有：

```text
ResBlock feature extraction
PCA visualization
Self-Attention visualization
```

的 code。([開放存取計算機視覺基金會][9])

他們還顯示 diffusion generation 過程中，同一 layer 的 PCA feature maps 會隨 timestep 發展出 spatial structure。([開放存取計算機視覺基金會][10])

這跟你的：

```text
40 EDM steps
×
U-Net blocks
```

非常適合。

---

# 十九、你甚至可以直接做 PCA feature map

假設：

$$
A
=
[384,28,18].
$$

每一個 pixel 是：

$$
384D.
$$

reshape：

$$
384\times504.
$$

轉置：

$$
504\times384.
$$

PCA：

$$
384D
\rightarrow
3D.
$$

得到：

$$
504\times3.
$$

reshape：

$$
28\times18\times3.
$$

就可以把前三個 PC 當成 RGB visualization：

```text
PC1 → R
PC2 → G
PC3 → B
```

再 resize：

$$
28\times18
\rightarrow
112\times72.
$$

這樣你就可以「看到」：

> U-Net 這一層怎麼把台灣 precipitation case 分區。

這是很值得你第一版就做的。

---

# 二十、第八篇：DAAM

## What the DAAM: Interpreting Stable Diffusion Using Cross Attention

ACL 2023 Best Paper。

DAAM 把 Stable Diffusion denoising U-Net 裡：

$$
\boxed{cross-attention}
$$

aggregate 成 pixel-level attribution maps。

例如：

```text
prompt = "a dog on grass"

dog token
      ↓
U-Net cross attention
      ↓
heatmap

grass token
      ↓
U-Net cross attention
      ↓
heatmap
```

用這種方式回答：

> 哪些 pixels 是受到哪個 conditioning token 影響？([ACL Anthology][11])

---

# 二十一、但是 DAAM 不能直接套你的 model

這裡非常重要。

Stable Diffusion：

```text
image latent
       +
text tokens
       ↓
cross-attention
```

你的 model：

```text
noisy residual
+
q700
t2m
u
v
msl
tp
mask
↓
channel concat
```

所以你沒有：

```text
q700 token
t2m token
...
```

的 cross-attention map。

因此你不能直接說：

> 我用 DAAM 看 q700 attention。

你的 architecture 沒有這個東西。

你比較適合的是：

$$
\boxed{\text{activation-based interpretability}}
$$

而不是：

$$
\boxed{\text{text cross-attention attribution}}
$$

---

# 二十二、但是 DAAM 的「驗證邏輯」非常值得學

DAAM 後來更進階的 mechanistic work，不只是：

> 看 heatmap 很漂亮。

而會做 intervention。

2026 ACL Findings 的 **Mechanistic Interpretability of Text-to-Image Diffusion Models via Cross-Attention Interventions**：

1. 記錄 U-Net attention activation。
2. 產生 attribution map。
3. 改 prompt 中某一個 word。
4. **保持 sampling seed 相同。**
5. 比較生成 output。
6. 看 attribution mechanism 是否真的跟 output change 有 causal relationship。([ACL Anthology][12])

這點對你非常重要。

---

# 二十三、你可以把它改成「U-Net channel intervention」

這會讓你的工作從普通 feature visualization 升級到：

$$
\boxed{\text{Mechanistic XAI}}
$$

例如你發現：

```text
Middle channel 183
```

似乎和 heavy-rain cases 很有關。

第一階段：

### Observation

保存：

$$
A_{183}
$$

看 activation。

---

第二階段：

### Correlation

找：

$$
corr(\phi_{183},rainfall\ intensity).
$$

---

第三階段：

### Intervention

Inference 同一天、同一 seed。

Original：

$$
A_{183}.
$$

Intervention：

$$
A_{183}'=0.
$$

然後重新從該層 forward。

比較：

$$
\Delta P
=
P_{original}
-
P_{ablated}.
$$

---

如果：

```text
Original prediction
↓
東部 80 mm

Ablate channel183
↓
東部只剩 45 mm
```

而其他地方幾乎不動，

那才有比較強的證據：

> Channel 183 對這類 spatial precipitation correction 有 causal importance。

---

# 二十四、這和單純 heatmap 的差別非常大

普通 XAI：

```text
這裡亮
→ 看起來重要
```

Mechanistic：

```text
這裡亮
↓
我把這個 feature 拿掉
↓
prediction真的改變
↓
而且改變方向有物理意義
```

也就是：

$$
\boxed{
Correlation
\rightarrow
Causal\ intervention
}
$$

這會讓研究嚴謹很多。

---

# 二十五、最新一篇很適合你的：Revelio

前面提過，但我再特別強調。

ICCV 2025 的 Revelio 明確研究：

> rich visual semantic information 在 diffusion architecture 的不同 layers 與 timesteps 怎麼 representation。

並透過 k-sparse representation 找 interpretable features，再用 lightweight classifiers 驗證這些 diffusion features 確實包含可用資訊。([開放存取計算機視覺基金會][5])

所以你將來如果覺得：

> PCA 太粗、單 channel 太不穩定

你的下一階段就是：

$$
\boxed{
U-Net activation
\rightarrow
Sparse Autoencoder
\rightarrow
interpretable latent concepts
}
$$

---

# 二十六、另外一篇非常新的「真的研究 diffusion components」

## Unveiling Concept Attribution in Diffusion Models

這篇提出：

$$
\boxed{\text{Component Attribution for Diffusion Models (CAD)}}
$$

它問的不是只：

> 哪一層存某個概念？

而是：

> 不同 diffusion model components 如何一起造成某個 concept？

而且會區分：

* positive contribution components
* negative contribution components

再透過 model editing / ablation 驗證。([arXiv][13])

它跟你的下一層想法很像：

```text
某一場大雨 prediction

哪些 U-Net blocks：
+ 加強這場雨？

哪些：
- 抑制這場雨？
```

這會比只找 top activated channel 更完整。

---

# 二十七、再來是「大氣領域」：你不能錯過這篇

## Mechanistic Interpretability Tool for AI Weather Models

Tempest et al., 2026。

它不是 diffusion，是 GraphCast。

但是研究方法跟你非常非常接近。

他們做：

$$
\boxed{
\text{internal latent representations}
}
$$

然後：

* PCA
* cosine similarity
* visualizing latent channels
* 找可能對應 meteorological features 的 latent directions

例如研究：

* mid-latitude waves
* specific humidity

而且目標就是：

> 找出 AI weather model 裡可能有 physical meaning 的 latent features。([arXiv][14])

---

# 二十八、這篇跟你的研究故事可以直接接起來

Computer Vision diffusion papers 告訴你：

> **Diffusion U-Net internal activations are meaningful representations.**

Weather interpretability paper 告訴你：

> **Internal latent representations of weather models may correspond to interpretable physical structures.**

你的問題就可以放在兩者交會：

$$
\boxed{
\text{What physically meaningful representations emerge inside a diffusion U-Net for precipitation downscaling?}
}
$$

這句其實已經很像 paper research question。

---

# 二十九、另一篇 Weather mechanistic interpretability 更猛

## Towards mechanistic understanding in a data-driven weather model: internal activations reveal interpretable physical features

MacMillan & Ouellette，2025 preprint。

他們把 Sparse Autoencoder 類方法用到 GraphCast internal activations，找到跟：

* tropical cyclones
* atmospheric rivers
* precipitation patterns
* seasonal/diurnal behavior
* geography
* sea ice

等現象相關的 internal features。

更重要的是他們還會：

$$
\boxed{\text{intervene on the feature}}
$$

例如修改 tropical-cyclone-related feature，看 forecast 怎麼改變。([arXiv][15])

這就是你非常值得學的：

```text
Diffusion literature
    ↓
怎麼抽 U-Net features

+

Weather mechanistic interpretability
    ↓
怎麼把 feature 跟物理 phenomena 對上

+

Causal intervention
    ↓
確認不是漂亮圖而已
```

---

# 三十、還有一篇跟你 application 幾乎直接重疊

## How far can we downscale? Resolution limits and physical interpretability of diffusion models for African precipitation

2026，**Machine Learning: Earth**。

這篇做的是：

$$
\boxed{
\text{precipitation downscaling}
+
\text{conditional diffusion}
+
\text{U-Net}
+
\text{physical interpretability}
}
$$

非常接近你的 application。([數字對象識別器][1])

---

# 三十一、但它的 XAI 跟你想做的不太一樣

它主要用：

$$
\boxed{\text{perturbation-based feature importance}}
$$

去問：

> Diffusion model 在不同環境下比較依賴哪個 atmospheric predictor？

例如它報告 model reliance 會依情境改變：

* rugged terrain / dry season：較依賴 low-level moisture / convergence
* monsoon 等情境：較依賴 mid-tropospheric dynamic forcing

藉此主張 diffusion framework 對 atmospheric predictors 有 context-dependent physical sensitivity。([數字對象識別器][1])

這是一篇你**一定值得引用的近鄰 paper**。

但它沒有完整回答你現在要問的：

> **U-Net 裡 128/256/384/512 個 latent channels 到底學到了什麼？**

所以你的 internal-feature analysis 還是另一條問題。

---

# 三十二、所以我會把 literature 分成三層

## A. 最直接支撐你的「Diffusion U-Net internal feature XAI」

最重要：

1. **Elucidating the representation of images within an unconditional diffusion model denoiser**
2. **Not All Diffusion Model Activations Have Been Evaluated as Discriminative Features**
3. **Revelio**
4. **Diffusion Hyperfeatures**
5. **DIFT**
6. **A Tale of Two Features**
7. **Plug-and-Play Diffusion Features**

這一群回答：

$$
\boxed{
Diffusion\ U-Net\ 裡真的存在 meaningful internal representations
}
$$

---

## B. 支撐「怎麼讓解釋具有因果性」

1. DAAM
2. Mechanistic Interpretability via Cross-Attention Interventions
3. CAD

這群告訴你：

$$
\boxed{
不要只 visualize
\rightarrow
還要 intervention / ablation
}
$$

---

## C. 支撐「怎麼變成 atmospheric physical interpretation」

1. Mechanistic Interpretability Tool for AI Weather Models
2. Towards mechanistic understanding in a data-driven weather model
3. African precipitation diffusion physical-interpretability paper

這群告訴你：

$$
\boxed{
latent\ feature
\rightarrow
meteorological\ phenomenon
}
$$

應該怎麼講才不只是 computer-vision visualization。

---

# 三十三、我認為你現在最合理的第一版方法

你不用一開始做得跟 SAE 那麼大。

先做一個：

$$
\boxed{\text{Diffusion U-Net Feature Explorer}}
$$

你的 inference 每次跑：

```python
model(...)
```

時，在 U-Net blocks 加 hook。

例如 conceptually：

```python
features = {}

def hook(name):
    def fn(module, inp, out):
        features[name] = out.detach().cpu()
    return fn
```

然後監測：

```text
Encoder:
72x112_block1
36x56_block1
18x28_block1
9x14_block1

Middle:
9x14_in0
9x14_in1

Decoder:
18x28_block2
36x56_block2
72x112_block2
```

---

# 三十四、你會得到一個完整的 feature tensor database

例如：

$$
F(d,m,t,l)
$$

其中：

* \(d\) = date
* \(m\) = ensemble member
* \(t\) = diffusion timestep
* \(l\) = U-Net layer

實際 feature：

$$
F_{d,m,t,l}
\in
\mathbb R^{C_l\times H_l\times W_l}.
$$

例如：

```text
2018-08-23
member 7
step 20
middle block

→ [512,14,9]
```

這就是你 XAI 的原料。

---

# 三十五、第一層分析：Spatial Feature Map

單一 channel：

$$
A_c\in\mathbb R^{14\times9}.
$$

resize：

$$
14\times9
\rightarrow112\times72.
$$

畫：

```text
Ground Truth
Prediction
Prediction - Coarse
Feature channel 183
Elevation
Mask
```

放在一起。

問：

> Feature 183 的 spatial activation 是否跟 model correction 出現在相同地方？

---

# 三十六、第二層：Channel activation profile

對每個 channel spatial average：

$$
\phi_c
=
Mean_{H,W}(A_c).
$$

得到：

$$
\phi
=
[\phi_1,\ldots,\phi_{512}].
$$

然後：

```text
Top activated channels:
#183
#41
#309
#87
...
```

再找這些 channel 在整個 dataset 的 highest activating cases。

這直接受到 Kadkhodaie 等人的方法啟發。([arXiv][2])

---

# 三十七、第三層：PCA

把很多日期：

$$
\phi^{(1)},\phi^{(2)},...,\phi^{(N)}
$$

組成：

$$
N\times512.
$$

PCA：

$$
512
\rightarrow
2.
$$

畫：

```text
             PC2
              ↑
 heavy rain ●●
           ●
              ○○ dry
        ○
──────────────→ PC1
```

再依：

* rainfall intensity
* season
* terrain regime
* weather event

上色。

問：

> U-Net latent representation 是否自然把不同 precipitation regimes 分開？

---

# 三十八、第四層：Diffusion timestep evolution

對同一天：

```text
σ=80
 ↓
Middle feature

σ=20
 ↓
Middle feature

σ=5
 ↓

σ=1
 ↓

σ=.1
 ↓

σ=.002
```

比較：

$$
cos(
\phi_t,\phi_{t+1}
).
$$

以及 PCA trajectory。

可能畫出：

```text
High noise
 ●
  \
   ●
     \
      ●────●────● Low noise
```

問：

> rainfall physical representation 是在 denoising 的什麼階段形成？

這是 Diffusion Hyperfeatures 特別能支撐你的地方。([NeurIPS 會議論文集][6])

---

# 三十九、第五層：Encoder vs Decoder

比較：

```text
Encoder
Middle
Decoder
```

例如對 heavy-rain vs normal-day classification：

使用 frozen feature：

$$
\phi_l
$$

訓練非常簡單的：

$$
LogisticRegression(\phi_l).
$$

如果：

```text
Encoder L1     60%
Encoder L3     76%
Middle         91%
Decoder L2     88%
Final Decoder  73%
```

你可以說：

> Middle representation contains stronger information associated with this event class.

而不是只靠眼睛判斷 PCA。

Revelio 也使用 lightweight classifiers 去 probe diffusion representations。([開放存取計算機視覺基金會][5])

---

# 四十、第六層：最重要的 Causal Ablation

這會是我最推薦你的核心。

假設找到：

$$
channel\ 183.
$$

Original inference：

$$
F_{183}
$$

照常。

Intervention：

$$
F_{183}=0.
$$

然後保持：

$$
\boxed{\text{same date + same seed}}
$$

重新 generation。

這點非常重要，因為 Diffusion 本身有 randomness。

如果 seed 不固定：

$$
P_A-P_B
$$

可能只是不同 noise。

固定 seed 後：

$$
\boxed{
\Delta P
\approx
\text{intervention effect}
}
$$

會乾淨很多。

這跟 2026 mechanistic diffusion paper 的 causal comparison logic 很吻合。([ACL Anthology][12])

---

# 四十一、你最後可以得到這樣一套論文圖

我覺得非常合理：

```text
Figure 1
Your diffusion architecture
+
XAI hooks

Figure 2
Layer-wise U-Net activation maps

Figure 3
PCA of latent representations
Encoder / Middle / Decoder

Figure 4
Feature evolution across diffusion timesteps

Figure 5
Top-activating weather cases
for selected latent features

Figure 6
Feature ↔ physical variable association

Figure 7
Channel/feature ablation
Original vs intervened prediction

Figure 8
Quantitative causal effect
ΔMAE / Δrainfall / regional change
```

---

# 四十二、你的 research story 可以變成

不是：

> 「我幫 Diffusion 畫 heatmap。」

而是：

> **We investigate the internal representations learned by a conditional diffusion U-Net for precipitation downscaling. Inspired by recent work showing that diffusion denoisers contain meaningful layer-, channel-, and timestep-dependent representations, we extract multiscale U-Net activations throughout the denoising trajectory, characterize their spatial and physical associations, and causally validate selected representations through controlled feature interventions under fixed stochastic seeds.**

這個研究定位就成熟很多。

---

# 四十三、你的方法和現有文獻的拼接關係

可以畫成：

```text
Diffusion Feature Literature
DIFT
Hyperfeatures
A Tale of Two Features
Generic Diffusion Features
        │
        │ 告訴你
        ▼
U-Net intermediate activations
contain useful representation
        │
        ▼
Kadkhodaie et al.
Revelio
        │
        │ 告訴你
        ▼
channels / latent directions
can become interpretable features
        │
        ▼
Mechanistic Diffusion XAI
DAAM / intervention / CAD
        │
        │ 告訴你
        ▼
Need causal validation
not just visualization
        │
        ▼
Weather Mechanistic Interpretability
GraphCast papers
        │
        │ 告訴你
        ▼
Connect latent feature
to physical phenomena
        │
        ▼
              YOUR WORK
──────────────────────────────────
Conditional precipitation
Diffusion U-Net

Encoder
↓
Middle
↓
Decoder
×
Diffusion timestep
×
Weather event
        │
        ▼
physical latent representation
        │
        ▼
causal ablation
```

這就是我目前認為你**最值得發展的 XAI 主線**。

---

# 四十四、如果你現在時間有限，只先看 5 篇

照這個順序：

### ① 最先看

**Elucidating the representation of images within an unconditional diffusion model denoiser**

因為跟你的 network 結構最接近：

$$
\boxed{\text{Diffusion + U-Net + internal channels}}
$$

而且沒有太依賴 text cross-attention。([arXiv][2])

### ②

**Not All Diffusion Model Activations Have Been Evaluated as Discriminative Features — NeurIPS 2024**

學：

$$
\boxed{\text{到底該抽哪些 activation}}
$$

以及 feature extraction / visualization methodology。([NeurIPS 會議論文集][3])

### ③

**Diffusion Hyperfeatures — NeurIPS 2023**

學：

$$
\boxed{
Layer\times timestep
}
$$

而不是只看一層。([NeurIPS 會議論文集][6])

### ④

**Revelio — ICCV 2025**

學：

$$
\boxed{
Sparse / interpretable latent features
}
$$

和 probing。([開放存取計算機視覺基金會][5])

### ⑤

**Mechanistic Interpretability Tool for AI Weather Models**

雖然不是 diffusion，但它幫你把：

$$
\boxed{\text{latent features}}
$$

翻譯成：

$$
\boxed{\text{meteorological scientific interpretation}}
$$

這一步對你的 domain 非常重要。([arXiv][16])

---

## 還要再加一篇和你 application 最接近的

**How far can we downscale? Resolution limits and physical interpretability of diffusion models for African precipitation**

因為 reviewer 很可能會覺得這篇是你的直接 related work：

$$
\boxed{
precipitation
+
downscaling
+
diffusion
+
physical interpretability
}
$$

但你可以很清楚地區分：

> 它主要在 **input predictor perturbation / feature importance** 層級研究 physical interpretability；你想進一步研究 **denoising U-Net internal representation itself — layers, channels and diffusion timesteps**。([數字對象識別器][1])

這個 gap 是目前我認為你最值得往下挖的地方，而且比單純在你的 Diffusion 上套 Grad-CAM 更有研究味道。

[1]: https://doi.org/10.1088/3049-4753/ae7114?utm_source=chatgpt.com "How far can we downscale? Resolution limits and physical interpretability of diffusion models for African precipitation - IOPscience"
[2]: https://arxiv.org/abs/2506.01912 "Elucidating the representation of images within an unconditional diffusion model denoiser"
[3]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/633780c1344d0c95e4d2dd3431fe08d9-Abstract-Conference.html?utm_source=chatgpt.com "Not All Diffusion Model Activations Have Been Evaluated as Discriminative Features"
[4]: https://github.com/darkbblue/generic-diffusion-feature?utm_source=chatgpt.com "GitHub - Darkbblue/generic-diffusion-feature: Official implementation of NeurIPS'24 paper Not All Diffusion Model Activations Have Been Evaluated as Discriminative Features · GitHub"
[5]: https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Revelio_Interpreting_and_leveraging_semantic_information_in_diffusion_models_ICCV_2025_paper.html?utm_source=chatgpt.com "ICCV 2025 Open Access Repository"
[6]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/942032b61720a3fd64897efe46237c81-Abstract-Conference.html?utm_source=chatgpt.com "Diffusion Hyperfeatures: Searching Through Time and Space for Semantic Correspondence"
[7]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/0503f5dce343a1d06d16ba103dd52db1-Abstract-Conference.html?utm_source=chatgpt.com "Emergent Correspondence from Image Diffusion"
[8]: https://sd-complements-dino.github.io/?utm_source=chatgpt.com "A Tale of Two Features: Stable Diffusion Complements DINO for Zero-Shot Semantic Correspondence"
[9]: https://openaccess.thecvf.com/content/CVPR2023/html/Tumanyan_Plug-and-Play_Diffusion_Features_for_Text-Driven_Image-to-Image_Translation_CVPR_2023_paper.html?utm_source=chatgpt.com "CVPR 2023 Open Access Repository"
[10]: https://openaccess.thecvf.com/content/CVPR2023/papers/Tumanyan_Plug-and-Play_Diffusion_Features_for_Text-Driven_Image-to-Image_Translation_CVPR_2023_paper.pdf?utm_source=chatgpt.com "Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation"
[11]: https://aclanthology.org/2023.acl-long.310/?utm_source=chatgpt.com "What the DAAM: Interpreting Stable Diffusion Using Cross Attention - ACL Anthology"
[12]: https://aclanthology.org/2026.findings-acl.1265/?utm_source=chatgpt.com "Mechanistic Interpretability of Text-to-Image Diffusion Models via Cross-Attention Interventions - ACL Anthology"
[13]: https://arxiv.org/abs/2412.02542v2?utm_source=chatgpt.com "[2412.02542v2] Unveiling Concept Attribution in Diffusion Models"
[14]: https://arxiv.org/html/2604.20467v1?utm_source=chatgpt.com "Mechanistic Interpretability Tool for AI Weather Models"
[15]: https://arxiv.org/abs/2512.24440?utm_source=chatgpt.com "Towards mechanistic understanding in a data-driven weather model: internal activations reveal interpretable physical features"
[16]: https://arxiv.org/abs/2604.20467?utm_source=chatgpt.com "Mechanistic Interpretability Tool for AI Weather Models"
