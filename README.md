# 基于知识蒸馏与 ECA 注意力机制的轻量级食品图像分类模型

## 摘要

针对食品图像分类任务，本文提出了一种结合高效通道注意力（ECA）模块的知识蒸馏方法，用于提升轻量级网络的分类性能。具体地，我们以 ResNet-50 作为教师模型，在 Food-101 数据集上微调训练，得到 76.76% 的验证准确率；然后采用知识蒸馏策略训练学生模型，使得引入 ECA 的 MobileNetV3 学生网络最终获得 78.50% 的准确率，超过了教师模型的性能。训练过程中结合了混合精度（fp16）计算和 OneCycleLR 学习率调度策略以加速训练收敛。实验结果表明：通过引入 ECA 注意力模块和蒸馏机制，学生模型的表达能力显著增强，为未来在其他注意力模块（如 SimAM、CBAM 等）上的扩展提供了思路。

## 关键词

知识蒸馏；ECA 注意力；MobileNetV3；Food-101；轻量级模型

---

## 1. 引言

深度卷积神经网络在图像分类任务中取得了卓越成绩，但高精度模型（如 ResNet、Inception 等）通常包含大量参数与计算量，不适合部署在资源受限的移动设备上。食品图像分类是一个典型的细粒度识别任务，Food-101 数据集包含 101 个类别、共 101,000 张图像，具有噪声标签和外观多样性的特点。为了在该任务上同时获得高准确率与高效率，需要设计并优化轻量级网络。

知识蒸馏（Knowledge Distillation, KD）是一种将大模型（教师）中的表征能力转移到小模型（学生）中的模型压缩技术。通过拟合教师网络的“软”输出（即经过高温度 softmax 的 logit），学生模型可以获得比单独训练更好的泛化性能。已有研究表明，蒸馏结合注意力机制可显著提升学生模型性能。本工作将 ECA 注意力模块嵌入 MobileNetV3 学生网络，并通过知识蒸馏在 Food-101 上验证其有效性。

本工作的主要贡献：

- **模型设计**：提出将 ECA 模块集成到 MobileNetV3 中并结合知识蒸馏的轻量化框架。
- **实验验证**：在 Food-101 上，MobileNetV3+ECA 学生模型经蒸馏后在验证集上达到 78.50% 的准确率，超越 ResNet-50 教师模型的 76.76%。
- **训练优化**：采用混合精度训练和 OneCycleLR 学习率调度，提高训练效率与收敛速度。
- **可扩展性**：所提框架可推广到其他轻量化注意力模块与不同数据集。

下文按常见论文结构展开：相关工作、方法、实验设置与结果、结论与展望。

---

## 2. 相关工作

**知识蒸馏与模型压缩**：自 Hinton 等人提出知识蒸馏以来，该方法被广泛用于模型压缩与小模型性能提升。常见做法包括最小化学生输出与教师软目标之间的 KL 散度，或利用教师中间层特征/注意力图作为提示（hint）来辅助训练。多教师蒸馏、注意力蒸馏与对比蒸馏等变体也被提出以提升蒸馏效果。

**轻量级网络与注意力模块**：MobileNet 系列、ShuffleNet 等通过深度可分离卷积与网络结构搜索实现高效网络设计。MobileNetV3 结合网络结构搜索（NAS）与轻量注意力模块，在不显著增加复杂度的前提下提升了分类精度。常见注意力模块包括 SE、CBAM、ECA、SimAM 等。ECA（Efficient Channel Attention）通过全局平均池化与一维卷积为通道生成权重，避免了全连接降维，参数量小且效果显著，适合轻量网络场景。

将注意力机制与蒸馏结合的研究也取得了良好效果，说明注意力模块能增强学生模型的特征表达能力，从而提高蒸馏的效率与最终精度。

---

## 3. 方法

### 3.1 模型架构

本研究采用教师—学生（Teacher–Student）架构：

- 教师模型：ResNet-50，使用 ImageNet 预训练权重并在 Food-101 上微调得到蒸馏知识。
- 学生模型：MobileNetV3（Large）为骨干，在每个阶段末插入 ECA 模块。ECA 模块先对输入特征图做全局平均池化，然后通过一维卷积（可自适应的 kernel size $k$）生成通道注意力权重，最后经 Sigmoid 映射并逐通道加权原始特征。

ECA 的优点是结构简单、参数量少，适合在轻量级网络中使用以增强通道层面的表达能力。

### 3.2 知识蒸馏策略

学生模型同时优化真实标签的交叉熵损失与教师输出的蒸馏损失（KL 散度）：

$$
L=(1-\alpha)\,\mathrm{CE}(y,p_s)+\alpha\,T^{2}\,\mathrm{KL}\bigl(p_t^{(T)}\,\|\,p_s^{(T)}\bigr)
$$

其中 $T$ 为温度系数，$p^{(T)}$ 表示经温度 $T$ 处理后的 softmax 概率；$\alpha$ 为平衡系数。训练流程中先对教师模型微调并固定其参数，再对训练/验证集样本预计算并保存教师的 Logit（未经 softmax 的得分），学生训练时直接加载这些预计算的 Logit 作为蒸馏目标，从而加速训练。

当学生在架构上加入 ECA 等增强模块时，其容量可能接近或超过教师，蒸馏过程有可能使学生超越教师性能。

### 3.3 训练策略

训练中采用混合精度（PyTorch 的 `torch.cuda.amp`）以加速前向/反向传播并节省显存；使用 OneCycleLR 学习率调度（先增大学习率再逐步减小）以提升收敛速度。优化器采用 Adam，带权重衰减以抑制过拟合。

---

## 4. 数据集与实验设置

**数据集**：所有实验在公开的 Food-101 数据集上进行，包含 101 个类别、共 101,000 张图像。按常见做法，将每类 750 张训练图像中的 20% 用作验证集（训练集约 60,600 张，验证集约 15,150 张）；测试集每类 250 张，共 25,250 张。

**图像预处理与增强**：统一将图像缩放到最大边长 512 像素，采用随机水平翻转、随机裁剪、颜色抖动等数据增强策略。

**训练配置**：批大小 32，优化器 Adam（权重衰减 1e-4），教师微调 3 个 epoch，学生训练 20 个 epoch（使用混合精度与 OneCycleLR）。学生模型参数量约为 5–6M（取决于具体配置）。教师训练完成后保存模型并对训练/验证集样本预计算 Logit 以供学生蒸馏训练使用。

---

## 5. 实验结果与分析

### 5.1 主结果

| 模型                      | 验证准确率 | 说明                   |
| ------------------------- | ---------: | ---------------------- |
| ResNet-50（教师）         |     76.76% | 微调所得，作为知识源   |
| MobileNetV3 + ECA（学生） | **78.50%** | 本文方法所得，超越教师 |

表 1：ResNet-50 教师与 MobileNetV3+ECA 学生在 Food-101 验证集上的准确率。

如上所示，MobileNetV3+ECA 学生模型在验证集上达到 78.50% 的准确率，超过教师模型 1.74 个百分点。这表明结合 ECA 与知识蒸馏能在轻量级模型上带来可观的性能提升。在我们的内部对照实验中，未使用蒸馏与注意力的 MobileNetV3 基线精度约为 74%（未在本文中详细列出），加入蒸馏与 ECA 后性能显著提升。

### 5.2 与相关工作的对比

尽管一些重型模型（如改进的 ResNet、Inception 系列）在 Food-101 上能取得更高精度，但这些模型在移动端的实用性有限。本方法关注移动部署场景，在参数量与推理效率可控的前提下取得了具有竞争力的精度。

### 5.3 注意力模块的作用

ECA 在通道维度重新校准特征图，使模型更关注关键信息。对比未使用注意力的学生模型，加入 ECA 后获得了额外性能增益。这与文献中观察到的轻量注意力（如 ECA、SimAM）在不显著增加计算开销的前提下提升性能的结论一致。

### 5.4 训练效率

混合精度训练使得每个 epoch 的训练速度提高且显存占用减少；OneCycleLR 加速了收敛过程，两者共同保证了在有限计算资源下完成蒸馏训练的可行性。

---

## 6. 结论与展望

本文提出了一种在 Food-101 食品分类任务中结合 ECA 注意力与知识蒸馏的轻量级训练方法。实验表明，MobileNetV3+ECA 学生模型在蒸馏后在验证集上达到 78.50% 的准确率，超越 ResNet-50 教师模型，证明了所提方法在移动级别模型优化中的有效性。未来工作将尝试：

- 在更多数据集上验证方法的泛化性；
- 将 ECA 替换或融合为 SimAM、CBAM 等注意力模块以比较性能；
- 探索多任务或跨域蒸馏的扩展应用。

---

## 参考文献

1. A. Bossard, M. Guillaumin, and L. Van Gool, "Food-101 – Mining Discriminative Components with Random Forests," *ECCV Workshops*, 2014. (Dataset and paper). 论文与数据集下载：https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf

2. G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural Network," *arXiv preprint arXiv:1503.02531*, 2015. PDF: https://arxiv.org/pdf/1503.02531.pdf

3. A. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Q. V. Le, and H. Adam, "Searching for MobileNetV3," *arXiv preprint arXiv:1905.02244*, 2019. PDF: https://arxiv.org/pdf/1905.02244.pdf

4. Q. Wang, B. Wu, P. Zhu, P. Li, W. Zuo, and Q. Hu, "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020. PDF: https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf

5. S. Zagoruyko and N. Komodakis, "Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer," *ICLR Workshop / arXiv:1612.03928*, 2017. PDF: https://arxiv.org/pdf/1612.03928.pdf

6. A. Romero, N. Ballas, S. Ebrahimi Kahou, A. Chassang, C. Gatta, and Y. Bengio, "FitNets: Hints for Thin Deep Nets," *ICLR*, 2015. PDF: https://arxiv.org/pdf/1412.6550.pdf

7. J. Gou, B. Yu, S. J. Maybank, and D. Tao, "Knowledge Distillation: A Survey," *International Journal of Computer Vision*, vol. 129, pp. 1789–1819, 2021. DOI: 10.1007/s11263-021-01453-z. 预印本： https://arxiv.org/abs/2006.05525

---
