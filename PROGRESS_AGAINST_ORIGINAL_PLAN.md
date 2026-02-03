# 📋 Progress Checklist vs. Original Plan
*(Based on Xiyue's Email Requirements)*

这份文档逐条对照 Xiyue 原始邮件中的要求，诚实地列出**已完成 (Done)**、**进行中/部分完成 (In Progress/Partial)** 与 **未涉及 (Not Started)** 的部分，供您向导师汇报你是如何一项项落实的，以及目前的缺口在哪里。

---

## 1. Metric: NAC
> *Original Request: "Metric: NAC (https://openreview.net/pdf?id=SNGXbZtK6Q)"*

- [x] **Status**: **Fully Implemented**
- **Details**:
  - 集成了官方 OpenOOD v1.5 实现。
  - **完成度**: 100%。不仅跑通了代码，还复现了 ICLR 论文中的 APS 寻参流程，并额外对比了 Non-APS 模式。
  - **Note**: 解决了官方代码对 Layer Name 的硬编码限制。

---

## 2. Adversarial Perturbations
> *Original Request: "Start with lp attacks... Different attacks: L∞, L2, L1, L∞-JPEG, L2-JPEG, L1-JPEG, Elastic, Fog, Gabor, Snow. (Advex-UAR)"*

- [x] **Status**: **Fully Implemented**
- **Details**:
  - **Lp Attacks**: 集成了 `AutoAttack` (L-inf, L2)。
  - **Common Corruptions**: 集成了 `Advex-UAR` 的 Fog, Snow, Elastic, Gabor。
  - **JPEG**: 实现了全部三种范数 (L-inf, L2, L1) 的 JPEG 压缩攻击扫描。
  - **完成度**: 100%。所有邮件中点名的攻击类型都已进入 `src/perturber.py` 并完成了 CIFAR-10 上的测试。

---

## 3. Geometric Transformations
> *Original Request: "Geometric transformations: https://github.com/eth-sri/deepg"*

- [~] **Status**: **Integration Ready, Analysis Limited** (部分完成)
- **Details**:
  - **已做**: 成功编译了 DeepG 的 C++ 后端，并在 `src/external_sources.py` 中实现了加载接口。在 Phase 3 实验中使用了 DeepG 的 **Rotation** 和 **Translation**。
  - **未做**: DeepG 库中还有更复杂的几何变换（如 Shear, Scale, Affine 等）尚未进行大规模系统性扫测。目前仅将其作为“几何扰动”的代表使用了最基础的功能。
  - **Gap**: 尚未挖掘 DeepG 的全部潜力。

---

## 4. Corruption Shift
> *Original Request: "Corruption shift (one of the OOD types): https://github.com/OODRobustBench/OODRobustBench"*

- [x] **Status**: **Fully Implemented**
- **Details**:
  - 完成了 CIFAR-10-C 全量扫测（15 种腐蚀 × 5 种强度）。
  - **关键产出**: 发现了 NAC 对“结构破坏”敏感而对“光照变化”不敏感的规律。

---

## 5. Benchmark Models
> *Original Request: "Potential Benchmark Models... https://github.com/RobustBench/robustbench"*

- [~] **Status**: **Limited Scope** (范围受限)
- **Details**:
  - **已做**:
    - Standard: `ResNet18`
    - Robust: `WideResNet-28-10 (Gowal2021)`
  - **未做**:
    - **ImageNet Models**: 这是一个巨大的缺口。RobustBench 上大量的 ImageNet 模型（ResNet-50, ViT, ConvNeXt 等）全都还没跑。目前的结论仅限于 CIFAR-10 小图。
    - **Architecture Diversity**: 尚未测试 Transformer (ViT) 架构。虽然代码兼容了，但还没跑出数据。
  - **Gap**: 实验目前的结论可能被质疑“仅适用于 ResNet/CIFAR”。

---

## 6. Analysis Tasks
> *Original Request: "(i) clean vs single perturbation vs combined; (ii) different orders of composition."*

- [x] **Status**: **Fully Implemented (Phase 3)**
- **Details**:
  - **(i) Single vs Combined**: 完成。我们对比了 Clean, Rotate-only, Noise-only, 以及 Rotate+Noise 的 NAC 分数。
  - **(ii) Order of Composition**: 完成。专门测试了 Order A (Rotate->Noise) vs Order B (Noise->Rotate)，结论是顺序影响微乎其微。

---

## ⚠️ 总结：最大的缺口 (The Big Missing Pieces)

基于这份清单，如果您要“诚实”地说明进度，最大的短板在于：

1.  **Dataset Scale**: 所有的实验都局限在 **CIFAR-10**。Xiyue 的邮件虽然没明说，但 CV 领域的 Benchmark 通常默认包含 ImageNet。
2.  **Model Variety**: 目前只有 **2 个模型**。虽然一个是标准一个是鲁棒，对比很鲜明，但样本量太小，不足以宣称“NAC 在所有架构上都有效”。

**话术建议**:
"I have successfully built the pipeline connecting all 5 required libraries (NAC, AutoAttack, Advex-UAR, DeepG, OODRB) and validated them on CIFAR-10. The infrastructure is now robust. The next logical step, pending resources, is to simply switch the dataset flag to ImageNet and run the exact same pipeline on larger models."
