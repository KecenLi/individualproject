# Project Progress Update: Infrastructure & Analysis Results

## 💡 核心回顾：Week 1 "被忽略"的工程底座
*（这部分是在上次会议中因沟通问题未充分展示的基础设施工作，它们是当前所有实验结果的基石）*

在生成图表之前，我构建了一个**通用的评估中间件 (Middleware)**，解决了学术界三大库 (RobustBench, Advex-UAR, OpenOOD) 之间的数据协议冲突。这不仅仅是“跑通代码”，而是保证了实验的理论正确性。

1.  **标准化中间件 (Normalization Middleware)**:
    *   **问题**: `Advex-UAR` 攻击库强制要求 ImageNet 标准化输入，而 `RobustBench` 模型期望原始 [0,1] 输入。直接对接会导致攻击失效或模型由于输入分布错误而性能崩塌。
    *   **解决**: 我在 `src/loader.py` 和 `src/perturber.py` 中实现了一个透明转换层。它能自动识别模型类型，并在攻击发生前/后进行逆标准化，确保攻击是在正确的像素空间进行的。
    *   **意义**: 保证了 Week 2 看到的 "AutoAttack" 和 "DeepG" 结果是真实的物理/几何攻击，而不是数据预处理错误的假象。

2.  **异构架构兼容层 (Architecture Compatibility Shim)**:
    *   **问题**: NAC (OpenOOD) 强依赖于特定的 Layer 名称（如 `layer4`），这使得它难以直接应用于 Vision Transformers (ViT) 或对抗训练的 WideResNet。
    *   **解决**: 开发了 `src/official_nac.py` 中的 `patched_get_intr_name` 钩子。
    *   **现状**: **ViT 的评估脚本 (`hpc_job_rb_vit.sh`) 已经就绪**。虽然目前的报告主要基于 ResNet，但我们随时可以一键切换到 ViT 或大型模型，无需重写代码。

3.  **几何攻击集成 (DeepG)**:
    *   **工作**: 编译并集成了 `DeepG` 库（C++ backend），使其能像普通 PyTorch transform 一样调用。
    *   **现状**: 这部分能力目前仅用于 Phase 3 的简单旋转/平移测试，但**基础设施已支持复杂的几何对抗攻击**，为后续深入研究几何鲁棒性预留了接口。

---

## 📊 Week 2 实验结果：NAC 的特性画像
*（基于上述基础设施产生的最新分析）*

### 1. 现实 vs 理想：APS 的双面性
我们澄清了 NAC 在 "有 OOD 验证集 (APS)" 和 "无 OOD 验证集 (Non-APS)" 下的巨大差异。
*   **APS (上界)**: 在 Near-OOD (CIFAR-100) 上表现极佳，但存在过拟合嫌疑。
*   **Non-APS (现实)**: 依然能很好地检测 Far-OOD (SVHN, Texture)，但在 Near-OOD 上能力回落。
*   **结论**: 我们必须诚实地使用 **Non-APS** 结果作为鲁棒性评估基准。

### 2. NAC 的敏感度光谱 (Sensitivity Spectrum)
通过 OODRobustBench 全量扫测，我们发现 NAC 并不是“万能检测器”，它有极强的偏科：
*   **高敏感 (Easy to Detect)**: 结构破坏与高频噪声 (Impulse Noise, Pixelate, L-inf Adversarial)。NAC AUROC 可达 **0.90+**。
*   **低敏感 (Hard to Detect)**: 全局光照与低频变化 (Brightness, Fog, Snow)。NAC AUROC 仅 **0.50-0.60** (接近随机)。
*   **洞察**: NAC 本质上是在检测“卷积特征的破坏”。如果扰动（如亮度变化）可以通过卷积层的线性变换适应，NAC 就涵盖不了这种异常。

### 3. 鲁棒模型的悖论 (The Robustness Paradox)
*（Week 1 提到的现象在 Week 2 得到了进一步数据支持）*
*   **现象**: 对抗训练模型 (Robust WideResNet) 在面对攻击时，其内部神经元激活的变化幅度**远小于**标准模型。
*   **结果**: 这导致 NAC 反而**更难**检测出针对鲁棒模型的攻击（AUROC 0.52 vs 0.90）。
*   **意义**: 这是一个极其反直觉但也合理的发现——**“鲁棒性”本身就是为了让内部特征在攻击下保持稳定，而这种稳定恰恰欺骗了基于“特征异动”的检测器 (NAC)**。

---

## 🚀 下一步：已就绪但未充分利用的能力 (Ready-to-Deploy)

你在代码库中可能会看到以下尚未出现在报告中的模块，它们是我们下一步的扩展方向：

1.  **Vision Transformers (ViT)**:
    *   脚本: `scripts/hpc_job_rb_vit.sh`
    *   计划: 对比 CNN (ResNet) 与 Transformer (ViT) 的 NAC 覆盖特性差异。ViT 的全局注意力机制可能会对“全局光照”类扰动有不同的反应。

2.  **多层集成 (Ensemble NAC)**:
    *   脚本: `run_official_aps_ensemble.py`
    *   计划: 目前主要看 `layer4`。基础设施已支持同时通过 `layer1+2+3` 计算 NAC，这可能弥补单层检测的盲区（例如 layer1 可能对纹理更敏感）。

3.  **高效计算 (Efficient NAC)**:
    *   模块: `src/nac_efficient.py`
    *   计划: 优化大规模评估时的内存占用。
