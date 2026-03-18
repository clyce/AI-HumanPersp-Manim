# 勘误与技术说明

本文档记录项目中已发现的事实性错误、技术缺陷和教学叙事问题，以及对应的修正方案。

---

## 1. `StandardBoltzmannMachine` 实现存在根本性缺陷

**文件**: `src/hopfield_tools.py`

### 1.1 随机种子管理错误

**问题**: `sample_unit()` 方法中使用 `np.random.seed(RANDOM_SEED + unit_index)` 在每次采样时重置全局随机种子。这导致同一个 unit 无论当前网络状态如何，总是产生相同的采样结果，从根本上破坏了 Gibbs 采样链的马尔可夫性质。

**影响**: Gibbs 采样无法正确探索状态空间，网络的"随机性"名存实亡。

**修正**: 使用实例级 `np.random.Generator` 对象，仅在 `__init__` 中初始化一次种子，后续让 RNG 自然演化。

### 1.2 训练过程中的种子重置

**问题**: `train()` 和 `train_with_convergence()` 方法中使用 `np.random.seed(RANDOM_SEED + epoch)` 在每个 epoch 开始时重置种子。结合 1.1 的 per-unit 种子重置，整个训练过程的随机性完全被破坏。

**修正**: 移除所有 per-epoch 和 per-unit 的种子重置，仅保留 `__init__` 中的初始种子。

### 1.3 训练温度为零

**问题**: SC08 中设置 `temperature = 0.0`，使得 Contrastive Divergence (CD) 学习算法的负相位变为完全确定性的。CD 算法的理论基础要求通过随机采样来近似模型分布的梯度；温度为零时，采样退化为阈值函数，产生严重偏差的梯度估计。

**对比**: Hopfield 网络使用 Hebbian 规则（解析解/闭式解），不依赖随机采样，因此不受此问题影响。

**修正**: 训练时使用 `temperature > 0`（如 1.0），推理时可使用退火策略逐步降低温度。

---

## 2. `shared.py` 测试配置不合理

**文件**: `SlideComponents/S02_RAMvsCAM/shared.py`

### 2.1 注释与实际数据不符

**问题**: 注释声明"12个不重复的6位二进制牌组"，但实际只定义了 8 个模式。

**修正**: 更新注释以匹配实际数据。

### 2.2 提示位组合完全相同

**问题**: `CUE_COMBINATIONS_3BIT` 的所有 8 个条目均为 `[0, 1, 2]`，意味着每次测试都使用完全相同的前3位作为提示、推导后3位。这无法测试模型在不同提示条件下的泛化能力。

**修正**: 恢复多样化的提示位组合，让不同模式使用不同的提示位。

---

## 3. `Compare.py` 引用不存在的类

**文件**: `Compare.py`

**问题**: `from src.hopfield_tools import HashExtendedHopfield` 引用了一个在当前代码中不存在的类，运行即抛出 `ImportError`。

**修正**: 移除相关引用和测试代码，或后续单独实现该类。

---

## 4. SC08 概念性定位错误（已修正）

**文件**: `SlideComponents/S02_RAMvsCAM/SC08_SubCons.py`

### 原始定位
SC08 将 Boltzmann Machine 定位为"改进版的 Hopfield Network"，用同一个 cued recall 任务评测两者，并比较准确率。

### 问题
- **HF 是 CAM（Content Addressable Memory）**：给定部分输入，收敛到最近的存储模式。Cued recall 正是它的本职。
- **BM 是生成模型**：它学习 P(data)，强项是从学到的分布中生成新样本，而非精确回忆。
- 在 cued recall 任务上比较两者，BM 的随机性（Gibbs 采样）反而是劣势，导致 BM (66%) 低于 HF (70%)，与"BM 是改进版"的叙事矛盾。
- "10% 谬论"类比也存在逻辑跳跃，与新的生成模型定位无关。

### 修正
- SC08 重写为"从记忆到理解"：强调 HF(CAM) → BM(Generative) 的范式转换
- 删除 cued recall 准确率比较，替换为生成演示（BM 自由生成样本 vs 训练数据对比）
- 删除"10% 谬论"段落
- RBM 从独立章节降级为一句话过渡

---

## 5. Hopfield 网络容量问题

**背景知识**: 经典 Hopfield 网络的存储容量约为 0.14N（N 为神经元数量）。对于 N=6 的网络，理论容量约为 0.84 个模式。

**当前配置**: 8 个模式存储在 6 个神经元的网络中，远超理论容量，导致 SC07 中观察到的记忆混乱是**预期行为**，而非实现错误。

**教学意义**: 这恰好是 SC07 要展示的要点——Hopfield 网络容量有限。SC08 不再试图用 BM 来"修复"这个问题，而是引出一个完全不同的范式：从回忆到生成。
