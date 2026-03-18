"""
SC05-SC08、Compare.py 共享的配置数据
统一牌组模式、提示位组合配置，以及 SC08 四级实验用数据
"""

import numpy as np
from collections import Counter

# 8 个不重复的 6 位二进制牌组
TRAINING_PATTERNS = [
    [0, 0, 0, 0, 1, 1],  # 牌组1: 2个1，4个0
    [0, 0, 1, 0, 0, 1],  # 牌组2: 2个1，4个0
    [0, 1, 0, 1, 0, 0],  # 牌组3: 2个1，4个0
    [0, 1, 1, 1, 1, 0],  # 牌组4: 4个1，2个0
    [1, 0, 0, 1, 1, 1],  # 牌组5: 4个1，2个0
    [1, 0, 1, 0, 1, 1],  # 牌组6: 4个1，2个0
    [1, 1, 0, 0, 0, 0],  # 牌组7: 2个1，4个0
    [1, 1, 1, 1, 1, 0],  # 牌组8: 5个1，1个0
]

# 多样化的 3 位提示位组合：不同牌组使用不同的提示位置
CUE_COMBINATIONS_3BIT = [
    [0, 1, 2],  # 牌组1: 提示位 0,1,2 → 推导位 3,4,5
    [0, 2, 4],  # 牌组2: 提示位 0,2,4 → 推导位 1,3,5
    [1, 3, 5],  # 牌组3: 提示位 1,3,5 → 推导位 0,2,4
    [0, 3, 4],  # 牌组4: 提示位 0,3,4 → 推导位 1,2,5
    [2, 4, 5],  # 牌组5: 提示位 2,4,5 → 推导位 0,1,3
    [0, 1, 5],  # 牌组6: 提示位 0,1,5 → 推导位 2,3,4
    [1, 2, 3],  # 牌组7: 提示位 1,2,3 → 推导位 0,4,5
    [0, 4, 5],  # 牌组8: 提示位 0,4,5 → 推导位 1,2,3
]

def get_cue_indices(pattern_idx, cue_size=None):
    """
    获取指定模式的提示位组合

    Args:
        pattern_idx: 模式索引 (0-based)
        cue_size: 未使用，保留接口兼容

    Returns:
        list: 提示位索引列表
    """
    return CUE_COMBINATIONS_3BIT[pattern_idx % len(CUE_COMBINATIONS_3BIT)]

def calculate_cued_accuracy(original_pattern, recalled_pattern, cue_indices):
    """
    计算仅针对未被提示部分的准确率
    """
    if len(original_pattern) != len(recalled_pattern):
        return 0.0

    uncued_indices = [i for i in range(len(original_pattern)) if i not in cue_indices]

    if len(uncued_indices) == 0:
        return 1.0

    correct = sum(1 for i in uncued_indices if original_pattern[i] == recalled_pattern[i])
    return correct / len(uncued_indices)

def get_patterns_for_compare():
    """获取所有训练牌组"""
    return TRAINING_PATTERNS

def get_pattern_names(count=None):
    """获取牌组名称列表"""
    if count is None:
        count = len(TRAINING_PATTERNS)
    return [f"牌组 {i+1}" for i in range(count)]


def evaluate_generated_samples(generated, training_patterns=None):
    """
    评估生成样本的质量：匹配数、新颖数、最近 Hamming 距离、比特密度。

    Args:
        generated: np.ndarray, shape (n_samples, n_visible)
        training_patterns: 训练数据列表，默认为 TRAINING_PATTERNS

    Returns:
        dict: {
            match_count, novel_count, total,
            match_indices (哪些样本精确匹配了训练数据),
            avg_min_hamming (生成样本到最近训练模式的平均 Hamming 距离),
            training_density, generated_density
        }
    """
    if training_patterns is None:
        training_patterns = TRAINING_PATTERNS
    tp = np.array(training_patterns)
    gen = np.array(generated)

    match_indices = []
    min_hammings = []
    for i, s in enumerate(gen):
        dists = [int(np.sum(np.abs(s - p))) for p in tp]
        min_hammings.append(min(dists))
        if min(dists) == 0:
            match_indices.append(i)

    return {
        "match_count": len(match_indices),
        "novel_count": len(gen) - len(match_indices),
        "total": len(gen),
        "match_indices": match_indices,
        "avg_min_hamming": float(np.mean(min_hammings)),
        "training_density": float(np.mean(tp)),
        "generated_density": float(np.mean(gen)),
    }


# ======================================================================
# SC08 四级理解实验数据
# ======================================================================

# L3 多模态条件分布实验：bit0=1 时对应两组不同的 bits[3:5] 模式
#   Group A (bit0=1, bits[3:5]=011): 各占约 50%
#   Group B (bit0=1, bits[3:5]=100): 各占约 50%
MULTIMODAL_PATTERNS = [
    [1, 0, 1, 0, 1, 1],  # bit0=1, Group A (ends 011)
    [1, 1, 0, 0, 1, 1],  # bit0=1, Group A (ends 011)
    [1, 0, 0, 1, 0, 0],  # bit0=1, Group B (ends 100)
    [1, 1, 1, 1, 0, 0],  # bit0=1, Group B (ends 100)
    [0, 0, 1, 1, 1, 0],  # bit0=0
    [0, 1, 0, 1, 0, 1],  # bit0=0
    [0, 0, 0, 0, 1, 0],  # bit0=0
    [0, 1, 1, 0, 0, 1],  # bit0=0
]

# L4 XOR 实验：bit5 = bit0 XOR bit1
XOR_PATTERNS = [
    [0, 0, 1, 0, 1, 0],  # 0 XOR 0 = 0
    [0, 1, 0, 1, 0, 1],  # 0 XOR 1 = 1
    [1, 0, 0, 1, 1, 1],  # 1 XOR 0 = 1
    [1, 1, 1, 0, 1, 0],  # 1 XOR 1 = 0
    [0, 0, 0, 1, 0, 0],  # 0 XOR 0 = 0
    [0, 1, 1, 0, 1, 1],  # 0 XOR 1 = 1
    [1, 0, 1, 0, 0, 1],  # 1 XOR 0 = 1
    [1, 1, 0, 1, 0, 0],  # 1 XOR 1 = 0
]


def get_multimodal_patterns():
    """获取 L3 多模态实验用训练数据"""
    return MULTIMODAL_PATTERNS


def get_xor_patterns():
    """获取 L4 XOR 实验用训练数据"""
    return XOR_PATTERNS


def compute_conditional_stats(samples, clamped_idx, clamped_val, target_idx):
    """
    从采样结果中计算条件概率 P(target=1 | clamped=val)。

    Args:
        samples: np.ndarray, shape (n, n_visible)
        clamped_idx: int, 被固定的可见单元索引
        clamped_val: int, 被固定的值 (0 或 1)
        target_idx: int, 目标可见单元索引

    Returns:
        float: P(target=1 | clamped=val)，若无有效样本返回 -1
    """
    mask = samples[:, clamped_idx] == clamped_val
    filtered = samples[mask]
    if len(filtered) == 0:
        return -1.0
    return float(np.mean(filtered[:, target_idx]))


def count_distinct_outputs(samples):
    """统计采样结果中不同模式的数量和频率分布。"""
    tuples = [tuple(int(x) for x in s) for s in samples]
    counts = Counter(tuples)
    return {
        "n_distinct": len(counts),
        "distribution": dict(counts.most_common()),
    }
