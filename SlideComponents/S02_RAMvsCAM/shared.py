"""
SC07、SC08、Compare.py 共享的配置数据
统一牌组模式和提示位组合配置
"""

import numpy as np

"""
# 12个不重复的6位二进制牌组（扩展到12个以支持Compare.py）
TRAINING_PATTERNS = [
    [1, 0, 1, 0, 1, 1],  # 牌组1: 4个1，2个0
    [0, 1, 1, 0, 0, 1],  # 牌组2: 3个1，3个0
    #[1, 1, 0, 1, 0, 0],  # 牌组3: 3个1，3个0
    #[0, 0, 1, 1, 1, 0],  # 牌组4: 3个1，3个0
    #[1, 0, 0, 1, 1, 1],  # 牌组5: 4个1，2个0
    #[0, 1, 0, 0, 1, 1],  # 牌组6: 3个1，3个0
    #[1, 1, 1, 0, 0, 0],  # 牌组7: 3个1，3个0
    #[0, 0, 0, 1, 1, 0],  # 牌组8: 2个1，4个0
    #[1, 1, 0, 0, 1, 1],  # 牌组9: 4个1，2个0
    #[0, 1, 1, 1, 0, 0],  # 牌组10: 3个1，3个0
    #[1, 0, 1, 1, 0, 1],  # 牌组11: 4个1，2个0
    #[0, 0, 1, 0, 1, 1],  # 牌组12: 3个1，3个0
]

# 12个不冲突的3位提示位组合
CUE_COMBINATIONS_3BIT = [
    [0, 1, 2],  # 实验1: 位置0,1,2
    [0, 1, 3],  # 实验2: 位置0,1,3
    #[0, 1, 4],  # 实验3: 位置0,1,4
    #[0, 1, 5],  # 实验4: 位置0,1,5
    #[0, 2, 3],  # 实验5: 位置0,2,3
    #[0, 2, 4],  # 实验6: 位置0,2,4
    #[0, 2, 5],  # 实验7: 位置0,2,5
    #[0, 3, 4],  # 实验8: 位置0,3,4
    #[0, 3, 5],  # 实验9: 位置0,3,5
    #[0, 4, 5],  # 实验10: 位置0,4,5
    #[1, 2, 3],  # 实验11: 位置1,2,3
    #[1, 2, 4],  # 实验12: 位置1,2,4
]
"""

# 12个不重复的6位二进制牌组（扩展到12个以支持Compare.py）
TRAINING_PATTERNS = [
    [0, 0, 0, 0, 1, 1],  # 牌组1: 4个1，2个0
    [0, 0, 1, 0, 0, 1],  # 牌组2: 3个1，3个0
    [0, 1, 0, 1, 0, 0],  # 牌组3: 3个1，3个0
    [0, 1, 1, 1, 1, 0],  # 牌组4: 3个1，3个0
    [1, 0, 0, 1, 1, 1],  # 牌组5: 4个1，2个0
    [1, 0, 1, 0, 1, 1],  # 牌组6: 3个1，3个0
    [1, 1, 0, 0, 0, 0],  # 牌组7: 3个1，3个0
    [1, 1, 1, 1, 1, 0],  # 牌组8: 2个1，4个0
]

# 12个不冲突的3位提示位组合
CUE_COMBINATIONS_3BIT = [
    [0, 1, 2],  # 实验1: 位置0,1,2
    [0, 1, 2],  # 实验1: 位置0,1,2
    [0, 1, 2],  # 实验1: 位置0,1,2
    [0, 1, 2],  # 实验1: 位置0,1,2
    [0, 1, 2],  # 实验1: 位置0,1,2
    [0, 1, 2],  # 实验1: 位置0,1,2
    [0, 1, 2],  # 实验1: 位置0,1,2
    [0, 1, 2],  # 实验1: 位置0,1,2
]

def get_cue_indices(pattern_idx, cue_size=None):
    """
    获取指定模式的提示位组合

    Args:
        pattern_idx: 模式索引 (0-based)
        cue_size: 提示位数量，如果为None则根据模式索引自动决定
                 - SC07/SC08: 前3个模式用3位，后6个用2位
                 - Compare.py: 前4个模式用3位，后8个用2位

    Returns:
        list: 提示位索引列表
    """
    return CUE_COMBINATIONS_3BIT[pattern_idx % len(CUE_COMBINATIONS_3BIT)]

def calculate_cued_accuracy(original_pattern, recalled_pattern, cue_indices):
    """
    计算仅针对未被提示部分的准确率

    Args:
        original_pattern: 原始模式
        recalled_pattern: 回忆结果模式
        cue_indices: 提示位索引列表

    Returns:
        float: 准确率 (0.0-1.0)
    """
    if len(original_pattern) != len(recalled_pattern):
        return 0.0

    # 获取未被提示的位置
    uncued_indices = [i for i in range(len(original_pattern)) if i not in cue_indices]

    if len(uncued_indices) == 0:
        return 1.0  # 如果所有位置都被提示了，认为准确率为100%

    # 计算未被提示位置的准确率
    correct = sum(1 for i in uncued_indices if original_pattern[i] == recalled_pattern[i])
    return correct / len(uncued_indices)

def get_patterns_for_compare():
    """获取Compare.py使用的12个牌组"""
    return TRAINING_PATTERNS

def get_pattern_names(count=None):
    """获取牌组名称列表"""
    if count is None:
        count = len(TRAINING_PATTERNS)
    return [f"牌组 {i+1}" for i in range(count)]
