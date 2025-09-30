import numpy as np
import traceback
from manim import *
from src.SlideFrames import *
from src.configs import *
from src.hopfield_tools import HopfieldNetworkTools
from SlideComponents.S02_RAMvsCAM.shared import get_patterns_for_compare, get_pattern_names, get_cue_indices, calculate_cued_accuracy

class MessedUpSlideComponent(ColumnLayoutSlideComponent):
    def __init__(self, context):
        super().__init__(context, "记忆逐渐混乱", num_columns=2)

    def render_columns(self):
        """
        Transcript：
        【播放动画 A】
        前面的例子中我们很容易想到，当我们让这群人举办的辩论赛越来越多的时候
        【播放动画 B】
        他们的"记忆"将开始变得混乱。相似的记忆可能互相混淆，相反的记忆可能互相抵消。
        【播放动画 C】
        【播放动画 D】
        那么，有没有什么办法呢
        ====
        动画定义：
            动画 A: 在 ColumnLayout 的左侧依次画六张脸，然后画连接线（注意不要画到圆心），组成一个全连接环形
            动画 B:
                1. 在 ColumnLayout 的右侧生成八个不同的牌组，排成一列（此时右侧分成两列，后一列用来校对）（注意调整行距，不要超出范围）
                2. 用 HopfieldNetwork 算法对齐进行训练
            动画 C:
                for 牌组id = 1 to 8
                1. 依次用下划线标明其中用于"提示"的2-3个位
                2. 然后把这些提示的位，依次快速移动到六张脸的位置（顺时针），然后淡出，同时接到红色牌子的变成哭脸，接到蓝色牌子的变成笑脸
                3. 使用真实 Hopfield Network 推导过程依次进行推理，并将结果放入比对区域
            动画 D：在屏幕中央用大字 write 平均准确率
        ====
        布局：
            左侧： 全连接网络
            右侧：
                牌组 1 （真实） | 牌组 1 （回忆）
                牌组 2 （真实） | 牌组 2 （回忆）
                牌组 3 （真实） | 牌组 3 （回忆）
                ...
        """
        # 获取栏位位置
        left_pos = self.column_positions[0]
        right_pos = self.column_positions[1]

        # 创建Hopfield网络工具
        self.hopfield = HopfieldNetworkTools(
            context=self.context,
            num_faces=6,
            face_size=0.5,  # 稍微小一点以适应布局
            circle_radius=1.5
        )

        # 动画 A: 在左侧创建六张脸的全连接网络
        self.hopfield.create_face_circle(left_pos, show_animation=True)
        self.context.next_slide()

        # 使用共享配置获取牌组和名称
        patterns = get_patterns_for_compare()
        pattern_names = get_pattern_names(len(patterns))

        # 动画 B: 在右侧生成 12 个牌组并训练
        # 右侧分为两列：真实模式列和回忆结果列
        true_patterns_pos = [right_pos[0] - 1.2, right_pos[1] + 2.5, 0]
        recall_results_pos = [right_pos[0] + 1.2, right_pos[1] + 2.5, 0]

        # 创建标题
        true_title = Text("真实牌组", font_size=16, color="#FFD700")
        true_title.move_to([true_patterns_pos[0], true_patterns_pos[1] + 0.5, 0])

        recall_title = Text("回忆结果", font_size=16, color="#90EE90")
        recall_title.move_to([recall_results_pos[0], recall_results_pos[1] + 0.5, 0])

        self.context.play(Write(true_title), Write(recall_title), run_time=0.3)

        # 显示所有牌组
        self._display_all_patterns(patterns, pattern_names, true_patterns_pos)
        self.context.next_slide()

        self.hopfield.train_hopfield_network(patterns, show_animation=True)
        self.context.next_slide()

        # 动画 C: 依次测试每个牌组的回忆能力
        accuracies = []

        for pattern_idx, pattern in enumerate(patterns):
            accuracy = self._test_pattern_recall(
                pattern, pattern_idx,
                true_patterns_pos, recall_results_pos
            )
            accuracies.append(accuracy)

        # 动画 D: 显示平均准确率
        avg_accuracy = np.mean(accuracies)
        self._show_final_accuracy(avg_accuracy)

    def _display_all_patterns(self, patterns, pattern_names, base_pos):
        """显示所有牌组"""
        dot_spacing = 0.15  # 减小间距以适应更多模式

        for pattern_idx, (pattern, name) in enumerate(zip(patterns, pattern_names)):
            # 计算显示位置（垂直排列）
            y_offset = pattern_idx * 0.6
            pattern_pos = [base_pos[0], base_pos[1] - y_offset, 0]

            # 创建组名
            name_label = Text(name, font_size=12, color="#FFD700")
            name_label.move_to([pattern_pos[0] - 1.0, pattern_pos[1], 0])
            self.context.play(Write(name_label), run_time=0.1)

            # 创建牌组圆点
            for i, value in enumerate(pattern):
                dot = self.hopfield.create_card(value)
                pos = [pattern_pos[0] + (i - 2.5) * dot_spacing, pattern_pos[1], 0]
                dot.move_to(pos)
                self.context.play(FadeIn(dot), run_time=0.05)


    def _test_pattern_recall(self, original_pattern, pattern_idx, true_pos, recall_pos):
        """测试单个模式的回忆能力"""
        # 使用共享配置获取提示位组合（统一使用3位掩码）
        cue_indices = get_cue_indices(pattern_idx)

        # 创建提示模式
        cue_pattern = [None] * 6
        for idx in cue_indices:
            cue_pattern[idx] = original_pattern[idx]

        # 重置脸部表情
        self.hopfield.reset_faces_to_neutral(show_animation=True)

        # 高亮提示位置（在真实牌组中）
        y_offset = pattern_idx * 0.6
        pattern_pos = [true_pos[0], true_pos[1] - y_offset, 0]

        # 创建下划线标明提示位置（根据cue_size动态生成）
        dot_spacing = 0.15
        underlines = []
        for idx in cue_indices:
            underline = Line(
                [pattern_pos[0] + (idx - 2.5) * dot_spacing - 0.06, pattern_pos[1] - 0.15, 0],
                [pattern_pos[0] + (idx - 2.5) * dot_spacing + 0.06, pattern_pos[1] - 0.15, 0],
                color=RED, stroke_width=3
            )
            underlines.append(underline)

        self.context.play(*[Create(line) for line in underlines], run_time=0.5)
        self.context.wait(0.5)

        # 分发提示到脸部（只显示提示圆点的移动，不更新脸部）
        for idx in cue_indices:
            value = original_pattern[idx]
            dot = self.hopfield.create_card(value)

            start_pos = [pattern_pos[0] + (idx - 2.5) * dot_spacing, pattern_pos[1], 0]
            target_pos = self.hopfield.face_positions[idx]

            dot.move_to(start_pos)
            self.context.play(dot.animate.move_to(target_pos + UP * 0.3), run_time=0.2)
            self.context.play(FadeOut(dot), run_time=0.1)

        #self.context.wait(0.5)

        # 进行网络回忆（让工具类处理所有脸部动画）
        recalled_pattern = self.hopfield.network_recall(cue_pattern, show_animation=True)

        # 安全检查：确保recalled_pattern是有效的
        if recalled_pattern is None:
            print(f"Warning: network_recall returned None for pattern {pattern_idx}")
            recalled_pattern = [0] * 6  # 默认模式
        elif len(recalled_pattern) != 6:
            print(f"Warning: recalled_pattern length {len(recalled_pattern)} != 6 for pattern {pattern_idx}")
            # 截断或填充到正确长度
            recalled_pattern = recalled_pattern[:6] + [0] * max(0, 6 - len(recalled_pattern))

        # 显示回忆结果
        recall_y_pos = [recall_pos[0], recall_pos[1] - pattern_idx * 0.6, 0]

        for i, value in enumerate(recalled_pattern):
            if i >= 6:  # 额外的安全检查
                break
            dot = self.hopfield.create_card(value)
            pos = [recall_y_pos[0] + (i - 2.5) * dot_spacing, recall_y_pos[1], 0]
            dot.move_to(pos)
            self.context.play(FadeIn(dot), run_time=0.1)

        # 计算准确率（只计算未被提示的部分）
        accuracy = calculate_cued_accuracy(original_pattern, recalled_pattern, cue_indices)

        # 显示准确率
        acc_text = Text(
            f"{accuracy:.1%}", font_size=10, color="#90EE90" if accuracy > 0.8 else "#FFD700" if accuracy > 0.5 else "#FF6B6B")
        acc_text.move_to([recall_y_pos[0] + 1.0, recall_y_pos[1], 0])
        self.context.play(Write(acc_text), run_time=0.3)

        # 清除下划线
        #self.context.play(*[FadeOut(line) for line in underlines], run_time=0.3)

        self.context.wait(0.3)
        return accuracy

        # _calculate_cued_accuracy 方法已移至 Shared.py，这里不再需要

    def _show_final_accuracy(self, avg_accuracy):
        """显示最终平均准确率"""
        # 清除当前内容，准备显示大字
        self.context.wait(0.3)

        # 在屏幕中央显示平均准确率
        center_text = Text(f"未提示部分平均准确率: {avg_accuracy:.1%}",
                          font_size=32, color="#FF6B9D", weight=BOLD)
        center_text.move_to(ORIGIN + UP * 0.3)

        self.context.play(Write(center_text), run_time=1.5)
        self.context.wait(0.5)

        # 添加说明文本
        explanation_line1 = Text("（仅计算网络自主推导的位置）", font_size=18, color="#FFD700")
        explanation_line1.move_to(ORIGIN + DOWN * 0.3)
        self.context.play(Write(explanation_line1), run_time=0.8)

        # 添加解释文本
        if avg_accuracy < 0.6:
            explanation = Text("记忆开始混乱！", font_size=24, color="#FF6B6B")
        elif avg_accuracy < 0.8:
            explanation = Text("部分记忆丢失", font_size=24, color="#FFD700")
        else:
            explanation = Text("记忆依然清晰", font_size=24, color="#90EE90")

        explanation.move_to(ORIGIN + DOWN * 1.5)
        self.context.play(Write(explanation), run_time=1.0)

        self.context.wait(2.0)
        self.context.next_slide()
