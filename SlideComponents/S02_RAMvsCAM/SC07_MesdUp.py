import numpy as np
from manim import *
from src.SlideFrames import *
from src.configs import *
from src.hopfield_tools import HopfieldNetworkTools
from SlideComponents.S02_RAMvsCAM.shared import (
    get_patterns_for_compare, get_pattern_names, get_cue_indices, calculate_cued_accuracy
)

FULL_DEMO_COUNT = 2
DOT_SPACING = 0.15


class MessedUpSlideComponent(ColumnLayoutSlideComponent):
    def __init__(self, context):
        super().__init__(context, "记忆逐渐混乱", num_columns=2)

    def render_columns(self):
        left_pos = self.column_positions[0]
        right_pos = self.column_positions[1]

        self.hopfield = HopfieldNetworkTools(
            context=self.context,
            num_faces=6,
            face_size=0.5,
            circle_radius=1.5
        )

        self.hopfield.create_face_circle(left_pos, show_animation=True)
        self.context.next_slide()

        patterns = get_patterns_for_compare()
        pattern_names = get_pattern_names(len(patterns))

        true_patterns_pos = [right_pos[0] - 1.2, right_pos[1] + 2.5, 0]
        recall_results_pos = [right_pos[0] + 1.2, right_pos[1] + 2.5, 0]

        true_title = Text("真实牌组", font_size=16, color="#FFD700")
        true_title.move_to([true_patterns_pos[0], true_patterns_pos[1] + 0.5, 0])
        recall_title = Text("回忆结果", font_size=16, color="#90EE90")
        recall_title.move_to([recall_results_pos[0], recall_results_pos[1] + 0.5, 0])

        self.context.play(Write(true_title), Write(recall_title), run_time=0.3)
        self._display_all_patterns(patterns, pattern_names, true_patterns_pos)
        self.context.next_slide()

        self.hopfield.train_hopfield_network(patterns, show_animation=True)
        self.context.next_slide()

        accuracies = []

        # 前 FULL_DEMO_COUNT 个模式：完整演示
        for idx in range(FULL_DEMO_COUNT):
            acc = self._test_pattern_recall_full(
                patterns[idx], idx, true_patterns_pos, recall_results_pos
            )
            accuracies.append(acc)

        # 剩余模式：批量快速展示结果
        batch_accs = self._batch_recall_remaining(
            patterns, FULL_DEMO_COUNT, true_patterns_pos, recall_results_pos
        )
        accuracies.extend(batch_accs)

        avg_accuracy = np.mean(accuracies)
        self._show_final_accuracy(avg_accuracy)

    def _display_all_patterns(self, patterns, pattern_names, base_pos):
        all_dots = []
        all_labels = []
        for pattern_idx, (pattern, name) in enumerate(zip(patterns, pattern_names)):
            y_offset = pattern_idx * 0.6
            pattern_pos = [base_pos[0], base_pos[1] - y_offset, 0]

            name_label = Text(name, font_size=12, color="#FFD700")
            name_label.move_to([pattern_pos[0] - 1.0, pattern_pos[1], 0])
            all_labels.append(name_label)

            for i, value in enumerate(pattern):
                dot = self.hopfield.create_card(value)
                pos = [pattern_pos[0] + (i - 2.5) * DOT_SPACING, pattern_pos[1], 0]
                dot.move_to(pos)
                all_dots.append(dot)

        self.context.play(
            *[Write(l) for l in all_labels],
            *[FadeIn(d) for d in all_dots],
            run_time=0.5
        )

    def _test_pattern_recall_full(self, original_pattern, pattern_idx, true_pos, recall_pos):
        """完整演示单个模式的回忆过程（用于前几个模式）"""
        cue_indices = get_cue_indices(pattern_idx)
        cue_pattern = [None] * 6
        for idx in cue_indices:
            cue_pattern[idx] = original_pattern[idx]

        self.hopfield.reset_faces_to_neutral(show_animation=True)

        y_offset = pattern_idx * 0.6
        pattern_pos = [true_pos[0], true_pos[1] - y_offset, 0]

        underlines = []
        for idx in cue_indices:
            underline = Line(
                [pattern_pos[0] + (idx - 2.5) * DOT_SPACING - 0.06, pattern_pos[1] - 0.15, 0],
                [pattern_pos[0] + (idx - 2.5) * DOT_SPACING + 0.06, pattern_pos[1] - 0.15, 0],
                color=RED, stroke_width=3
            )
            underlines.append(underline)

        self.context.play(*[Create(line) for line in underlines], run_time=0.3)

        for idx in cue_indices:
            value = original_pattern[idx]
            dot = self.hopfield.create_card(value)
            start_pos = [pattern_pos[0] + (idx - 2.5) * DOT_SPACING, pattern_pos[1], 0]
            target_pos = self.hopfield.face_positions[idx]
            dot.move_to(start_pos)
            self.context.play(dot.animate.move_to(target_pos + UP * 0.3), run_time=0.15)
            self.context.remove(dot)

        recalled_pattern = self.hopfield.network_recall(cue_pattern, show_animation=True)
        if recalled_pattern is None:
            recalled_pattern = [0] * 6
        elif len(recalled_pattern) != 6:
            recalled_pattern = recalled_pattern[:6] + [0] * max(0, 6 - len(recalled_pattern))

        accuracy = self._show_recall_result(
            original_pattern, recalled_pattern, cue_indices, pattern_idx, recall_pos
        )

        self.context.play(*[FadeOut(line) for line in underlines], run_time=0.2)
        self.context.next_slide()
        return accuracy

    def _batch_recall_remaining(self, patterns, start_idx, true_pos, recall_pos):
        """批量快速处理剩余模式，只显示结果"""
        self.hopfield.reset_faces_to_neutral(show_animation=False)
        accuracies = []
        batch_anims = []

        for pattern_idx in range(start_idx, len(patterns)):
            pattern = patterns[pattern_idx]
            cue_indices = get_cue_indices(pattern_idx)
            cue_pattern = [None] * 6
            for idx in cue_indices:
                cue_pattern[idx] = pattern[idx]

            recalled_pattern = self.hopfield.network_recall(cue_pattern, show_animation=False)
            if recalled_pattern is None:
                recalled_pattern = [0] * 6
            elif len(recalled_pattern) != 6:
                recalled_pattern = recalled_pattern[:6] + [0] * max(0, 6 - len(recalled_pattern))

            accuracy = calculate_cued_accuracy(pattern, recalled_pattern, cue_indices)
            accuracies.append(accuracy)

            recall_y_pos = [recall_pos[0], recall_pos[1] - pattern_idx * 0.6, 0]
            for i, value in enumerate(recalled_pattern[:6]):
                dot = self.hopfield.create_card(value)
                pos = [recall_y_pos[0] + (i - 2.5) * DOT_SPACING, recall_y_pos[1], 0]
                dot.move_to(pos)
                batch_anims.append(FadeIn(dot))

            acc_color = "#90EE90" if accuracy > 0.8 else "#FFD700" if accuracy > 0.5 else "#FF6B6B"
            acc_text = Text(f"{accuracy:.1%}", font_size=10, color=acc_color)
            acc_text.move_to([recall_y_pos[0] + 1.0, recall_y_pos[1], 0])
            batch_anims.append(Write(acc_text))

        if batch_anims:
            self.context.play(*batch_anims, run_time=0.8)

        self.context.next_slide()
        return accuracies

    def _show_recall_result(self, original_pattern, recalled_pattern, cue_indices, pattern_idx, recall_pos):
        recall_y_pos = [recall_pos[0], recall_pos[1] - pattern_idx * 0.6, 0]
        result_anims = []
        for i, value in enumerate(recalled_pattern[:6]):
            dot = self.hopfield.create_card(value)
            pos = [recall_y_pos[0] + (i - 2.5) * DOT_SPACING, recall_y_pos[1], 0]
            dot.move_to(pos)
            result_anims.append(FadeIn(dot))

        accuracy = calculate_cued_accuracy(original_pattern, recalled_pattern, cue_indices)
        acc_color = "#90EE90" if accuracy > 0.8 else "#FFD700" if accuracy > 0.5 else "#FF6B6B"
        acc_text = Text(f"{accuracy:.1%}", font_size=10, color=acc_color)
        acc_text.move_to([recall_y_pos[0] + 1.0, recall_y_pos[1], 0])
        result_anims.append(Write(acc_text))

        self.context.play(*result_anims, run_time=0.4)
        return accuracy

    def _show_final_accuracy(self, avg_accuracy):
        self.context.wait(0.3)

        center_text = Text(
            f"未提示部分平均准确率: {avg_accuracy:.1%}",
            font_size=32, color="#FF6B9D", weight=BOLD
        )
        center_text.move_to(ORIGIN + UP * 0.3)
        self.context.play(Write(center_text), run_time=1.0)

        explanation_line1 = Text("（仅计算网络自主推导的位置）", font_size=18, color="#FFD700")
        explanation_line1.move_to(ORIGIN + DOWN * 0.3)
        self.context.play(Write(explanation_line1), run_time=0.5)

        if avg_accuracy < 0.6:
            msg, color = "记忆开始混乱！", "#FF6B6B"
        elif avg_accuracy < 0.8:
            msg, color = "部分记忆丢失", "#FFD700"
        else:
            msg, color = "记忆依然清晰", "#90EE90"

        explanation = Text(msg, font_size=24, color=color)
        explanation.move_to(ORIGIN + DOWN * 1.5)
        self.context.play(Write(explanation), run_time=0.8)

        bridge1 = Text(
            "记忆容量 ≈ 0.14N —— 6 个神经元存不下这么多模式",
            font=font_main_text, font_size=20, color="#87CEEB"
        )
        bridge1.move_to(ORIGIN + DOWN * 2.3)
        self.context.play(Write(bridge1), run_time=0.8)
        self.context.next_slide()

        bridge2 = Text(
            "自然的想法：加更多神经元？加一个隐藏层？",
            font=font_main_text, font_size=18, color="#FFD700"
        )
        bridge2.move_to(ORIGIN + DOWN * 2.9)
        self.context.play(Write(bridge2), run_time=0.8)

        bridge3 = Text(
            "如果我们想要的不仅是更大的记忆，而是真正的理解呢？",
            font=font_main_text, font_size=22, color="#90EE90"
        )
        bridge3.move_to(ORIGIN + DOWN * 3.5)
        self.context.play(Write(bridge3), run_time=0.8)
        self.context.next_slide()
