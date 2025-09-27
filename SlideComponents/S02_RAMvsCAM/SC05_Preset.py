from manim import *
from src.SlideFrames import *
from src.configs import *
from src.hopfield_tools import HopfieldNetworkTools

class PreSetSlideComponent(ColumnLayoutSlideComponent):
    def __init__(self, context):
        super().__init__(context, "……预设立场", num_columns=2, show_dividers=True)

    def render_columns(self):
        """
        使用 HopfieldNetworkTools 重构的版本
        """
        # 获取栏位位置
        left_pos = self.column_positions[0]
        right_pos = self.column_positions[1]

        # 创建Hopfield网络工具
        self.hopfield = HopfieldNetworkTools(
            context=self.context,
            num_faces=6,
            face_size=0.6,
            circle_radius=1.8
        )

        # 牌组定义
        self.pattern1 = [1, 0, 1, 0, 1, 1]  # 101011
        self.pattern2 = [0, 1, 1, 0, 0, 1]  # 011001

        # 动画 A: 在左栏创建六张脸的环形网络
        self.hopfield.create_face_circle(left_pos, show_animation=True)

        # 在右栏创建权重矩阵显示
        self.hopfield.create_weight_matrix_display(right_pos, show_animation=True)

        # 在左栏底部设置牌组显示区域
        self._setup_card_display_area(left_pos)

        # 动画 B: 生成牌组1并分发
        self._animate_card_distribution(self.pattern1, "牌组 1")

        # 动画 C: 运行Hopfield Network学习过程
        self.hopfield.train_hopfield_network([self.pattern1], show_animation=True)
        self.context.next_slide()

        # 动画 D: 收回牌组1
        self._animate_card_collection("牌组 1")

        # 动画 E: 生成牌组2并分发
        self._animate_card_distribution(self.pattern2, "牌组 2")

        # 动画 F: 批量学习两个模式
        self.hopfield.train_hopfield_network([self.pattern1, self.pattern2], show_animation=True)
        self.context.next_slide()

        # 动画 G: 部分提示和网络收敛（牌组1的前三位）
        self._animate_partial_cue_and_recall()

        # 动画 H: 提取结果并比对（牌组1）
        self._animate_result_comparison()

        # 动画 I: 对牌组2的中间两位进行部分提示和回忆
        self._animate_partial_cue_pattern2()

        # 动画 J: 提取结果并比对（牌组2）
        self._animate_result_comparison_pattern2()


    def _setup_card_display_area(self, left_pos):
        """设置牌组显示区域"""
        # 获取可用空间
        available_space = self.get_available_space()

        # 牌组标签位置（在左栏底部，分开更多距离避免重叠）
        self.card_area_1 = [left_pos[0] - 1.3, available_space["bottom"] + 0.8, 0]
        self.card_area_2 = [left_pos[0] + 1.3, available_space["bottom"] + 0.8, 0]

    def _animate_card_distribution(self, pattern, group_name):
        """动画：分发牌组"""
        # 显示牌组标签
        group_label = Text(group_name, font_size=16, color="#FFD700")
        base_pos = self.card_area_1 if "1" in group_name else self.card_area_2
        group_label.move_to(base_pos)
        group_label.shift(UP * 0.4)
        self.context.play(Write(group_label), run_time=0.5)

        # 创建牌组（使用圆点，减小间距）
        cards = []
        dot_spacing = 0.25  # 减小间距

        for i, value in enumerate(pattern):
            dot = self.hopfield.create_card(value)
            # 水平排列牌组，增加间距
            pos = [base_pos[0] + (i - 2.5) * dot_spacing, base_pos[1], 0]
            dot.move_to(pos)
            cards.append(dot)

            self.context.play(FadeIn(dot), run_time=0.15)

        self.context.wait(0.3)
        self.context.next_slide()

        # 分发给脸
        self.hopfield.distribute_pattern_to_faces(pattern, show_animation=True)

        self.context.wait(0.3)
        self.context.next_slide()


    def _animate_card_collection(self, group_name):
        """动画：收回牌组"""
        # 从脸收回圆点（脸的表情保持不变）
        base_pos = self.card_area_1 if "1" in group_name else self.card_area_2
        dot_spacing = 0.25  # 使用相同的间距

        dots_to_fade = []  # 收集需要淡出的圆点

        for i in range(6):  # 使用固定数量
            # 使用工具类跟踪的表情状态
            value = self.hopfield.current_emotions[i] if self.hopfield.current_emotions[i] is not None else 0

            dot = self.hopfield.create_card(value)
            dot.move_to(self.hopfield.face_positions[i] + UP * 0.3)

            # 水平排列收回的圆点
            target_pos = [base_pos[0] + (i - 2.5) * dot_spacing, base_pos[1], 0]

            self.context.play(
                FadeIn(dot),
                dot.animate.move_to(target_pos),
                run_time=0.15
            )

            dots_to_fade.append(dot)

        self.context.wait(0.3)
        self.context.play(*[FadeOut(dot) for dot in dots_to_fade], run_time=0.1)
        self.context.next_slide()

    def _animate_partial_cue_and_recall(self):
        """动画：部分提示和回忆"""
        # 创建红框框选前三张圆点（使用新的间距计算）
        dot_spacing = 0.25
        frame_width = 3 * dot_spacing + 0.2  # 3个圆点的宽度加一点边距
        frame = Rectangle(
            width=frame_width, height=0.4,
            color=RED, stroke_width=3, fill_opacity=0
        )
        # 计算前三个圆点的中心位置
        first_three_center = self.card_area_1[0] - 1.5 * dot_spacing
        frame.move_to([first_three_center, self.card_area_1[1], 0])

        self.hopfield.reset_faces_to_neutral(show_animation=True)
        self.context.play(Create(frame), run_time=0.5)
        self.context.wait(0.5)

        # 使用工具类进行部分提示回忆
        cue_pattern = [1, 0, 1, None, None, None]  # 牌组1的前三位
        recalled_pattern = self.hopfield.network_recall(cue_pattern, show_animation=True)

        # 清除框选
        self.context.play(FadeOut(frame), run_time=0.3)
        self.context.next_slide()


    def _animate_result_comparison(self):
        """动画：结果比对"""
        # 使用工具类提取模式并显示
        target_pos = [self.card_area_1[0], self.card_area_1[1] - 0.6, 0]
        extracted_pattern = self.hopfield.extract_pattern_from_faces(target_pos, "回忆结果:", show_animation=True)

        # 计算并显示准确率
        accuracy = self.hopfield.calculate_accuracy(extracted_pattern, self.pattern1)
        accuracy_pos = [self.card_area_1[0], self.card_area_1[1] - 1.3, 0]
        self.hopfield.show_accuracy_text(accuracy, accuracy_pos, show_animation=True)

        self.context.wait(2.0)
        self.context.next_slide()

    def _animate_partial_cue_pattern2(self):
        """动画：对牌组2中间两位进行部分提示和回忆"""
        # 重置脸部表情为中性
        self.hopfield.reset_faces_to_neutral(show_animation=True)
        self.context.wait(0.5)

        # 创建红框框选牌组2的最后两张圆点（索引4-5）
        dot_spacing = 0.25
        frame_width = 2 * dot_spacing + 0.2
        frame = Rectangle(
            width=frame_width, height=0.4,
            color=RED, stroke_width=3, fill_opacity=0
        )
        indices_4_5_center = self.card_area_2[0] + 2.0 * dot_spacing
        frame.move_to([indices_4_5_center, self.card_area_2[1], 0])

        self.context.play(Create(frame), run_time=0.5)
        self.context.wait(0.5)

        # 添加提示文本
        cue_text = Text("牌组2最后两位提示", font_size=14, color="#FFD700")
        cue_text.move_to([self.card_area_2[0], self.card_area_2[1] + 0.6, 0])
        self.context.play(Write(cue_text), run_time=0.5)

        # 使用工具类进行部分提示回忆（牌组2的最后两位：索引4-5）
        cue_pattern = [None, None, None, None, 0, 1]  # 牌组2的最后两位
        recalled_pattern = self.hopfield.network_recall(cue_pattern, show_animation=True)

        # 清除框选和提示文本
        self.context.play(FadeOut(frame), FadeOut(cue_text), run_time=0.3)
        self.context.next_slide()

    def _animate_result_comparison_pattern2(self):
        """动画：牌组2的结果比对"""
        # 使用工具类提取模式并显示
        target_pos = [self.card_area_2[0], self.card_area_2[1] - 0.6, 0]
        extracted_pattern = self.hopfield.extract_pattern_from_faces(target_pos, "回忆结果:", show_animation=True)

        # 计算并显示准确率
        accuracy = self.hopfield.calculate_accuracy(extracted_pattern, self.pattern2)
        accuracy_pos = [self.card_area_2[0], self.card_area_2[1] - 1.3, 0]
        self.hopfield.show_accuracy_text(accuracy, accuracy_pos, show_animation=True)

        # 添加对比说明
        if accuracy > 0.8:
            result_text = Text("成功回忆牌组2！", font_size=12, color="#90EE90")
        elif accuracy > 0.5:
            result_text = Text("部分回忆成功", font_size=12, color="#FFD700")
        else:
            result_text = Text("可能回忆到牌组1", font_size=12, color="#FF6B6B")

        result_text.move_to([self.card_area_2[0], self.card_area_2[1] - 1.6, 0])
        self.context.play(Write(result_text), run_time=0.5)

        self.context.wait(2.0)
        self.context.next_slide()
