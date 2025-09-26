from manim import *
from manim_slides import Slide
from src.configs import *
from src.SlideFrames import (
    SlideWithCover, SlideComponent, CoverSlideComponent,
    TitleSlideComponent, ColumnLayoutSlideComponent, ListSlideComponent,
    QuestionAnswerSlideComponent
)
from src.mobjects.faces import HumanHappyFace, HumanSadFace, BotHappyFace, BotSadFace


class QuestionSlideComponent(ColumnLayoutSlideComponent):
    """
    问题展示slide组件 - 使用三栏布局基类
    展示三个核心问题：人工智能与人类智能的关联差异、数学公式背后的认知、AI视角的误导
    """

    def __init__(self, context):
        super().__init__(context, "核心问题", num_columns=3, show_dividers=True, auto_clear=True)

    def render_columns(self):
        """渲染三栏问题内容"""
        # 左栏：人机关系
        self.render_column_content(0, self._render_human_ai_column)

        # 中栏：数学公式
        self.render_column_content(1, self._render_formula_column)

        # 右栏：行为树
        self.render_column_content(2, self._render_tree_column)

    def _render_human_ai_column(self, position):
        """左栏：人类智能与AI的关联和差异"""
        # 人类开心表情在上方
        human_face = HumanHappyFace(size=1.2, stroke_color="#FFD700", eye_color="#FFD700")
        human_face.move_to([position[0], position[1] + 0.8, 0])

        # AI开心表情在下方
        ai_face = BotHappyFace(size=1.2, stroke_color="#87CEEB", eye_color="#87CEEB")
        ai_face.move_to([position[0], position[1] - 0.8, 0])

        # 连接线（垂直）
        connection_line = Line(
            human_face.get_bottom(),
            ai_face.get_top(),
            color="#FF6B6B", stroke_width=3
        )

        # 快速动画绘制
        self.context.play(Create(human_face), run_time=0.5)
        self.context.wait(0.3)
        self.context.next_slide()
        self.context.play(Create(ai_face), run_time=0.5)
        self.context.wait(0.3)
        self.context.next_slide()

        # 建立连接
        self.context.play(Create(connection_line), run_time=0.2)
        self.context.wait(0.3)
        self.context.next_slide()

        # 断开连接（差异）
        break_symbol = Text("✗", font_size=20, color="#FF4444")
        break_symbol.move_to(connection_line.get_center())

        self.context.play(
            Create(break_symbol),
            connection_line.animate.set_opacity(0.3),
            run_time=0.3
        )
        self.context.wait(0.3)
        self.context.next_slide()

    def _render_formula_column(self, position):
        """中栏：Self-Attention公式和问号"""
        # Self-Attention 公式
        attention_formula = MathTex(
            r"\text{Att}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            font_size=18,
            color="#FFD700"
        )
        attention_formula.move_to([position[0], position[1] + 0.3, 0])

        # 大问号
        question_mark = Text("?", font_size=48, color="#FF6B6B", weight=BOLD)
        question_mark.next_to(attention_formula, DOWN, buff=0.5)

        # 快速动画绘制
        self.context.play(Write(attention_formula), run_time=0.8)
        self.context.wait(0.3)
        self.context.next_slide()

        self.context.play(
            DrawBorderThenFill(question_mark),
            question_mark.animate.scale(1.2).scale(1/1.2),  # 弹跳效果
            run_time=0.5
        )
        self.context.wait(0.3)
        self.context.next_slide()

    def _render_tree_column(self, position):
        """右栏：行为树和叉号"""
        # 创建更复杂的行为树结构，增加水平节点
        # 根节点 - 选择器
        root_selector = Circle(radius=0.18, color="#87CEEB", fill_opacity=0.8)
        root_selector.move_to([position[0], position[1] + 1.2, 0])
        selector_text = Text("?", font_size=12, color="WHITE")
        selector_text.move_to(root_selector.get_center())

        # 第二层 - 更多水平分布的节点
        sequence1 = Square(side_length=0.3, color="#87CEEB", fill_opacity=0.6)
        sequence1.move_to([position[0] - 0.7, position[1] + 0.6, 0])
        seq1_text = Text("→", font_size=10, color="WHITE")
        seq1_text.move_to(sequence1.get_center())

        parallel_node = Circle(radius=0.15, color="#87CEEB", fill_opacity=0.6)
        parallel_node.move_to([position[0], position[1] + 0.6, 0])
        parallel_text = Text("∥", font_size=10, color="WHITE")
        parallel_text.move_to(parallel_node.get_center())

        sequence2 = Square(side_length=0.3, color="#87CEEB", fill_opacity=0.6)
        sequence2.move_to([position[0] + 0.7, position[1] + 0.6, 0])
        seq2_text = Text("→", font_size=10, color="WHITE")
        seq2_text.move_to(sequence2.get_center())

        # 第三层 - 叶子节点（更多水平分布）
        leaf1 = Square(side_length=0.25, color="#87CEEB", fill_opacity=0.4)
        leaf1.move_to([position[0] - 1.0, position[1] + 0.0, 0])

        leaf2 = Square(side_length=0.25, color="#87CEEB", fill_opacity=0.4)
        leaf2.move_to([position[0] - 0.4, position[1] + 0.0, 0])

        leaf3 = Circle(radius=0.12, color="#87CEEB", fill_opacity=0.4)
        leaf3.move_to([position[0], position[1] - 0.2, 0])

        leaf4 = Square(side_length=0.25, color="#87CEEB", fill_opacity=0.4)
        leaf4.move_to([position[0] + 0.4, position[1] + 0.0, 0])

        leaf5 = Square(side_length=0.25, color="#87CEEB", fill_opacity=0.4)
        leaf5.move_to([position[0] + 1.0, position[1] + 0.0, 0])

        # 连接线
        lines = [
            # 根节点到第二层
            Line(root_selector.get_bottom(), sequence1.get_top(), color="#87CEEB", stroke_width=1.5),
            Line(root_selector.get_bottom(), parallel_node.get_top(), color="#87CEEB", stroke_width=1.5),
            Line(root_selector.get_bottom(), sequence2.get_top(), color="#87CEEB", stroke_width=1.5),
            # 第二层到叶子节点
            Line(sequence1.get_bottom(), leaf1.get_top(), color="#87CEEB", stroke_width=1.5),
            Line(sequence1.get_bottom(), leaf2.get_top(), color="#87CEEB", stroke_width=1.5),
            Line(parallel_node.get_bottom(), leaf3.get_top(), color="#87CEEB", stroke_width=1.5),
            Line(sequence2.get_bottom(), leaf4.get_top(), color="#87CEEB", stroke_width=1.5),
            Line(sequence2.get_bottom(), leaf5.get_top(), color="#87CEEB", stroke_width=1.5),
        ]

        behavior_tree = VGroup(
            root_selector, selector_text,
            sequence1, seq1_text, parallel_node, parallel_text, sequence2, seq2_text,
            leaf1, leaf2, leaf3, leaf4, leaf5, *lines
        )

        # 快速生成动画 - 从上到下，从左到右逐层出现
        # 第一层：根节点
        self.context.play(
            AnimationGroup(Create(root_selector), Write(selector_text), lag_ratio=0.1),
            run_time=0.15
        )

        # 第二层：连接线（从左到右）
        self.context.play(
            AnimationGroup(Create(lines[0]), Create(lines[1]), Create(lines[2]), lag_ratio=0.1),
            run_time=0.2
        )

        # 第三层：节点（从左到右）
        self.context.play(
            AnimationGroup(
                Create(sequence1), Write(seq1_text),
                lag_ratio=0.1
            ),
            run_time=0.1
        )
        self.context.play(
            AnimationGroup(
                Create(parallel_node), Write(parallel_text),
                lag_ratio=0.1
            ),
            run_time=0.1
        )
        self.context.play(
            AnimationGroup(Create(sequence2), Write(seq2_text), lag_ratio=0.1),
            run_time=0.1
        )

        # 第四层：连接线（从左到右）
        self.context.play(
            AnimationGroup(
                Create(lines[3]), Create(lines[4]), Create(lines[5]),
                Create(lines[6]), Create(lines[7]),
                lag_ratio=0.1
            ),
            run_time=0.2
        )

        # 第五层：叶子节点（从左到右）
        self.context.play(
            AnimationGroup(
                Create(leaf1), Create(leaf2),
                lag_ratio=0.1
            ),
            run_time=0.1
        )
        self.context.play(
            AnimationGroup(
                Create(leaf3),
                lag_ratio=0.1
            ),
            run_time=0.1
        )
        self.context.play(
            AnimationGroup(
                Create(leaf4), Create(leaf5),
                lag_ratio=0.1
            ),
            run_time=0.1
        )
        self.context.wait(0.3)
        self.context.next_slide()

        # 画大叉
        cross_line1 = Line(
            behavior_tree.get_corner(UL) + UP * 0.1 + LEFT * 0.1,
            behavior_tree.get_corner(DR) + DOWN * 0.1 + RIGHT * 0.1,
            color="#FF4444", stroke_width=4
        )
        cross_line2 = Line(
            behavior_tree.get_corner(UR) + UP * 0.1 + RIGHT * 0.1,
            behavior_tree.get_corner(DL) + DOWN * 0.1 + LEFT * 0.1,
            color="#FF4444", stroke_width=4
        )

        # 快速动画绘制叉号并变暗
        self.context.play(
            Create(cross_line1), Create(cross_line2),
            behavior_tree.animate.set_opacity(0.3),
            run_time=0.3
        )


class TargetAudienceSlideComponent(ColumnLayoutSlideComponent):
    """
    目标受众slide组件 - 使用三栏布局基类
    """

    def __init__(self, context):
        super().__init__(context, "面向的目标人群", num_columns=3)

        # 定义三栏内容
        self.audience_data = [
            ("对 AI 感兴趣的普通人", [
                "以更贴近人类的视角看待AI",
                "理解 AI 系统的设计思路"
            ]),
            ("摸索中的 AI 工具使用者", [
                "理解：AI 擅长什么，不擅长什么",
                "能够有理有据地理解各类资料"
            ]),
            ("有一定基础的学习者", [
                "新视角",
                "新灵感",
                "新启发"
            ])
        ]

    def render_columns(self):
        """渲染三栏内容"""
        for i, (title, content_list) in enumerate(self.audience_data):
            self.render_column_content(i, lambda pos, t=title, c=content_list:
                                     self._render_audience_column(pos, t, c))

    def _render_audience_column(self, position, title, content_list):
        """渲染单个受众栏"""
        # 栏标题
        column_title = Text(title, font=font_heading, font_size=20, color="#87CEEB")
        column_title.move_to([position[0], position[1] + 1.5, 0])
        self.context.play(FadeIn(column_title), run_time=0.5)
        self.context.next_slide()

        # 栏内容
        content_items = [(None, item, None) for item in content_list]

        def position_content(items):
            content_group = VGroup(*items).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
            content_group.next_to(column_title, DOWN, buff=0.5)
            content_group.move_to([position[0], content_group.get_y(), 0])

        self.interactive_list(
            content_items,
            position_content,
            font_size=16,
            color="#F0F8FF"
        )


class ContraSlideComponent(TitleSlideComponent):
    """
    核心冲突slide组件
    展示大部分人对于"计算机"的理解与现代AI工作方式的核心冲突
    """

    def __init__(self, context):
        super().__init__(context, "大部分人对于\"计算机\"的理解，和现代 AI 的工作方式，是有着核心冲突的")

    def render_content(self):
        """渲染四象限对比展示"""
        # 添加分割线
        divider_line = Line(LEFT * 6, RIGHT * 6, color="#4A90E2", stroke_width=2)
        divider_line.next_to(self.small_title, DOWN, buff=0.3)

        self.context.play(FadeIn(divider_line), run_time=0.8)

        # 创建四个象限的对比展示
        self.create_quadrant_comparisons(divider_line)

    def create_quadrant_comparisons(self, divider_line):
        """创建四个象限的对比展示"""
        # 计算象限位置
        available_top = divider_line.get_bottom()[1] - 0.3
        available_bottom = self.context.canvas["footer_line"].get_top()[1] + 0.3
        available_height = available_top - available_bottom

        quad_height = available_height / 2

        # 四个象限的中心位置
        quad_centers = [
            LEFT * 3 + UP * (available_bottom + quad_height * 1.5),  # 左上
            RIGHT * 3 + UP * (available_bottom + quad_height * 1.5), # 右上
            LEFT * 3 + UP * (available_bottom + quad_height * 0.5),  # 左下
            RIGHT * 3 + UP * (available_bottom + quad_height * 0.5)  # 右下
        ]

        # 象限标题和内容
        quadrants = [
            ("控制 vs 涌现", self.create_control_vs_emergence),
            ("逻辑 vs 直觉", self.create_logic_vs_intuition),
            ("离散 vs 连续", self.create_discrete_vs_continuous),
            ("组合 vs 插值", self.create_composition_vs_interpolation)
        ]

        # 依次显示每个象限
        for i, (title, create_func) in enumerate(quadrants):
            self.show_quadrant(title, create_func, quad_centers[i], quad_height)

    def show_quadrant(self, title, create_func, center, height):
        """显示单个象限"""
        # 显示标题
        title_text = Text(title, font=font_heading, font_size=20, color="#87CEEB")
        title_text.move_to(center)

        self.context.play(FadeIn(title_text), run_time=0.5)
        self.context.wait(0.3)
        self.context.next_slide()

        # 标题上移
        title_target = center + UP * (height / 3)
        self.context.play(title_text.animate.move_to(title_target), run_time=0.3)

        # 创建内容
        create_func(center, height)

    def create_control_vs_emergence(self, center, height):
        """控制 vs 涌现"""
        # 左侧：流程图（控制）
        left_center = center + LEFT * 1.5

        self.context.wait(0.3)
        self.context.next_slide()
        # 控制流程图：先做 -> 后做 -> 条件 -> (选择1, 选择2)
        # 先做
        first_box = Rectangle(width=0.6, height=0.25, color="#87CEEB", fill_opacity=0.3)
        first_box.move_to(left_center + UP * 0.7)
        first_text = Text("先做", font=font_main_text, font_size=10, color="WHITE")
        first_text.move_to(first_box.get_center())

        # 后做
        second_box = Rectangle(width=0.6, height=0.25, color="#87CEEB", fill_opacity=0.3)
        second_box.move_to(left_center + UP * 0.3)
        second_text = Text("后做", font=font_main_text, font_size=10, color="WHITE")
        second_text.move_to(second_box.get_center())

        # 条件判断（压扁的菱形）
        decision_diamond = Polygon(
            [0, 0.25, 0], [0.3, 0, 0], [0, -0.25, 0], [-0.3, 0, 0],
            color="#FFD700", fill_opacity=0.3
        )
        decision_diamond.move_to(left_center + DOWN * 0.1)
        decision_text = Text("条件?", font=font_main_text, font_size=9, color="BLACK")
        decision_text.move_to(decision_diamond.get_center())

        # 选择1
        choice1_box = Rectangle(width=0.5, height=0.2, color="#87CEEB", fill_opacity=0.3)
        choice1_box.move_to(left_center + LEFT * 0.4 + DOWN * 0.5)
        choice1_text = Text("选择1", font=font_main_text, font_size=8, color="WHITE")
        choice1_text.move_to(choice1_box.get_center())

        # 选择2
        choice2_box = Rectangle(width=0.5, height=0.2, color="#87CEEB", fill_opacity=0.3)
        choice2_box.move_to(left_center + RIGHT * 0.4 + DOWN * 0.5)
        choice2_text = Text("选择2", font=font_main_text, font_size=8, color="WHITE")
        choice2_text.move_to(choice2_box.get_center())

        # 连接线
        line1 = Arrow(first_box.get_bottom(), second_box.get_top(), buff=0.05, color="#4A90E2", stroke_width=2)
        line2 = Arrow(second_box.get_bottom(), decision_diamond.get_top(), buff=0.05, color="#4A90E2", stroke_width=2)
        line3 = Arrow(decision_diamond.get_bottom() + LEFT * 0.15, choice1_box.get_top(), buff=0.05, color="#4A90E2", stroke_width=2)
        line4 = Arrow(decision_diamond.get_bottom() + RIGHT * 0.15, choice2_box.get_top(), buff=0.05, color="#4A90E2", stroke_width=2)

        # 动画显示流程图
        self.context.play(Create(first_box), Write(first_text), run_time=0.3)
        self.context.play(Create(line1), run_time=0.2)
        self.context.play(Create(second_box), Write(second_text), run_time=0.3)
        self.context.play(Create(line2), run_time=0.2)
        self.context.play(Create(decision_diamond), Write(decision_text), run_time=0.3)
        self.context.play(Create(line3), Create(line4), run_time=0.3)
        self.context.play(Create(choice1_box), Write(choice1_text), Create(choice2_box), Write(choice2_text), run_time=0.4)

        self.context.wait(0.3)
        self.context.next_slide()
        # 右侧：元胞自动机（涌现）
        right_center = center + RIGHT * 1.5

        # 创建大规模元胞自动机网格
        cells = []
        grid_size = 12
        cell_size = 0.08

        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                cell = Square(side_length=cell_size, color="#4A90E2", fill_opacity=0.05, stroke_width=0.3, stroke_color="#666666")
                cell.move_to(right_center +
                           LEFT * (grid_size / 2 - 0.5 - j) * cell_size +
                           UP * (grid_size / 2 - 0.5 - i) * cell_size)
                row.append(cell)
            cells.append(row)

        # 显示网格
        all_cells = VGroup(*[cell for row in cells for cell in row])
        self.context.play(Create(all_cells), run_time=0.3)

        # 反应扩散模型：从混沌到有序的涌现
        import numpy as np
        import random

        # 初始化反应扩散系统
        # 使用Gray-Scott模型的简化版本
        A = np.ones((grid_size, grid_size))  # 物质A浓度
        B = np.zeros((grid_size, grid_size))  # 物质B浓度

        # 参数设置（调整以获得好的视觉效果）
        Da, Db = 1.0, 0.5  # 扩散系数
        f, k = 0.037, 0.06  # 反应参数
        dt = 1.0  # 时间步长

        # 随机初始化：在中心区域添加随机扰动
        random.seed(42)
        center_grid = grid_size // 2
        for i in range(center_grid-2, center_grid+3):
            for j in range(center_grid-2, center_grid+3):
                if 0 <= i < grid_size and 0 <= j < grid_size:
                    A[i][j] = 1.0 - 0.5 * random.random()
                    B[i][j] = 0.25 * random.random()

        # 初始显示
        for i in range(grid_size):
            for j in range(grid_size):
                intensity = B[i][j]
                cells[i][j].set_fill(color="#FFD700", opacity=intensity)

        self.context.wait(0.3)
        self.context.next_slide(loop=True)

        # 反应扩散演化 - 15步后循环播放
        evolution_steps = 15
        saved_states = []  # 保存每步的状态用于循环

        for step in range(evolution_steps):
            # 计算拉普拉斯算子（扩散项）
            def laplacian(matrix):
                result = np.zeros_like(matrix)
                for i in range(1, grid_size-1):
                    for j in range(1, grid_size-1):
                        result[i][j] = (matrix[i-1][j] + matrix[i+1][j] +
                                       matrix[i][j-1] + matrix[i][j+1] - 4*matrix[i][j])
                return result

            lap_A = laplacian(A)
            lap_B = laplacian(B)

            # Gray-Scott反应扩散方程
            reaction = A * B * B
            new_A = A + dt * (Da * lap_A - reaction + f * (1 - A))
            new_B = B + dt * (Db * lap_B + reaction - (k + f) * B)

            # 边界条件：保持边界值
            new_A = np.clip(new_A, 0, 1)
            new_B = np.clip(new_B, 0, 1)

            # 保存当前状态
            saved_states.append((A.copy(), B.copy()))

            # 更新显示
            animations = []
            for i in range(grid_size):
                for j in range(grid_size):
                    old_intensity = B[i][j]
                    new_intensity = new_B[i][j]

                    if abs(new_intensity - old_intensity) > 0.05:  # 只更新变化明显的
                        # 使用B物质浓度作为颜色强度
                        from manim.utils.color.core import ManimColor
                        color1 = ManimColor("#4A90E2")
                        color2 = ManimColor("#FFD700")
                        color = color1.interpolate(color2, new_intensity)
                        animations.append(
                            cells[i][j].animate.set_fill(color=color, opacity=0.3 + 0.7*new_intensity)
                        )

            if animations:
                self.context.play(*animations, run_time=0.15)
            else:
                self.context.wait(0.15)

            A, B = new_A, new_B

        self.context.wait(0.5)

        self.context.next_slide()

    def create_logic_vs_intuition(self, center, height):
        """逻辑 vs 直觉"""
        # 左侧：三段论（逻辑）- 垂直布局
        self.context.wait(0.3)
        self.context.next_slide()
        left_center = center + LEFT * 1.5

        premise_a = Text("神都是永生的", font=font_main_text, font_size=11, color="#87CEEB")
        premise_a.move_to(left_center + UP * 0.6)

        premise_b = Text("宙斯是神", font=font_main_text, font_size=11, color="#87CEEB")
        premise_b.move_to(left_center + UP * 0.2)

        conclusion = Text("宙斯是永生的", font=font_main_text, font_size=11, color="#FFD700")
        conclusion.move_to(left_center + DOWN * 0.4)

        # 向下箭头（在前提B和结论之间）
        down_arrow = Arrow(
            premise_b.get_bottom() + DOWN * 0.05,
            conclusion.get_top() + UP * 0.05,
            color="#4A90E2", buff=0.05
        )

        # 动画显示三段论
        self.context.play(Write(premise_a), run_time=0.4)
        self.context.play(Write(premise_b), run_time=0.4)
        self.context.play(Create(down_arrow), run_time=0.3)
        self.context.play(Write(conclusion), run_time=0.4)

        self.context.wait(0.3)
        self.context.next_slide()
        # 右侧：太阳→笑脸（直觉）- 上下布局
        right_center = center + RIGHT * 1.5

        # 上方：太阳
        sun_center = right_center + UP * 0.6
        sun_circle = Circle(radius=0.25, color="#FFD700", fill_opacity=0.8, stroke_width=3)
        sun_circle.move_to(sun_center)

        # 太阳光芒（8条射线）
        sun_rays = VGroup()
        for i in range(8):
            angle = i * PI / 4
            ray_start = sun_center + 0.25 * np.array([np.cos(angle), np.sin(angle), 0])
            ray_end = sun_center + 0.4 * np.array([np.cos(angle), np.sin(angle), 0])
            ray = Line(ray_start, ray_end, color="#FFD700", stroke_width=2)
            sun_rays.add(ray)

        # 下方：创建一个可变的人类表情，从无表情到笑脸
        face_center = right_center + DOWN * 0.3

        # 创建一个中性表情（使用圆脸，直线嘴巴）
        neutral_face = Circle(radius=0.3, color="#FFD700", fill_opacity=0.1, stroke_width=3)
        neutral_face.move_to(face_center)

        # 眼睛
        left_eye = Circle(radius=0.05, color="#FFD700", fill_opacity=1)
        left_eye.move_to(face_center + LEFT * 0.1 + UP * 0.1)
        right_eye = Circle(radius=0.05, color="#FFD700", fill_opacity=1)
        right_eye.move_to(face_center + RIGHT * 0.1 + UP * 0.1)

        # 无表情嘴巴（直线）
        neutral_mouth = Line(
            face_center + LEFT * 0.1 + DOWN * 0.05,
            face_center + RIGHT * 0.1 + DOWN * 0.05,
            color="#FFD700", stroke_width=3
        )

        # 创建目标笑脸
        happy_face_target = HumanHappyFace(size=0.6, stroke_color="#FFD700", eye_color="#FFD700")
        happy_face_target.move_to(face_center)

        # 动画显示直觉过程：太阳→无表情→笑脸
        self.context.play(
            Create(sun_circle), Create(sun_rays),
            run_time=0.4
        )
        self.context.wait(0.3)

        self.context.next_slide()
        # 先显示无表情的脸
        self.context.play(
            Create(neutral_face), Create(left_eye), Create(right_eye), Create(neutral_mouth),
            run_time=0.4
        )
        self.context.wait(0.2)

        # 然后变成笑脸
        neutral_face_group = VGroup(neutral_face, left_eye, right_eye, neutral_mouth)
        self.context.play(
            Transform(neutral_face_group, happy_face_target),
            run_time=0.6
        )

        self.context.wait(0.3)
        self.context.next_slide()

    def create_discrete_vs_continuous(self, center, height):
        """离散 vs 连续"""
        # 左侧：笑脸哭脸（离散）
        left_center = center + LEFT * 1.5

        # 人类开心表情
        happy_face = HumanHappyFace(size=0.5, stroke_color="#FFD700", eye_color="#FFD700")
        happy_face.move_to(left_center + UP * 0.4)

        # 人类悲伤表情
        sad_face = HumanSadFace(size=0.5, stroke_color="#87CEEB", eye_color="#87CEEB")
        sad_face.move_to(left_center + DOWN * 0.4)

        # 显示离散状态
        self.context.wait(0.3)
        self.context.next_slide()
        self.context.play(
            Create(happy_face),
            run_time=0.4
        )
        self.context.play(
            Create(sad_face),
            run_time=0.4
        )

        # 右侧：连续变化（连续）
        right_center = center + RIGHT * 1.2

        # 创建可变化的脸
        morph_face = Circle(radius=0.25, color="#87CEEB", fill_opacity=0.8)
        morph_face.move_to(right_center)

        morph_eyes = VGroup(
            Circle(radius=0.04, color="BLACK", fill_opacity=1).move_to(morph_face.get_center() + LEFT * 0.08 + UP * 0.08),
            Circle(radius=0.04, color="BLACK", fill_opacity=1).move_to(morph_face.get_center() + RIGHT * 0.08 + UP * 0.08)
        )

        # 初始哭脸
        morph_smile = Arc(radius=0.12, start_angle=0, angle=PI, color="BLACK", stroke_width=3)
        morph_smile.move_to(morph_face.get_center() + DOWN * 0.06)

        # 添加竖直滑块
        slider_center = right_center + RIGHT * 0.8

        # 滑块轨道（竖直线）
        slider_track = Line(
            slider_center + UP * 0.6,    # 对应 +1
            slider_center + DOWN * 0.6,  # 对应 -1
            color="#4A90E2", stroke_width=4
        )

        # 滑块指示器（小方块）
        slider_indicator = Square(side_length=0.08, color="#FFD700", fill_opacity=1, stroke_width=2)
        slider_indicator.move_to(slider_center + DOWN * 0.6)  # 初始位置对应 -1（不开心）

        # 滑块标签
        label_happy = Text("+1", font=font_main_text, font_size=10, color="#4A90E2")
        label_happy.move_to(slider_center + UP * 0.6 + RIGHT * 0.15)

        label_sad = Text("-1", font=font_main_text, font_size=10, color="#4A90E2")
        label_sad.move_to(slider_center + DOWN * 0.6 + RIGHT * 0.15)

        label_neutral = Text("0", font=font_main_text, font_size=10, color="#4A90E2")
        label_neutral.move_to(slider_center + RIGHT * 0.15)

        self.context.wait(0.3)
        self.context.next_slide()
        self.context.play(
            Create(morph_face), Create(morph_eyes), Create(morph_smile),
            Create(slider_track), Create(slider_indicator),
            Write(label_happy), Write(label_sad), Write(label_neutral),
            run_time=0.6
        )

        # 连续变化动画：哭脸→笑脸的ping-pong循环
        # 创建笑脸目标
        happy_smile = Arc(radius=0.12, start_angle=-PI, angle=PI, color="BLACK", stroke_width=3)
        happy_smile.flip(UP)
        happy_smile.move_to(morph_face.get_center() + DOWN * 0.06)

        # 创建哭脸目标
        sad_smile_copy = Arc(radius=0.12, start_angle=0, angle=PI, color="BLACK", stroke_width=3)
        sad_smile_copy.move_to(morph_face.get_center() + DOWN * 0.06)

        self.context.next_slide(loop=True)
        # Ping-pong动画：哭→笑→哭，滑块同步变化
        for cycle in range(2):
            # 哭脸变笑脸，滑块从-1滑到+1
            self.context.play(
                morph_face.animate.set_color("#FFD700"),
                Transform(morph_smile, happy_smile),
                slider_indicator.animate.move_to(slider_center + UP * 0.6),  # 滑到+1位置
                run_time=1.0
            )
            self.context.wait(0.3)

            # 笑脸变哭脸，滑块从+1滑到-1
            self.context.play(
                morph_face.animate.set_color("#87CEEB"),
                Transform(morph_smile, sad_smile_copy),
                slider_indicator.animate.move_to(slider_center + DOWN * 0.6),  # 滑到-1位置
                run_time=1.0
            )
            if cycle < 1:  # 最后一次不等待
                self.context.wait(0.3)

        self.context.play(
            morph_face.animate.set_color("#FFD700"),
            Transform(morph_smile, happy_smile),
            slider_indicator.animate.move_to(slider_center + UP * 0.6),  # 滑到+1位置
            run_time=1.0
        )
        self.context.wait(0.3)

        self.context.next_slide()

    def create_composition_vs_interpolation(self, center, height):
        """组合 vs 插值"""
        # 左侧：组合（正方形+圆角）
        left_center = center + LEFT * 1.5

        # 先画正方形
        base_square = Square(side_length=0.8, color="#87CEEB", stroke_width=3)
        base_square.move_to(left_center)

        self.context.wait(0.3)
        self.context.next_slide()
        self.context.play(Create(base_square), run_time=0.4)

        # 画四个小圆
        corner_radius = 0.12
        corners = [
            Circle(radius=corner_radius, color="#FFD700", fill_opacity=0.8).move_to(base_square.get_corner(UL) + DOWN * corner_radius + RIGHT * corner_radius),
            Circle(radius=corner_radius, color="#FFD700", fill_opacity=0.8).move_to(base_square.get_corner(UR) + DOWN * corner_radius + LEFT * corner_radius),
            Circle(radius=corner_radius, color="#FFD700", fill_opacity=0.8).move_to(base_square.get_corner(DL) + UP * corner_radius + RIGHT * corner_radius),
            Circle(radius=corner_radius, color="#FFD700", fill_opacity=0.8).move_to(base_square.get_corner(DR) + UP * corner_radius + LEFT * corner_radius)
        ]

        self.context.wait(0.3)
        self.context.next_slide()
        for corner in corners:
            self.context.play(Create(corner), run_time=0.15)

        # 去掉正方形的角
        rounded_square = RoundedRectangle(width=0.8, height=0.8, corner_radius=corner_radius, color="#87CEEB", stroke_width=3)
        rounded_square.move_to(left_center)

        self.context.wait(0.3)
        self.context.next_slide()
        self.context.play(
            Transform(base_square, rounded_square),
            *[FadeOut(corner) for corner in corners],
            run_time=0.6
        )

        # 右侧：插值（正方形→圆形→圆角正方形）的ping-pong循环
        right_center = center + RIGHT * 1.5

        # 初始正方形
        start_square = Square(side_length=0.8, color="#FFD700", stroke_width=3)
        start_square.move_to(right_center)

        self.context.wait(0.3)
        self.context.next_slide()
        self.context.play(Create(start_square), run_time=0.4)

        # 创建目标形状
        target_circle = Circle(radius=0.35, color="#FF6B6B", stroke_width=3)
        target_circle.move_to(right_center)

        final_rounded = RoundedRectangle(width=0.8, height=0.8, corner_radius=0.12, color="#87CEEB", stroke_width=3)
        final_rounded.move_to(right_center)

        original_square = Square(side_length=0.8, color="#FFD700", stroke_width=3)
        original_square.move_to(right_center)

        # Ping-pong动画：正方形→圆形→圆角正方形→正方形
        self.context.next_slide(loop=True)
        for cycle in range(2):
            # 正方形→圆形
            self.context.play(Transform(start_square, target_circle), run_time=0.8)
            self.context.wait(0.2)

            # 圆形→圆角正方形
            self.context.play(Transform(start_square, final_rounded), run_time=0.8)
            self.context.wait(0.2)

            # 圆角正方形→正方形（回到起点）
            if cycle < 1:  # 最后一次保持在圆角正方形
                self.context.play(Transform(start_square, original_square), run_time=0.8)
                self.context.wait(0.2)

        self.context.wait(0.3)
        self.context.next_slide()


class CurrentProblemsSlideComponent(QuestionAnswerSlideComponent):
    """
    当前问题分析 - 使用问答布局基类
    """

    def __init__(self, context):
        questions = [
            "课程与讲解方面的问题？",
            "科普与教学方面的问题？"
        ]

        answers = [
            "大部分 AI 课程专注于技术实现，关注 How 而非 What；讲解缺乏统一视角；没有回答视角来源",
            "科普往往执念于特定技术或过度简化；'教你XXX'的视频过于实用主义"
        ]

        super().__init__(context, "当前 AI 课程与科普的现状", questions, answers)

    def render_columns(self):
        """渲染问答内容，并添加总结"""
        # 调用父类方法渲染左右栏
        super().render_columns()

        # 添加总结
        summary = Text(
            "总的来说，就是讲技术的讲技术，作类比的作类比，瞎扯淡的瞎扯淡。",
            font=font_main_text, font_size=20, color="#FF6B6B", weight=BOLD
        )
        summary.to_edge(DOWN, buff=1.5)

        self.context.play(FadeIn(summary), run_time=1.0)
        self.context.wait(0.3)
        self.context.next_slide()


class VideoPlanSlideComponent(ColumnLayoutSlideComponent):
    """
    视频计划slide组件 - 使用两栏布局
    """

    def __init__(self, context):
        super().__init__(context, "视频计划", num_columns=2)

    def render_content(self):
        """重写render_content以添加副标题"""
        # 先添加副标题
        subtitle = VGroup(
            Text("保证大部分人能够学到真正 AI 相关的思维", font=font_heading, font_size=20, color="#87CEEB"),
            Text("而有基础的人，也可以获得更深更广的视角", font=font_heading, font_size=20, color="#87CEEB")
        ).arrange(DOWN, buff=0.3)
        subtitle.next_to(self.small_title, DOWN, buff=0.8)

        for line in subtitle:
            self.context.play(FadeIn(line), run_time=0.8)
            self.context.wait(0.3)
            self.context.next_slide()

        # 然后调用父类方法
        super().render_content()

    def render_columns(self):
        """渲染两栏内容"""
        # 左栏内容
        left_content = [
            "遇到、想到的各类视角 → 现代 AI 技术的部分核心思想",
            "由浅入深，逐步开始解读现代常用的各类 AI 的工作方式",
            "尽量保证信息准确的情况下，给予最低的认知压力",
            "目前计划每 2 周 - 1 个月更新一个视频（大概）",
            "前期可能以文字为主，后面看看能不能增加图示或者动画演示"
        ]

        # 右栏内容
        right_content = [
            "理解的本质，是关联已知与未知：你总会找到你所熟悉的视角",
            "创造的本质，是对已知信息的重组：你总会有所启发"
        ]

        # 渲染左栏
        self.render_column_content(0, lambda pos: self._render_plan_content(pos, left_content, 16))

        # 渲染右栏
        self.render_column_content(1, lambda pos: self._render_plan_content(pos, right_content, 18, "#FFD700"))

    def _render_plan_content(self, position, content_list, font_size, color="#F0F8FF"):
        """渲染计划内容"""
        content_items = [(None, item, None) for item in content_list]

        def position_content(items):
            content_group = VGroup(*items).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
            content_group.move_to([position[0], position[1], 0])

        self.interactive_list(
            content_items,
            position_content,
            font_size=font_size,
            color=color
        )


class CourageSlideComponent(SlideComponent):
    """
    鼓起勇气slide组件 - 使用最基础的SlideComponent
    """

    def render(self):
        """显示鼓起勇气文字"""
        # 利用基类的定位工具
        available_space = self.get_available_space()

        courage_text = Text(
            "鼓起勇气，去成为历史的车轮吧",
            font=font_heading,
            font_size=48,
            color="#FFD700",
            weight=BOLD
        )
        courage_text.move_to([0, available_space["center_y"], 0])

        # 使用 Write 动画
        self.context.play(Write(courage_text), run_time=3.0)
        self.context.wait(1.0)
        self.context.next_slide()


class NextPreviewSlideComponent(TitleSlideComponent):
    """
    下节预告slide组件
    """

    def __init__(self, context):
        super().__init__(context, "下节预告")

    def render_content(self):
        """渲染预告内容"""
        # 视频标题在底部最后出现
        subtitle = Text("《序曲：我们真的能抵达世界的真实吗》", font=font_heading, font_size=24, color="#87CEEB")
        subtitle.next_to(self.small_title, DOWN, buff=3.0)

        self.context.play(FadeIn(subtitle))
        self.context.wait(0.3)
        self.context.next_slide()


class ThanksSlideComponent(SlideComponent):
    """
    谢谢大家slide组件
    """

    def render(self):
        """显示谢谢大家"""
        # 利用基类的定位工具
        available_space = self.get_available_space()

        # 谢谢大家文字
        thanks_text = Text(
            "谢谢大家",
            font=font_heading,
            font_size=64,
            color="#FFD700",
            weight=BOLD
        )
        thanks_text.move_to([0, available_space["center_y"] + 0.5, 0])

        # 联系方式
        contact_info = VGroup(
            Text("Bilibili: @Clyce", font=font_main_text, font_size=18, color="#87CEEB"),
            Text("公众号：酒缸中的玻尔兹曼脑", font=font_main_text, font_size=18, color="#87CEEB"),
            Text("知乎：@Clyce", font=font_main_text, font_size=18, color="#87CEEB")
        ).arrange(DOWN, buff=0.3)
        contact_info.move_to([0, available_space["center_y"] - 1.5, 0])

        # 动画显示
        self.context.play(FadeIn(thanks_text), run_time=1.0)
        self.context.wait(0.5)
        self.context.play(FadeIn(contact_info), run_time=1.0)
        self.context.wait(0.3)
        self.context.next_slide()


class Prepare(SlideWithCover):
    """
    使用新架构的Prepare类 - 完全重构版本
    """

    def construct(self):
        # 添加封面页
        self.add_cover("视频介绍")

        # 添加各种组件
        self.slide_manager.add_component(QuestionSlideComponent)
        self.slide_manager.add_component(TargetAudienceSlideComponent)
        self.slide_manager.add_component(ContraSlideComponent)
        self.slide_manager.add_component(CurrentProblemsSlideComponent)
        self.slide_manager.add_component(VideoPlanSlideComponent)
        self.slide_manager.add_component(CourageSlideComponent)
        self.slide_manager.add_component(NextPreviewSlideComponent)
        self.slide_manager.add_component(ThanksSlideComponent)

        # 执行所有组件
        self.slide_manager.simple_execute_all()