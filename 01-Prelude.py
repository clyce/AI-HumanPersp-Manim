from manim import *
from src.configs import *
from src.SlideFrames import (
    SlideWithCover, SlideComponent, CoverSlideComponent,
    TitleSlideComponent, ColumnLayoutSlideComponent, ListSlideComponent,
    QuestionAnswerSlideComponent,
    ThanksSlideComponent
)
from src.mobjects.faces import *
from src.mobjects.bio import *
from src.mobjects.icons import *
from src.mobjects.nature import *


class ElectromagneticQuoteSlideComponent(TitleSlideComponent):
    """
    电磁波引言slide组件
    第一幕：显示引用文本和电磁波动画
    第二幕：三栏布局展示光的传播
    """

    def __init__(self, context):
        super().__init__(context, "电磁波不是看不见摸不着的东西，恰恰相反，它是你唯一看得见摸得着的东西", auto_clear=False)

    def render_content(self):
        """渲染两幕内容"""
        # 第一幕：核心引用文本和背景动画
        self.show_electromagnetic_quote()

        # 第二幕：三栏布局演示
        self.show_three_column_demonstration()

    def show_electromagnetic_quote(self):
        """第一幕：显示电磁波引用和背景动画"""
        # 创建背景电磁波
        self.create_electromagnetic_background()

        # 引用文本（使用大标题，覆盖小标题）
        quote_text = Text(
            "电磁波不是看不见摸不着的东西，\n恰恰相反，\n它是你唯一看得见摸得着的东西",
            font=font_heading,
            font_size=32,
            color="#FFD700",
            line_spacing=1.3,
            weight=BOLD
        )
        quote_text.move_to(ORIGIN)

        # 逐字显现效果
        self.context.play(Write(quote_text), run_time=3.0)
        self.context.wait(0.3)

        # 保存引用文本以便后续移动
        self.quote_text = quote_text
        self.context.next_slide()

    def create_electromagnetic_background(self):
        """创建背景电磁波动画"""
        # 创建多条正弦波作为背景
        waves = VGroup()

        for i in range(5):
            # 创建正弦波
            wave = ParametricFunction(
                lambda t: np.array([
                    t * 2,
                    0.3 * np.sin(2 * PI * t + i * PI/3) + i * 0.4 - 1,
                    0
                ]),
                t_range=[-4, 4, 0.1],
                color="#4A90E2",
                stroke_width=2,
                stroke_opacity=0.3
            )
            waves.add(wave)

        # 添加波动效果
        self.context.play(Create(waves), run_time=0.3)

        # 保存波浪用于后续动画
        self.background_waves = waves

    def show_three_column_demonstration(self):
        """第二幕：三栏布局演示光的传播"""
        # 引用文本上移
        target_position = self.context.canvas["header_line"].get_bottom() + DOWN * 0.5
        small_quote = Text(
            "电磁波不是看不见摸不着的东西，恰恰相反，它是你唯一看得见摸得着的东西",
            font=font_heading,
            font_size=16,
            color="#FFD700"
        )
        small_quote.move_to(target_position)

        self.context.play(
            Transform(self.quote_text, small_quote),
            self.background_waves.animate.set_opacity(0.1),
            run_time=0.8
        )

        # 创建两栏内容
        self.create_left_column_vision()
        self.create_right_column_atoms()

    def create_left_column_vision(self):
        """左栏：视觉系统 - 灯泡、立方体、眼睛"""
        left_center = LEFT * 3.5

        # 立方体（简化实现，使用正方形代替，因为Cube类未实现）
        cube = Square(side_length=0.8, color="#87CEEB", fill_opacity=0.3, stroke_width=3)
        cube.move_to(left_center)

        # 灯泡
        light_bulb = LightBulb(size=0.6)
        light_bulb.move_to(left_center + LEFT * 1.5 + UP * 0.8)

        # 眼睛（使用Eye SVG类，高度与灯泡一致）
        eye_group = Eye(size=0.8)
        eye_group.move_to(left_center + RIGHT * 1.5 + UP * 0.8)  # 与灯泡同高度

        self.context.play(Create(cube), run_time=0.4)
        self.context.wait(0.3)
        self.context.next_slide()

        self.context.play(Create(light_bulb), run_time=0.4)
        self.context.wait(0.3)
        self.context.next_slide()

        # 从灯泡发射正弦波到立方体
        light_wave = ParametricFunction(
            lambda t: light_bulb.get_center() + t * (cube.get_center() - light_bulb.get_center()) +
                     UP * 0.1 * np.sin(10 * PI * t),
            t_range=[0, 1, 0.05],
            color="#FFFF00",
            stroke_width=3
        )

        self.context.play(Create(light_wave), run_time=0.6)
        self.context.wait(0.3)

        # 反射到眼睛
        reflection_wave = ParametricFunction(
            lambda t: cube.get_center() + t * (eye_group.get_center() - cube.get_center()) +
                     UP * 0.1 * np.sin(10 * PI * t),
            t_range=[0, 1, 0.05],
            color="#FFD700",
            stroke_width=3
        )

        self.context.play(Create(reflection_wave), run_time=0.6)
        self.context.wait(0.3)

        self.context.play(Create(eye_group), run_time=0.4)
        self.context.wait(0.3)

    def create_right_column_atoms(self):
        """右栏：原子间的电磁场"""
        right_center = RIGHT * 3.5

        # 第一个原子
        atom1 = Atom(size=0.8)
        atom1.move_to(right_center + LEFT * 1.0)

        # 第二个原子
        atom2 = Atom(size=0.8)
        atom2.move_to(right_center + RIGHT * 1.0)

        self.context.play(Create(atom1), run_time=0.4)
        self.context.wait(0.3)
        self.context.next_slide()

        self.context.play(Create(atom2), run_time=0.4)
        self.context.wait(0.3)
        self.context.next_slide()

        # 原子间的电磁波（从原子边界到边界）
        atom1_edge = atom1.get_right()
        atom2_edge = atom2.get_left()
        electromagnetic_field = ParametricFunction(
            lambda t: atom1_edge + t * (atom2_edge - atom1_edge) +
                     UP * 0.15 * np.sin(8 * PI * t),
            t_range=[0, 1, 0.05],
            color="#00FFFF",
            stroke_width=3
        )

        self.context.play(Create(electromagnetic_field), run_time=0.6)
        self.context.wait(0.3)
        self.context.next_slide()

        # 将两个原子向外推
        self.context.play(
            atom1.animate.shift(LEFT * 0.5),
            atom2.animate.shift(RIGHT * 0.5),
            electromagnetic_field.animate.scale(1.5),
            run_time=0.8
        )
        self.context.wait(0.3)
        self.context.next_slide()


class PerceptionChainSlideComponent(ListSlideComponent):
    """
    感知链条slide组件 - 使用列表基类
    """

    def __init__(self, context):
        romantic_poem_lines = [
            "大气散射后的阳光洒在少女身上",
            "捎来了关于她的消息",
            "这消息触及我的双眼",
            "挑动我的神经",
            "唤醒我的意识"
        ]
        super().__init__(context, "我看见少女在夕阳下奔跑", romantic_poem_lines)

    def get_list_style(self):
        """自定义列表样式"""
        return {
            "font_size": 20,
            "color": "#F0F8FF",
            "run_time": 0.8,
            "animation": "Write",
            "bullet": "circle",
            "bullet_size": 0.08,
            "bullet_color": "#FFD700"
        }


class CodingConceptsSlideComponent(TitleSlideComponent):
    """
    编码概念slide组件 - 使用3D场景
    """

    def __init__(self, context):
        super().__init__(context, "编码，转码，解码")

    def render_content(self):
        """渲染3D编码演示"""
        # 启用3D相机并固定UI元素
        self.context.enable_3d_camera_with_fixed_ui(phi=75*DEGREES, theta=-45*DEGREES)

        # 创建真正的3D场景
        self.create_3d_coding_demo()

    def create_3d_coding_demo(self, y_position=-2):
        """创建真正的3D编码层"""
        # 3D立方体
        encoding_cube = Cube(side_length=1, fill_opacity=0.3, stroke_width=1, color="#87CEEB")
        self.context.play(Create(encoding_cube), run_time=0.3)
        self.context.wait(0.3)
        self.context.next_slide()

        self.context.play(encoding_cube.animate.move_to([0, y_position - 0.5, 0]), run_time=0.5)
        self.context.wait(0.3)
        self.context.next_slide()

        arrows_3d_origin = VGroup()
        arrows_3d = VGroup()
        icons_3d = VGroup()
        icon_classes = [Binary, Sound, Photo]
        # 3 part of a circle
        directions_3d = [
            np.array([np.sin(0), 0, np.cos(0)]),
            np.array([np.sin(2 * PI/3), 0, np.cos(2 * PI/3)]),
            np.array([np.sin(4 * PI/3), 0, np.cos(4 * PI/3)]),
        ]

        # 3D箭头向外辐射
        for direction, icon_class in zip(directions_3d, icon_classes):
            # 3D箭头
            arrow_3d = Arrow3D(
                start=[0, y_position, 0],
                end=direction + np.array([0, -0.2, 0]),
                color="#FFFFFF",
                thickness=0.005, height=0.1, base_radius=0.03)
            arrows_3d_origin.add(Arrow3D(
                start=[0, y_position, 0], end=[0, y_position, 0], color="#FFFFFF",
                thickness=0.005, height=0.1, base_radius=0.03))
            arrows_3d.add(arrow_3d)

            # 图标（2D图标固定在相机视角中，显示在3D位置）
            icon = icon_class(size=0.6)
            icon.move_to(direction)
            icon.rotate(PI/2, axis=RIGHT)
            icons_3d.add(icon)

        for idx in range(len(arrows_3d)):
            self.context.add(arrows_3d_origin[idx])
            self.context.play(Transform(arrows_3d_origin[idx], arrows_3d[idx]), run_time=0.3)
            self.context.play(Create(icons_3d[idx]), run_time=0.3)
            self.context.wait(0.3)
            self.context.next_slide()

        connect_circle = Circle(radius=1, color="#FFFFFF")
        connect_circle.move_to([0, 0, 0])
        connect_circle.rotate(PI/2, axis=RIGHT)

        self.context.play(Create(connect_circle), run_time=0.5)
        self.context.wait(0.3)
        self.context.next_slide()

        arrows_3d_decode_origin = VGroup()
        arrows_3d_decode = VGroup()

        for direction in directions_3d:
            arrow_3d_decode = Arrow3D(
                start=direction + np.array([0, 0.2, 0]),
                end=[0, -y_position, 0],
                color="#FFFFFF",
                thickness=0.005, height=0.1, base_radius=0.03)
            arrows_3d_decode_origin.add(Arrow3D(
                start=direction + np.array([0, 0.2, 0]),
                end=direction + np.array([0, 0.2, 0]),
                color="#FFFFFF",
                thickness=0.005, height=0.1, base_radius=0.03))
            arrows_3d_decode.add(arrow_3d_decode)

        transforms = [Transform(arrows_3d_decode_origin[idx], arrows_3d_decode[idx]) for idx in range(len(arrows_3d_decode))]
        self.context.play(*transforms, run_time=0.3)
        decoding_cube = Cube(
            side_length=1, fill_opacity=0.3, stroke_width=1, color="#FFD700")
        decoding_cube.move_to([0, -y_position + 0.5, 0])

        self.context.play(Create(decoding_cube), run_time=0.3)
        self.context.wait(0.3)
        self.context.next_slide()

        decoding_align_arrow = Arrow3D(
            start=[0, y_position, 0],
            end=[0, -y_position, 0],
            color="#FF6B6B",
            thickness=0.005, height=0.1, base_radius=0.03)
        self.context.play(Create(decoding_align_arrow), run_time=0.3)

        decoding_question_mark = Text("?", font=font_heading, font_size=24, color="#FF6B6B")
        decoding_question_mark.rotate(PI/2, axis=RIGHT)
        self.context.play(Create(decoding_question_mark), run_time=0.3)
        self.context.wait(0.3)
        self.context.next_slide()

        origin_question_mark = Text("?", font=font_heading, font_size=24, color="#FF6B6B")
        origin_question_mark.rotate(PI/2, axis=RIGHT)
        origin_question_mark.move_to([0, 2 * y_position - 0.5, 0])
        self.context.play(Create(origin_question_mark), run_time=0.3)
        self.context.wait(0.3)
        origin_arrow = Arrow3D(
            start=[0, 2 * y_position, 0],
            end=[0, y_position - 1, 0],
            color="#FFFFFF",
            thickness=0.005, height=0.1, base_radius=0.03)
        self.context.play(Create(origin_arrow), run_time=0.3)
        self.context.wait(0.3)

        self.context.next_slide()

        question_mark_group = VGroup()
        question_mark_encode = Text(
            "?", font=font_heading, font_size=28, color="#FF6B6B")
        question_mark_encode.rotate(PI/2, axis=RIGHT)
        question_mark_encode.move_to([0, y_position - 0.5, 0])
        for direction in directions_3d:
            question_mark_encode_direction = Text(
                "?", font=font_heading, font_size=28, color="#FF6B6B")
            question_mark_encode_direction.rotate(PI/2, axis=RIGHT)
            question_mark_encode_direction.move_to(direction * 1.4)
            question_mark_group.add(question_mark_encode_direction)
        question_mark_decoded = Text(
            "?", font=font_heading, font_size=28, color="#FF6B6B")
        question_mark_decoded.rotate(PI/2, axis=RIGHT)
        question_mark_decoded.move_to([0, -y_position + 0.5, 0])
        question_mark_group.add(question_mark_encode)
        question_mark_group.add(question_mark_decoded)
        self.context.play(Create(question_mark_group), run_time=0.5)
        self.context.wait(0.3)
        self.context.next_slide()


class InformationRealismSlideComponent(TitleSlideComponent):
    """
    信息实在论slide组件
    "'存在'存在于何处？"
    左右分栏布局：左侧问题列表，右侧几何演示
    """

    def __init__(self, context):
        super().__init__(context, "'存在'存在于何处？")

    def render_content(self):
        """渲染信息实在论内容"""
        # 重置到2D相机
        self.context.reset_to_2d_camera()

        # 创建左右分栏
        self.create_left_column_questions()
        self.create_right_column_geometry_demos()

    def create_left_column_questions(self):
        """创建左侧问题列表（直接在目标位置显示小字，横向排版）"""
        # 哲学问题列表
        questions = [
            "灵魂存在吗？",
            "海浪存在吗？",
            "电子游戏世界存在吗？"
        ]

        # 创建横向排列的问题文本，直接放在目标位置（小字，靠近标题）
        self.question_texts = []
        for question in questions:
            question_text = Text(question, font=font_main_text, font_size=14, color="#87CEEB")
            self.question_texts.append(question_text)

        # 横向排列，间距紧凑
        questions_group = VGroup(*self.question_texts).arrange(RIGHT, buff=0.6)

        # 直接定位在目标位置（标题下方很近的位置）
        title_bottom = self.context.canvas.get("title", self.small_title).get_bottom()[1] if self.context.canvas.get("title") else self.context.canvas["header_line"].get_bottom()[1] - 0.3
        target_position = title_bottom - 0.2
        questions_group.move_to([0, target_position, 0])

        # 显示横向排列的问题
        for question_text in self.question_texts:
            self.context.play(Write(question_text), run_time=0.5)
            self.context.wait(0.3)
            self.context.next_slide()

        # 保存问题组
        self.questions_group = questions_group

    def create_right_column_geometry_demos(self):
        """创建右侧几何演示"""
        right_center = RIGHT * 3.5

        # 长方形演示
        self.create_rectangle_demo(right_center + UP * 1.5)
        self.context.next_slide()

        # 2D点演示
        self.create_2d_point_demo(right_center + DOWN * 1.5)
        self.context.next_slide()

        # 最后同时播放两个循环动画
        self.add_synchronized_change_animations()

        # 在所有演示完毕后，添加问题上推和interactive_list
        self.add_existence_questions_analysis()

    def create_rectangle_demo(self, position):
        """长方形的定义演示，使用BraceAnnotation标注"""
        # 创建长方形
        self.rect_width = 2.0
        self.rect_height = 1.2
        self.rectangle = Rectangle(width=self.rect_width, height=self.rect_height,
                                  color="#4A90E2", stroke_width=2)
        self.rectangle.move_to(position)
        self.rect_position = position

        self.context.play(Create(self.rectangle), run_time=0.5)
        self.context.wait(0.3)
        self.context.next_slide()

        # 使用BraceAnnotation标注长宽
        self.width_brace = BraceLabel(self.rectangle, f"{self.rect_width:.1f}", brace_direction=DOWN,
                                     label_constructor=MathTex, font_size=24,
                                     buff=0.1, color="#FFD700")
        self.height_brace = BraceLabel(self.rectangle, f"{self.rect_height:.1f}", brace_direction=LEFT,
                                      label_constructor=MathTex, font_size=24,
                                      buff=0.1, color="#FFD700")

        self.context.play(Create(self.width_brace), run_time=0.4)
        self.context.play(Create(self.height_brace), run_time=0.4)
        self.context.wait(0.3)

        # 计算并显示第一组向量：长宽向量
        self.rectangle_vector1 = MathTex(f"({self.rect_width:.1f}, {self.rect_height:.1f})",
                                        font_size=20, color="#FFD700")
        self.rectangle_vector1.next_to(self.rectangle, RIGHT, buff=0.8)

        self.context.play(Write(self.rectangle_vector1), run_time=0.4)
        self.context.wait(0.3)
        self.context.next_slide()

        # 添加对角线
        self.diagonal = Line(self.rectangle.get_corner(DL), self.rectangle.get_corner(UR),
                            color="#FF6B6B", stroke_width=2)

        # 计算对角线长度
        import math
        diagonal_length = math.sqrt(self.rect_width**2 + self.rect_height**2)

        # 创建对角线的正确Brace方向（沿着对角线的法向量）
        diagonal_direction = self.rectangle.get_corner(UR) - self.rectangle.get_corner(DL)
        diagonal_normal = np.array([-diagonal_direction[1], diagonal_direction[0], 0])
        diagonal_normal = diagonal_normal / np.linalg.norm(diagonal_normal)

        # 使用简单的线和文本标注对角线（避免Brace方向问题）
        diagonal_mid = (self.rectangle.get_corner(DL) + self.rectangle.get_corner(UR)) / 2
        self.diagonal_label = MathTex(f"{diagonal_length:.1f}", font_size=20, color="#FF6B6B")
        self.diagonal_label.move_to(diagonal_mid + diagonal_normal * 0.3)

        self.context.play(Create(self.diagonal), Write(self.diagonal_label), run_time=0.4)
        self.context.wait(0.3)
        self.context.next_slide()

        # 计算并显示第二组向量：长宽比和对角线长度
        aspect_ratio = self.rect_width / self.rect_height
        self.rectangle_vector2 = MathTex(f"({aspect_ratio:.2f}, {diagonal_length:.1f})",
                                        font_size=20, color="#FF6B6B")
        self.rectangle_vector2.next_to(self.rectangle_vector1, DOWN, buff=0.3)

        self.context.play(Write(self.rectangle_vector2), run_time=0.4)
        self.context.wait(0.3)
        self.context.next_slide()

        # 保存组件用于后续动画
        self.rect_braces_group = VGroup(self.width_brace, self.height_brace, self.diagonal_label)
        self.rect_aux_lines = VGroup(self.diagonal)

    def create_2d_point_demo(self, position):
        """2D点的直角坐标与极坐标演示"""
        # 创建2D坐标系
        self.axes = Axes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            x_length=2.5,
            y_length=2.5,
            tips=False,
            axis_config={"color": "#4A90E2"}
        )
        self.axes.move_to(position)

        self.context.play(Create(self.axes), run_time=0.5)
        self.context.wait(0.3)

        # 添加点
        self.point_coords = [1.0, 0.8]  # (x, y)
        self.point_dot = Dot(self.axes.coords_to_point(self.point_coords[0], self.point_coords[1]),
                            color="#FFD700", radius=0.06)
        self.context.play(Create(self.point_dot), run_time=0.3)
        self.context.wait(0.3)
        self.context.next_slide()

        # 直角坐标标注
        self.x_line = DashedLine(
            self.axes.coords_to_point(self.point_coords[0], 0),
            self.axes.coords_to_point(self.point_coords[0], self.point_coords[1]),
            color="#FF4444", stroke_width=2
        )
        self.y_line = DashedLine(
            self.axes.coords_to_point(0, self.point_coords[1]),
            self.axes.coords_to_point(self.point_coords[0], self.point_coords[1]),
            color="#44FF44", stroke_width=2
        )

        self.x_brace = BraceLabel(
            Line(self.axes.coords_to_point(0, 0), self.axes.coords_to_point(self.point_coords[0], 0)),
            f"{abs(self.point_coords[0]):.1f}", brace_direction=DOWN, label_constructor=MathTex,
            font_size=16, buff=0.1, color="#FF4444"
        )
        self.y_brace = BraceLabel(
            Line(self.axes.coords_to_point(0, 0), self.axes.coords_to_point(0, self.point_coords[1])),
            f"{abs(self.point_coords[1]):.1f}", brace_direction=LEFT, label_constructor=MathTex,
            font_size=16, buff=0.1, color="#44FF44"
        )

        self.context.play(Create(self.x_line), Create(self.y_line), run_time=0.4)
        self.context.play(Create(self.x_brace), Create(self.y_brace), run_time=0.4)
        self.context.wait(0.3)

        # 计算并显示第一组向量：直角坐标
        self.point_vector1 = MathTex(f"({self.point_coords[0]:.1f}, {self.point_coords[1]:.1f})",
                                    font_size=20, color="#FFD700")
        self.point_vector1.next_to(self.axes, RIGHT, buff=0.8)

        self.context.play(Write(self.point_vector1), run_time=0.4)
        self.context.wait(0.3)
        self.context.next_slide()

        # 极坐标标注
        import math
        r = math.sqrt(self.point_coords[0]**2 + self.point_coords[1]**2)
        theta = math.atan2(self.point_coords[1], self.point_coords[0])

        # 半径线
        self.radius_line = Line(self.axes.coords_to_point(0, 0), self.point_dot.get_center(),
                               color="#FF6B6B", stroke_width=3)

        # 角度弧
        self.angle_arc = Arc(radius=0.3, start_angle=0, angle=theta, color="#87CEEB", stroke_width=2)
        self.angle_arc.move_arc_center_to(self.axes.coords_to_point(0, 0))

        # 使用简单标签替代复杂的BraceLabel（避免方向问题）
        radius_mid = (self.axes.coords_to_point(0, 0) + self.point_dot.get_center()) / 2
        self.radius_label = MathTex(f"{r:.1f}", font_size=16, color="#FF6B6B")
        self.radius_label.move_to(radius_mid + UP * 0.2)

        self.angle_label = MathTex(f"{theta:.1f}", font_size=16, color="#87CEEB")
        self.angle_label.next_to(self.angle_arc, UR, buff=0.1)

        self.context.play(Create(self.radius_line), Create(self.angle_arc), run_time=0.4)
        self.context.play(Write(self.radius_label), Write(self.angle_label), run_time=0.4)
        self.context.wait(0.3)
        self.context.next_slide()

        # 计算并显示第二组向量：极坐标
        self.point_vector2 = MathTex(f"({r:.1f}, {theta:.1f})", font_size=20, color="#FF6B6B")
        self.point_vector2.next_to(self.point_vector1, DOWN, buff=0.3)

        self.context.play(Write(self.point_vector2), run_time=0.4)
        self.context.wait(0.3)
        self.context.next_slide()

        # 保存组件用于后续动画
        self.point_braces_group = VGroup(self.x_brace, self.y_brace, self.radius_label, self.angle_label)
        self.point_aux_lines = VGroup(self.x_line, self.y_line, self.radius_line, self.angle_arc)

    def add_existence_questions_analysis(self):
        """添加存在问题分析：分界线、左栏内容"""
        # 1. 画分界线（左栏中央）
        available_top = self.context.canvas["header_line"].get_bottom()[1] - 0.5
        available_bottom = self.context.canvas["footer_line"].get_top()[1] + 0.5
        divider_y = available_top - (available_top - available_bottom) * 0.5

        divider_line = Line([-5.5, divider_y, 0], [-1.5, divider_y, 0],
                           color="#4A90E2", stroke_width=2)

        self.context.play(Create(divider_line), run_time=0.4)
        self.context.wait(0.3)
        self.context.next_slide()

        # 2. 左栏内容：信息定义在分界线上方，编码三层结构在分界线下方
        self.create_left_column_content(divider_y, available_top, available_bottom)

    def create_left_column_content(self, divider_y, available_top, available_bottom):
        """创建左栏内容：信息定义在分界线上方，编码三层结构在分界线下方"""
        # 信息定义：分界线上方
        info_content = [
            (None, "\"数\"的数量", None),
            (None, "  —— 最小信息量不变", None),
            (None, "\"数\"的编码-转码之间总是遵循某种关系", None),
            (None, "  —— 信息以某种结构加以组织", None),
            (None, "结构隐含在\"编/转\"码过程中", None),
            (None, "  —— \"类型\" 以信息的构造以及信息转换的结构描述信息", None),
        ]

        def position_info(items):
            content_group = VGroup(*items).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
            # 在分界线上方居中
            upper_center_y = divider_y + (available_top - divider_y) * 0.5
            content_group.move_to([-3.5, upper_center_y, 0])

        self.interactive_list(
            info_content,
            position_info,
            font_size=16,
            color="#F0F8FF",  # 白色字
            run_time=0.8,
            animation="Write",
            bullet="circle",
            bullet_size=0.05,
            bullet_color="#FFD700"
        )
        self.context.wait(0.5)
        self.context.next_slide()

        # 编码三层结构：分界线下方
        encoding_content = [
            (None, "框架编码：这是一个有意义的编码", None),
            (None, "外在编码：这个编码该如何转码/解码", None),
            (None, "内在编码：这个编码指向的\"具体内容\"", None)
        ]

        def position_encoding(items):
            content_group = VGroup(*items).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
            # 在分界线下方的上半部分
            lower_center_y = divider_y - (divider_y - available_bottom) * 0.3
            content_group.move_to([-3.5, lower_center_y, 0])

        self.interactive_list(
            encoding_content,
            position_encoding,
            font_size=16,
            color="#F0F8FF",  # 白色字
            run_time=0.8,
            animation="Write",
            bullet="circle",
            bullet_size=0.05,
            bullet_color="#FFD700"
        )

        # 核心概念
        core_concept = Text(
            "当\"外在编码\"和\"内在编码\"齐备时，\"存在\"便得以存在",
            font=font_heading, font_size=18, color="#FF6B6B", weight=BOLD
        )
        core_concept.move_to([-3.5, available_bottom + 0.8, 0])
        self.context.play(Write(core_concept), run_time=1.2)
        self.context.wait(0.3)
        self.context.next_slide()

        # 核心问题：如何打破无穷嵌套的编码循环？
        core_question = Text(
            "我们如何打破无穷嵌套的编码循环？",
            font=font_heading, font_size=18, color="#FFD700", weight=BOLD
        )
        core_question.move_to([-3.5, available_bottom + 0.3, 0])
        self.context.play(Write(core_question), run_time=1.2)
        self.context.wait(0.3)
        self.context.next_slide()

    def add_synchronized_change_animations(self):
        """同时播放长方形和点的变化动画"""

        self.context.next_slide()
        # 只在第一次动画时隐藏辅助线
        self.context.play([
            FadeOut(self.rect_braces_group),
            FadeOut(self.rect_aux_lines),
            FadeOut(self.point_braces_group),
            FadeOut(self.point_aux_lines)
        ], run_time=0.5)

        self.context.next_slide(loop=True)

        import math

        # 定义同步的变化序列：变化两次后第三次回到初始状态
        target_dimensions = [
            (1.5, 1.8),  # 第一次变化
            (2.5, 0.8),  # 第二次变化
            (2.0, 1.2)   # 第三次变化：回到初始状态
        ]

        target_positions = [
            [0.5, 1.2],  # 第一次变化
            [-0.8, 0.6], # 第二次变化
            [1.0, 0.8]   # 第三次变化：回到初始状态
        ]

        for (new_width, new_height), new_coords in zip(target_dimensions, target_positions):
            # 准备长方形动画
            new_rectangle = Rectangle(width=new_width, height=new_height,
                                    color="#4A90E2", stroke_width=2)
            new_rectangle.move_to(self.rect_position)

            new_diagonal_length = math.sqrt(new_width**2 + new_height**2)
            new_aspect_ratio = new_width / new_height

            new_rect_vector1 = MathTex(f"({new_width:.1f}, {new_height:.1f})",
                                      font_size=20, color="#FFD700")
            new_rect_vector1.move_to(self.rectangle_vector1.get_center())

            new_rect_vector2 = MathTex(f"({new_aspect_ratio:.2f}, {new_diagonal_length:.1f})",
                                      font_size=20, color="#FF6B6B")
            new_rect_vector2.move_to(self.rectangle_vector2.get_center())

            # 准备点动画
            r_new = math.sqrt(new_coords[0]**2 + new_coords[1]**2)
            theta_new = math.atan2(new_coords[1], new_coords[0])

            new_point_pos = self.axes.coords_to_point(new_coords[0], new_coords[1])

            new_point_vector1 = MathTex(f"({new_coords[0]:.1f}, {new_coords[1]:.1f})",
                                       font_size=20, color="#FFD700")
            new_point_vector1.move_to(self.point_vector1.get_center())

            new_point_vector2 = MathTex(f"({r_new:.1f}, {theta_new:.1f})",
                                       font_size=20, color="#FF6B6B")
            new_point_vector2.move_to(self.point_vector2.get_center())

            # 同时执行两个动画
            animations = [
                # 长方形动画
                Transform(self.rectangle, new_rectangle),
                Transform(self.rectangle_vector1, new_rect_vector1),
                Transform(self.rectangle_vector2, new_rect_vector2),
                # 点动画
                self.point_dot.animate.move_to(new_point_pos),
                Transform(self.point_vector1, new_point_vector1),
                Transform(self.point_vector2, new_point_vector2),
            ]
            self.context.play(*animations, run_time=1.2)
            self.context.wait(0.5)

            # 更新当前状态
            self.rect_width = new_width
            self.rect_height = new_height
            self.point_coords = new_coords

        self.context.next_slide()


class IntelligenceHeatTransferSlideComponent(ColumnLayoutSlideComponent):
    """
    智能热传递slide组件 - 使用两栏布局
    """

    def __init__(self, context):
        super().__init__(context, "智能的\"热传递\"视角", num_columns=2)

    def render_columns(self):
        """渲染两栏内容"""
        # 左栏：物理热传递
        self.render_column_content(0, self._render_physical_heat_transfer)

        # 右栏：智能学习
        self.render_column_content(1, self._render_intelligence_learning)

    def _render_physical_heat_transfer(self, position):
        """左栏：杯子+铁球的热传递"""
        # 1. 画杯子（没有上横线的矩形）
        cup_width = 1.2
        cup_height = 1.5
        cup = VGroup(
            Line([position[0] - cup_width/2, position[1] - cup_height/2, 0],
                 [position[0] - cup_width/2, position[1] + cup_height/2, 0], color="#87CEEB", stroke_width=3),
            Line([position[0] - cup_width/2, position[1] - cup_height/2, 0],
                 [position[0] + cup_width/2, position[1] - cup_height/2, 0], color="#87CEEB", stroke_width=3),
            Line([position[0] + cup_width/2, position[1] - cup_height/2, 0],
                 [position[0] + cup_width/2, position[1] + cup_height/2, 0], color="#87CEEB", stroke_width=3)
        )

        self.context.play(Create(cup), run_time=0.8)
        self.context.wait(0.3)
        self.context.next_slide()

        # 2. 涨水动画（填充矩形）
        water_target_height = cup_height * 2/3
        water = Rectangle(
            width=cup_width - 0.05,
            height=water_target_height,
            color="#4A90E2",
            fill_opacity=0.6,
            stroke_width=0
        )
        water.move_to([position[0], position[1] - cup_height/2 + water_target_height/2, 0])

        # 从高度0开始的水
        water_start = Rectangle(
            width=cup_width - 0.05,
            height=0.01,
            color="#4A90E2",
            fill_opacity=0.6,
            stroke_width=0
        )
        water_start.move_to([position[0], position[1] - cup_height/2, 0])

        self.context.play(Create(water_start), run_time=0.2)
        self.context.play(Transform(water_start, water), run_time=1.2)
        self.context.wait(0.3)
        self.context.next_slide()

        # 3. 红色铁球从上方出现
        ball_start_pos = [position[0], position[1] + cup_height/2 + 0.8, 0]
        iron_ball = Circle(radius=0.15, color="#FF4444", fill_opacity=0.8, stroke_width=2)
        iron_ball.move_to(ball_start_pos)

        self.context.play(Create(iron_ball), run_time=0.5)
        self.context.wait(0.3)
        self.context.next_slide()

        # 4. 铁球下落到杯子里（沉到杯底）
        ball_target_pos = [position[0], position[1] - cup_height/2 + 0.15, 0]
        self.context.play(iron_ball.animate.move_to(ball_target_pos), run_time=1.0)
        self.context.wait(0.3)
        self.context.next_slide()

        # 5. 红色球逐渐变成灰色球
        gray_ball = Circle(radius=0.15, color="#888888", fill_opacity=0.8, stroke_width=2)
        gray_ball.move_to(ball_target_pos)
        self.context.play(Transform(iron_ball, gray_ball), run_time=1.5)
        self.context.wait(0.5)
        self.context.next_slide()

    def _render_intelligence_learning(self, position):
        """右栏：智能学习过程"""
        # 右栏上方：客观世界（圆形）
        objective_world = Circle(radius=0.4, color="#FFD700", fill_opacity=0.3, stroke_width=3)
        objective_world.move_to([position[0], position[1] + 1.0, 0])

        # 右栏下方：主观世界（方形）
        subjective_world = Square(side_length=0.8, color="#87CEEB", fill_opacity=0.3, stroke_width=3)
        subjective_world.move_to([position[0], position[1] - 1.0, 0])

        self.context.play(Create(objective_world), Create(subjective_world), run_time=0.8)
        self.context.wait(0.3)
        self.context.next_slide()

        # 学习过程：方形逐渐变成圆形
        learning_circle = Circle(radius=0.4, color="#87CEEB", fill_opacity=0.3, stroke_width=3)
        learning_circle.move_to([position[0], position[1] - 1.0, 0])

        self.context.play(Transform(subjective_world, learning_circle), run_time=1.5)
        self.context.wait(0.5)
        self.context.next_slide()

        # 回归方形（比热有限）
        back_to_square = Square(side_length=0.8, color="#87CEEB", fill_opacity=0.3, stroke_width=3)
        back_to_square.move_to([position[0], position[1] - 1.0, 0])

        # 添加旁白文字
        narration1 = Text("但你的比热并不是无穷小的", font=font_main_text, font_size=16, color="#F0F8FF")
        narration1.next_to([position[0], position[1], 0], RIGHT, buff=0.8)
        narration2 = Text("世界的比热也并不是无穷大的", font=font_main_text, font_size=16, color="#F0F8FF")
        narration2.next_to(narration1, DOWN, buff=0.2)

        self.context.play(
            Transform(subjective_world, back_to_square),
            Write(narration1), Write(narration2),
            run_time=1.5
        )
        self.context.wait(0.8)
        self.context.next_slide()

        # 最终状态：两个形状都变成0.7圆-0.3方的插值
        # 创建插值形状：真正的圆形和方形的插值
        def create_interpolated_shape(center_pos, t=0.7, color="#FF6B6B"):
            # t=0为方形，t=1为圆形，t=0.7为70%圆形+30%方形
            # 确保两个形状有相同的顶点数和顺序

            import numpy as np

            # 统一使用32个顶点，确保顶点起始位置和方向一致
            num_vertices = 32
            radius = 0.4

            # 创建圆形的顶点（从右侧开始，逆时针）
            circle_vertices = []
            for i in range(num_vertices):
                angle = 2 * PI * i / num_vertices  # 从0开始，逆时针
                x = center_pos[0] + radius * np.cos(angle)
                y = center_pos[1] + radius * np.sin(angle)
                circle_vertices.append(np.array([x, y, 0]))

            # 创建方形的顶点（从右侧开始，逆时针，与圆形对应）
            square_vertices = []
            square_radius = radius  # 使用相同的外接圆半径

            for i in range(num_vertices):
                # 将圆形的角度映射到方形的周长上
                angle = 2 * PI * i / num_vertices

                # 根据角度确定在方形的哪条边上
                # 方形的四个角对应的角度：0, π/2, π, 3π/2
                # 修复边界条件：使用更清晰的角度范围判断
                if angle < PI/4 or angle >= 7*PI/4:  # 右边：[-π/4, π/4)
                    # 将角度转换到[-π/4, π/4]范围
                    if angle >= 7*PI/4:
                        angle_norm = angle - 2*PI  # 转换为负角度
                    else:
                        angle_norm = angle
                    # 从-π/4到π/4，映射到y坐标从-1到1
                    t_local = (angle_norm + PI/4) / (PI/2)
                    x = center_pos[0] + square_radius
                    y = center_pos[1] + square_radius * (2 * t_local - 1)
                elif angle < 3*PI/4:  # 上边：[π/4, 3π/4)
                    # 从π/4到3π/4，映射到x坐标从1到-1
                    t_local = (angle - PI/4) / (PI/2)
                    x = center_pos[0] + square_radius * (1 - 2 * t_local)
                    y = center_pos[1] + square_radius
                elif angle < 5*PI/4:  # 左边：[3π/4, 5π/4)
                    # 从3π/4到5π/4，映射到y坐标从1到-1
                    t_local = (angle - 3*PI/4) / (PI/2)
                    x = center_pos[0] - square_radius
                    y = center_pos[1] + square_radius * (1 - 2 * t_local)
                else:  # 下边：[5π/4, 7π/4)
                    # 从5π/4到7π/4，映射到x坐标从-1到1
                    t_local = (angle - 5*PI/4) / (PI/2)
                    x = center_pos[0] + square_radius * (2 * t_local - 1)
                    y = center_pos[1] - square_radius

                square_vertices.append(np.array([x, y, 0]))

            # 创建VMobjects并设置points（使用正确的贝塞尔曲线格式）
            square_mob = VMobject(color=color, fill_opacity=0.3, stroke_width=3)
            circle_mob = VMobject(color=color, fill_opacity=0.3, stroke_width=3)

            # 将顶点转换为贝塞尔曲线格式（每个线段需要4个点：起点、控制点1、控制点2、终点）
            def vertices_to_bezier_points(vertices):
                bezier_points = []
                for i in range(len(vertices)):
                    start = vertices[i]
                    end = vertices[(i + 1) % len(vertices)]
                    # 对于直线段，控制点就是起点和终点的1/3和2/3位置
                    control1 = start + (end - start) / 3
                    control2 = start + 2 * (end - start) / 3
                    bezier_points.extend([start, control1, control2, end])
                return np.array(bezier_points)

            square_mob.points = vertices_to_bezier_points(square_vertices)
            circle_mob.points = vertices_to_bezier_points(circle_vertices)

            # 使用插值创建混合形状
            interpolated = VMobject()
            interpolated.interpolate(square_mob, circle_mob, t)
            interpolated.set_color(color)
            interpolated.set_fill(opacity=0.3)
            interpolated.set_stroke(width=3)
            return interpolated

        final_obj_shape = create_interpolated_shape([position[0], position[1] + 1.0, 0], 0.7, "#FFD700")
        final_subj_shape = create_interpolated_shape([position[0], position[1] - 1.0, 0], 0.7, "#87CEEB")

        # 添加最终旁白
        narration3 = Text("所以你有了主观能动性", font=font_main_text, font_size=18, color="#FF6B6B", weight=BOLD)
        narration3.move_to([position[0] + 1.5, position[1], 0])

        self.context.play(
            Transform(objective_world, final_obj_shape),
            Transform(subjective_world, final_subj_shape),
            FadeOut(narration1), FadeOut(narration2),
            Write(narration3),
            run_time=2.0
        )
        self.context.wait(1.0)
        self.context.next_slide()


class MathPhysicsWorldSlideComponent(ListSlideComponent):
    """
    数学物理世界slide组件 - 使用列表基类
    """

    def __init__(self, context):
        concepts = [
            "数学描述信息的结构",
            "数学与物理的组合，即是在对信息的描述之上编码我们的世界",
            "人类对于世界的感知，来自物理世界到精神世界的层层转码"
        ]
        super().__init__(context, "数学，物理与我们所知的世界", concepts)

    def get_list_style(self):
        """自定义列表样式"""
        return {
            "font_size": 22,
            "color": "#87CEEB",
            "run_time": 1.0,
            "animation": "Write",
            "bullet": "circle",
            "bullet_size": 0.06,
            "bullet_color": "#FFD700"
        }


class TranscodingToolsSlideComponent(TitleSlideComponent):
    """
    转码工具slide组件
    """

    def __init__(self, context):
        super().__init__(context, "以转码为剑")

    def render_content(self):
        """渲染转码演示"""
        # 计算可用空间
        available_top = self.small_title.get_bottom()[1] - 0.5
        available_bottom = self.context.canvas["footer_line"].get_top()[1] + 0.3
        center_y = (available_top + available_bottom) / 2

        # 三栏位置
        left_center = LEFT * 4 + UP * center_y
        center_pos = ORIGIN + UP * center_y
        right_center = RIGHT * 4 + UP * center_y

        # 左栏：代数方程 x^2 + y^2 = 1
        equation = MathTex("x^2 + y^2 = 1", font_size=32, color="#87CEEB")
        equation.move_to(left_center)

        self.context.play(Write(equation), run_time=0.8)
        self.context.wait(0.3)
        self.context.next_slide()

        # 中栏：直角坐标系下的单位圆
        axes = Axes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            x_length=2.5,
            y_length=2.5,
            tips=False,
            axis_config={"color": "#4A90E2"}
        )
        axes.move_to(center_pos)

        unit_circle = Circle(radius=1.25, color="#FFD700", stroke_width=3)
        unit_circle.move_to(center_pos)

        self.context.play(Create(axes), run_time=0.5)
        self.context.play(Create(unit_circle), run_time=0.6)
        self.context.wait(0.3)
        self.context.next_slide()

        # 右栏：极坐标方程 r = 1
        polar_equation = MathTex("r = 1", font_size=32, color="#FF6B6B")
        polar_equation.move_to(right_center)

        self.context.play(Write(polar_equation), run_time=0.8)
        self.context.wait(0.5)
        self.context.next_slide()


class SummarySlideComponent(ListSlideComponent):
    """
    总结slide组件 - 使用列表基类
    """

    def __init__(self, context):
        summary_content = [
            "我们不知道\"真实的本源\"是什么",
            "但\"存在\"存在于我们的\"表示\"中",
            "找到合适的编码，我们即可更高效地接近\"真实\"",
            "当你通过合适的\"编码-转码\"，去描述信息的结构与结构的变换，你就得到了\"模型\""
        ]
        super().__init__(context, "……我们所知道的", summary_content)

    def get_list_style(self):
        """自定义列表样式"""
        return {
            "font_size": 22,
            "color": "#F0F8FF",
            "run_time": 1.0,
            "animation": "Write",
            "bullet": "circle",
            "bullet_size": 0.08,
            "bullet_color": "#FFD700"
        }


class DetailedReadingSlideComponent(ColumnLayoutSlideComponent):
    """
    拓展阅读slide组件 - 使用三栏布局
    """

    def __init__(self, context):
        super().__init__(context, "拓展阅读（书籍）", num_columns=3)

        # 定义三栏书籍内容
        self.book_data = [
            ("入门级", [
                "《哲学导论》",
                "《哥德尔、艾舍尔、巴赫》",
                "《信息论：本质，多样性，统一》"
            ]),
            ("进阶级", [
                "《计算机程序的构造与解释》",
                "《The Fabric of Reality》"
            ]),
            ("专业级", [
                "《Elements of Information Theory》",
                "《The Conscious Mind》"
            ])
        ]

    def render_columns(self):
        """渲染三栏书籍内容"""
        for i, (level_name, book_list) in enumerate(self.book_data):
            self.render_column_content(i, lambda pos, level=level_name, books=book_list:
                                     self._render_book_column(pos, level, books))

    def _render_book_column(self, position, level_name, book_list):
        """渲染单个书籍栏"""
        # 分区标题
        section_title = Text(level_name, font=font_heading, font_size=20, color="#87CEEB")
        section_title.move_to([position[0], position[1] + 1.5, 0])
        self.context.play(FadeIn(section_title), run_time=0.5)
        self.context.wait(0.3)
        self.context.next_slide()

        # 书籍内容
        book_items = [(None, book, None) for book in book_list]

        def position_books(items):
            content_group = VGroup(*items).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
            content_group.next_to(section_title, DOWN, buff=0.5)
            content_group.move_to([position[0], content_group.get_y(), 0])

        self.interactive_list(
            book_items,
            position_books,
            font_size=16,
            color="#F0F8FF",
            run_time=0.8,
            animation="Write",
            bullet="circle",
            bullet_size=0.05,
            bullet_color="#FFD700"
        )


class NextPreviewSlideComponent(TitleSlideComponent):
    """
    下集预告slide组件
    """

    def __init__(self, context):
        super().__init__(context, "下集预告")

    def render_content(self):
        """渲染预告内容"""
        # 核心问题
        questions = [
            "在编码的世界中，我们如何看待计算机的'存储'与智能的'记忆'？",
            "系统如何自发地产生一段'编码'，去指代对应的信息？"
        ]

        for i, question in enumerate(questions):
            question_text = Text(question, font=font_main_text, font_size=20, color="#87CEEB")
            question_text.next_to(self.small_title, DOWN, buff=1.5 + i * 1.0)
            question_text.move_to([0, question_text.get_y(), 0])
            self.context.play(Write(question_text), run_time=1.5)
            self.context.wait(0.5)
            self.context.next_slide()

        # 下集标题
        next_title = Text("下一集：《存储与记忆：\"事实\"如何被唤起？》",
                         font=font_heading, font_size=24, color="#FF6B6B", weight=BOLD)
        next_title.next_to(self.small_title, DOWN, buff=4.0)
        self.context.play(FadeIn(next_title), run_time=1.2)
        self.context.wait(0.8)
        self.context.next_slide()


class Prelude(SlideWithCover):
    """
    使用新架构的Prelude类 - 完全重构版本
    """

    def construct(self):
        # 添加封面页
        self.add_cover("序曲：我们真的能抵达世界的'真实'吗？")

        # 添加各种组件
        self.slide_manager.add_component(ElectromagneticQuoteSlideComponent)
        self.slide_manager.add_component(PerceptionChainSlideComponent)
        self.slide_manager.add_component(CodingConceptsSlideComponent)
        self.slide_manager.add_component(InformationRealismSlideComponent)
        self.slide_manager.add_component(IntelligenceHeatTransferSlideComponent)
        self.slide_manager.add_component(MathPhysicsWorldSlideComponent)
        self.slide_manager.add_component(TranscodingToolsSlideComponent)
        self.slide_manager.add_component(SummarySlideComponent)
        self.slide_manager.add_component(DetailedReadingSlideComponent)
        self.slide_manager.add_component(NextPreviewSlideComponent)
        self.slide_manager.add_component(ThanksSlideComponent)

        # 执行所有组件
        self.slide_manager.simple_execute_all()