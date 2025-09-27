from manim import *
from src.SlideFrames import *
from src.configs import *

class GangingUpSlideComponent(TitleSlideComponent):
    def __init__(self, context):
        super().__init__(context, "拉帮结派！")

    def render_content(self):
        """
        Transcript：
        想象这样一个社区【播放动画 A】，
        圈子中的每个人互相之间都有个"支持度"【播放动画 B】。
        两个人互相的支持度为正数时，他们倾向于互相附和【播放动画 C】
        两个人互相的支持度为负数时，他们倾向于互相反对【播放动画 D】

        于此同时，他们之间的支持度也会变化：
        当他们对同一件事持相同态度时，他们的支持度会增加【播放动画 E】
        当他们对同一件事持不同态度时，他们的支持度会减少【播放动画 F】
        ====
        动画定义：
            动画 A: 依次画四对脸（共八张）
            动画 B: 在每一对脸之间画一条白线，上面显示数字 0.0
            动画 C:
                1. 对于左上方的一对脸，让数字上涨到 1.0
                2. 与此同时，白线变成蓝线
                3. 接下来，两张脸一同过渡成笑脸，然后过渡成哭脸，再过渡成笑脸
            动画 D:
                1. 对于左下方的一对脸，让数字下跌到 -1.0
                2. 与此同时，白线变成红线
                3. 接下来，两张脸，左边按照 笑脸 - 哭脸 - 笑脸 的顺序过渡，右边按照 哭脸 - 笑脸 - 哭脸 的顺序过渡
            动画 E:
                1. 对于右上方的一对脸，让他们同时过渡成笑脸
                2. 接下来，数字上涨到 1.0
                3. 与此同时，白线变成蓝线
            动画 F:
                1. 对于右下方的一对脸，让左边脸过渡成哭脸，右边脸过渡成笑脸
                2. 接下来，数字下跌到 -1.0
                3. 与此同时，白线变成红线
        ====
        布局：
            脸 - 脸 | 脸 - 脸
            脸 - 脸 | 脸 - 脸
        """
        # 设置脸的位置参数
        face_size = 1.0
        face_spacing_x = 3.0  # 水平间距
        face_spacing_y = 2.0  # 垂直间距
        pair_spacing = 1.5    # 每对脸之间的距离

        # 计算四对脸的位置（2x2布局）
        positions = [
            # 左上对：左脸、右脸
            [[-face_spacing_x - pair_spacing/2, face_spacing_y/2, 0],
             [-face_spacing_x + pair_spacing/2, face_spacing_y/2, 0]],
            # 右上对：左脸、右脸
            [[face_spacing_x - pair_spacing/2, face_spacing_y/2, 0],
             [face_spacing_x + pair_spacing/2, face_spacing_y/2, 0]],
            # 左下对：左脸、右脸
            [[-face_spacing_x - pair_spacing/2, -face_spacing_y/2, 0],
             [-face_spacing_x + pair_spacing/2, -face_spacing_y/2, 0]],
            # 右下对：左脸、右脸
            [[face_spacing_x - pair_spacing/2, -face_spacing_y/2, 0],
             [face_spacing_x + pair_spacing/2, -face_spacing_y/2, 0]]
        ]

        # 动画 A: 依次画四对脸（共八张）
        self.faces = []
        for pair_idx, pair_positions in enumerate(positions):
            pair_faces = []
            for face_pos in pair_positions:
                # 创建中性表情的脸
                face = HumanNeutralFace(size=face_size)
                face.move_to(face_pos)
                pair_faces.append(face)

            self.faces.append(pair_faces)
            # 依次显示每对脸
            self.context.play(*[FadeIn(face) for face in pair_faces], run_time=0.15)

        self.context.wait(0.3)
        self.context.next_slide()

        # 动画 B: 在每一对脸之间画一条白线，上面显示数字 0.0
        self.connection_lines = []
        self.support_numbers = []

        for pair_idx, pair_faces in enumerate(self.faces):
            left_face, right_face = pair_faces

            # 创建连接线
            line = Line(
                left_face.get_center() + RIGHT * face_size/2,
                right_face.get_center() + LEFT * face_size/2,
                color=WHITE,
                stroke_width=3
            )

            # 创建数字标签
            number = DecimalNumber(
                0.0,
                num_decimal_places=1,
                font_size=20,
                color=WHITE
            )
            number.move_to(line.get_center() + UP * 0.3)

            self.connection_lines.append(line)
            self.support_numbers.append(number)

            # 显示连接线和数字
            self.context.play(Create(line), FadeIn(number), run_time=0.15)

        self.context.wait(0.3)
        self.context.next_slide()

        # 动画 C: 左上方的一对脸 - 正支持度和同步表情
        self._animate_support_and_sync_emotion(
            0, 1.0, BLUE,
            [("happy", "happy"), ("sad", "sad"), ("happy", "happy")])

        # 动画 D: 左下方的一对脸 - 负支持度和反向表情
        self._animate_support_and_opposite_emotion(
            2, -1.0, RED,
            [("happy", "sad"), ("sad", "happy"), ("happy", "sad")])

        # 动画 E: 右上方的一对脸 - 先同步表情再正支持度
        self._animate_emotion_then_support(1, 1.0, BLUE, [("happy", "happy")])

        # 动画 F: 右下方的一对脸 - 先反向表情再负支持度
        self._animate_emotion_then_support(3, -1.0, RED, [("sad", "happy")])

    def _create_face_from_emotion(self, emotion_type, size=1.0):
        """根据表情类型创建脸"""
        if emotion_type == "happy":
            return HumanHappyFace(size=size)
        elif emotion_type == "sad":
            return HumanSadFace(size=size)
        else:  # neutral
            return HumanNeutralFace(size=size)

    def _animate_face_transition(self, old_face, new_emotion, position):
        """脸部表情过渡动画"""
        new_face = self._create_face_from_emotion(new_emotion, old_face.size)
        new_face.move_to(position)

        self.context.play(Transform(old_face, new_face), run_time=0.8)
        return new_face

    def _animate_support_and_sync_emotion(self, pair_idx, target_value, target_color, emotion_sequence):
        """支持度变化和同步表情变化"""
        # 1. 数字上涨/下跌 + 线条变色
        self.context.play(
            ChangeDecimalToValue(self.support_numbers[pair_idx], target_value),
            self.connection_lines[pair_idx].animate.set_color(target_color),
            run_time=1.0
        )
        self.context.wait(0.3)

        # 2. 表情同步变化序列
        for left_emotion, right_emotion in emotion_sequence:
            left_pos = self.faces[pair_idx][0].get_center()
            right_pos = self.faces[pair_idx][1].get_center()

            # 同时过渡两张脸
            self.context.play(
                Transform(
                    self.faces[pair_idx][0],
                    self._create_face_from_emotion(left_emotion, 1.0).move_to(left_pos)),
                Transform(
                    self.faces[pair_idx][1],
                    self._create_face_from_emotion(right_emotion, 1.0).move_to(right_pos)),
                run_time=0.5
            )
            self.context.wait(0.3)

        self.context.next_slide()

    def _animate_support_and_opposite_emotion(self, pair_idx, target_value, target_color, emotion_sequence):
        """支持度变化和反向表情变化"""
        # 1. 数字上涨/下跌 + 线条变色
        self.context.play(
            ChangeDecimalToValue(self.support_numbers[pair_idx], target_value),
            self.connection_lines[pair_idx].animate.set_color(target_color),
            run_time=1.0
        )
        self.context.wait(0.5)

        # 2. 表情反向变化序列
        for left_emotion, right_emotion in emotion_sequence:
            left_pos = self.faces[pair_idx][0].get_center()
            right_pos = self.faces[pair_idx][1].get_center()

            # 过渡两张脸到不同表情
            self.context.play(
                Transform(
                    self.faces[pair_idx][0],
                    self._create_face_from_emotion(left_emotion, 1.0).move_to(left_pos)),
                Transform(
                    self.faces[pair_idx][1],
                    self._create_face_from_emotion(right_emotion, 1.0).move_to(right_pos)),
                run_time=0.5
            )
            self.context.wait(0.3)

        self.context.next_slide()

    def _animate_emotion_then_support(self, pair_idx, target_value, target_color, emotion_sequence):
        """先表情变化，再支持度变化"""
        # 1. 表情变化
        for left_emotion, right_emotion in emotion_sequence:
            left_pos = self.faces[pair_idx][0].get_center()
            right_pos = self.faces[pair_idx][1].get_center()

            self.context.play(
                Transform(
                    self.faces[pair_idx][0],
                    self._create_face_from_emotion(left_emotion, 1.0).move_to(left_pos)),
                Transform(
                    self.faces[pair_idx][1],
                    self._create_face_from_emotion(right_emotion, 1.0).move_to(right_pos)),
                run_time=0.8
            )
            self.context.wait(0.5)

        # 2. 数字上涨/下跌 + 线条变色
        self.context.play(
            ChangeDecimalToValue(self.support_numbers[pair_idx], target_value),
            self.connection_lines[pair_idx].animate.set_color(target_color),
            run_time=1.0
        )
        self.context.wait(0.5)
        self.context.next_slide()
