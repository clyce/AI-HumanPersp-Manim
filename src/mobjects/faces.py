from manim import *

class FaceBase(VGroup):
    """
    表情基类，所有表情继承自此类
    """
    def __init__(self, size=2, stroke_width=3, stroke_color="#FFD700",
                 eye_color="#FFFFFF", fill_opacity=0.5, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.radius = size / 2
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self.eye_color = eye_color
        self.fill_opacity = fill_opacity
        self.create_face()


class HumanHappyFace(FaceBase):
    """
    人类开心表情 mobject
    """
    def __init__(self, size=2, stroke_width=3, stroke_color="#0080FF",
                 eye_color="#FFFFFF", fill_opacity=0.5, **kwargs):
        super().__init__(size, stroke_width, stroke_color, eye_color, fill_opacity, **kwargs)

    def create_face(self):
        radius = self.radius

        # 创建脸部轮廓（圆形）
        face_outline = Circle(
            radius=radius,
            color=self.stroke_color,
            stroke_width=self.stroke_width,
            fill_opacity=self.fill_opacity
        )

        # 创建眼睛（向下弯的弧线）
        eye_radius = radius * 0.15
        eye_y = radius * 0.3
        eye_x = radius * 0.4

        # 左眼 - 向下弯的弧线
        left_eye = Arc(
            radius=eye_radius,
            start_angle=0,      # 从左开始
            angle=PI,            # 半圆弧
            color=self.eye_color,
            stroke_width=self.stroke_width
        )
        left_eye.move_to([-eye_x, eye_y, 0])

        # 右眼 - 向下弯的弧线
        right_eye = Arc(
            radius=eye_radius,
            start_angle=0,      # 从左开始
            angle=PI,            # 半圆弧
            color=self.eye_color,
            stroke_width=self.stroke_width
        )
        right_eye.move_to([eye_x, eye_y, 0])

        # 创建笑脸弧线（向上弯的弧线）
        smile_radius = radius * 0.4
        smile_arc = Arc(
            radius=smile_radius,
            start_angle=PI,          # 从右开始
            angle=PI,               # 半圆弧，向上弯
            color=self.eye_color,
            stroke_width=self.stroke_width
        )
        smile_arc.move_to([0, -radius * 0.3, 0])

        # 组合所有元素
        self.add(face_outline, left_eye, right_eye, smile_arc)


class HumanSadFace(FaceBase):
    """
    人类悲伤表情 mobject
    """
    def __init__(self, size=2, stroke_width=3, stroke_color="#FF4040",
                 eye_color="#FFFFFF", fill_opacity=0.5, **kwargs):
        super().__init__(size, stroke_width, stroke_color, eye_color, fill_opacity, **kwargs)

    def create_face(self):
        radius = self.radius

        # 创建脸部轮廓（圆形）
        face_outline = Circle(
            radius=radius,
            color=self.stroke_color,
            stroke_width=self.stroke_width,
            fill_opacity=self.fill_opacity
        )

        # 创建眼睛（向上弯的弧线）
        eye_radius = radius * 0.15
        eye_y = radius * 0.3
        eye_x = radius * 0.4

        # 左眼 - 向上弯的弧线
        left_eye = Arc(
            radius=eye_radius,
            start_angle=PI,       # 从右开始
            angle=PI,            # 半圆弧，向上弯
            color=self.eye_color,
            stroke_width=self.stroke_width
        )
        left_eye.move_to([-eye_x, eye_y, 0])

        # 右眼 - 向上弯的弧线
        right_eye = Arc(
            radius=eye_radius,
            start_angle=PI,       # 从右开始
            angle=PI,            # 半圆弧，向上弯
            color=self.eye_color,
            stroke_width=self.stroke_width
        )
        right_eye.move_to([eye_x, eye_y, 0])

        # 创建悲伤弧线（向下弯的弧线，开心脸旋转180度）
        sad_radius = radius * 0.4
        sad_arc = Arc(
            radius=sad_radius,
            start_angle=0,         # 从左开始
            angle=PI,               # 半圆弧，向下弯
            color=self.eye_color,
            stroke_width=self.stroke_width
        )
        sad_arc.move_to([0, -radius * 0.3, 0])

        # 组合所有元素
        self.add(face_outline, left_eye, right_eye, sad_arc)


class BotHappyFace(FaceBase):
    """
    AI开心表情 mobject
    """
    def create_face(self):
        radius = self.radius

        # 创建斜角方形脸部轮廓
        # 使用Polygon创建带斜角的方形
        bevel = radius * 0.2
        face_outline = Polygon(
            [-radius + bevel, radius, 0],      # 左上角（有斜角）
            [radius - bevel, radius, 0],       # 右上角（有斜角）
            [radius, radius - bevel, 0],       # 右上斜角
            [radius, -radius + bevel, 0],      # 右下角（有斜角）
            [radius - bevel, -radius, 0],      # 右下角（有斜角）
            [-radius + bevel, -radius, 0],     # 左下角（有斜角）
            [-radius, -radius + bevel, 0],     # 左下斜角
            [-radius, radius - bevel, 0],      # 左上斜角
            color=self.stroke_color,
            stroke_width=self.stroke_width,
            fill_opacity=self.fill_opacity
        )

        # 创建方形眼睛
        eye_size = radius * 0.2
        left_eye = Square(side_length=eye_size, color=self.eye_color, fill_opacity=1)
        right_eye = Square(side_length=eye_size, color=self.eye_color, fill_opacity=1)

        # 眼睛位置
        eye_y = radius * 0.3
        eye_x = radius * 0.4
        left_eye.move_to([-eye_x, eye_y, 0])
        right_eye.move_to([eye_x, eye_y, 0])

        # 创建 \__/ 形状的嘴巴
        mouth_width = radius * 0.8
        mouth_height = radius * 0.3

        # 左斜线 \
        left_line = Line(
            [-mouth_width/2, -radius * 0.3, 0],
            [-mouth_width/4, -radius * 0.5, 0],
            color=self.eye_color,
            stroke_width=self.stroke_width
        )

        # 底线 __
        bottom_line = Line(
            [-mouth_width/4, -radius * 0.5, 0],
            [mouth_width/4, -radius * 0.5, 0],
            color=self.eye_color,
            stroke_width=self.stroke_width
        )

        # 右斜线 /
        right_line = Line(
            [mouth_width/4, -radius * 0.5, 0],
            [mouth_width/2, -radius * 0.3, 0],
            color=self.eye_color,
            stroke_width=self.stroke_width
        )

        # 创建天线
        antenna_height = radius * 0.4
        antenna_ball_radius = radius * 0.08

        # 左天线
        left_antenna_line = Line(
            [-radius * 0.3, radius, 0],
            [-radius * 0.3, radius + antenna_height, 0],
            color="#FF1493",
            stroke_width=self.stroke_width
        )
        left_antenna_ball = Circle(
            radius=antenna_ball_radius,
            color="#FF1493",
            fill_opacity=1
        ).move_to([-radius * 0.3, radius + antenna_height, 0])

        # 右天线
        right_antenna_line = Line(
            [radius * 0.3, radius, 0],
            [radius * 0.3, radius + antenna_height, 0],
            color="#FF1493",
            stroke_width=self.stroke_width
        )
        right_antenna_ball = Circle(
            radius=antenna_ball_radius,
            color="#FF1493",
            fill_opacity=1
        ).move_to([radius * 0.3, radius + antenna_height, 0])

        # 组合所有元素
        self.add(
            face_outline, left_eye, right_eye,
            left_line, bottom_line, right_line,
            left_antenna_line, left_antenna_ball,
            right_antenna_line, right_antenna_ball
        )


class BotSadFace(FaceBase):
    """
    AI悲伤表情 mobject
    """
    def __init__(self, size=2, stroke_width=3, stroke_color="#87CEEB",
                 eye_color="#FFFFFF", fill_opacity=0.5, **kwargs):
        super().__init__(size, stroke_width, stroke_color, eye_color, fill_opacity, **kwargs)

    def create_face(self):
        radius = self.radius

        # 创建斜角方形脸部轮廓
        bevel = radius * 0.2
        face_outline = Polygon(
            [-radius + bevel, radius, 0],      # 左上角（有斜角）
            [radius - bevel, radius, 0],       # 右上角（有斜角）
            [radius, radius - bevel, 0],       # 右上斜角
            [radius, -radius + bevel, 0],      # 右下角（有斜角）
            [radius - bevel, -radius, 0],      # 右下角（有斜角）
            [-radius + bevel, -radius, 0],     # 左下角（有斜角）
            [-radius, -radius + bevel, 0],     # 左下斜角
            [-radius, radius - bevel, 0],      # 左上斜角
            color=self.stroke_color,
            stroke_width=self.stroke_width,
            fill_opacity=self.fill_opacity
        )

        # 创建X形状的眼睛（表示悲伤）
        eye_size = radius * 0.15
        eye_y = radius * 0.3
        eye_x = radius * 0.4

        # 左眼的X
        left_eye_line1 = Line(
            [-eye_x - eye_size/2, eye_y + eye_size/2, 0],
            [-eye_x + eye_size/2, eye_y - eye_size/2, 0],
            color=self.eye_color,
            stroke_width=self.stroke_width
        )
        left_eye_line2 = Line(
            [-eye_x - eye_size/2, eye_y - eye_size/2, 0],
            [-eye_x + eye_size/2, eye_y + eye_size/2, 0],
            color=self.eye_color,
            stroke_width=self.stroke_width
        )

        # 右眼的X
        right_eye_line1 = Line(
            [eye_x - eye_size/2, eye_y + eye_size/2, 0],
            [eye_x + eye_size/2, eye_y - eye_size/2, 0],
            color=self.eye_color,
            stroke_width=self.stroke_width
        )
        right_eye_line2 = Line(
            [eye_x - eye_size/2, eye_y - eye_size/2, 0],
            [eye_x + eye_size/2, eye_y + eye_size/2, 0],
            color=self.eye_color,
            stroke_width=self.stroke_width
        )

        # 创建 /--\ 形状的悲伤嘴巴
        mouth_width = radius * 0.8

        # 左斜线 /（向上）
        left_line = Line(
            [-mouth_width/2, -radius * 0.5, 0],
            [-mouth_width/4, -radius * 0.3, 0],
            color=self.eye_color,
            stroke_width=self.stroke_width
        )

        # 顶线 --
        top_line = Line(
            [-mouth_width/4, -radius * 0.3, 0],
            [mouth_width/4, -radius * 0.3, 0],
            color=self.eye_color,
            stroke_width=self.stroke_width
        )

        # 右斜线 \（向下）
        right_line = Line(
            [mouth_width/4, -radius * 0.3, 0],
            [mouth_width/2, -radius * 0.5, 0],
            color=self.eye_color,
            stroke_width=self.stroke_width
        )

        # 创建下垂的天线
        antenna_height = radius * 0.3  # 比开心版本短
        antenna_ball_radius = radius * 0.08

        # 左天线（稍微向左倾斜）
        left_antenna_line = Line(
            [-radius * 0.3, radius, 0],
            [-radius * 0.4, radius + antenna_height, 0],  # 向左倾斜
            color="#8B0000",  # 深红色
            stroke_width=self.stroke_width
        )
        left_antenna_ball = Circle(
            radius=antenna_ball_radius,
            color="#8B0000",
            fill_opacity=1
        ).move_to([-radius * 0.4, radius + antenna_height, 0])

        # 右天线（稍微向右倾斜）
        right_antenna_line = Line(
            [radius * 0.3, radius, 0],
            [radius * 0.4, radius + antenna_height, 0],  # 向右倾斜
            color="#8B0000",
            stroke_width=self.stroke_width
        )
        right_antenna_ball = Circle(
            radius=antenna_ball_radius,
            color="#8B0000",
            fill_opacity=1
        ).move_to([radius * 0.4, radius + antenna_height, 0])

        # 组合所有元素
        self.add(
            face_outline,
            left_eye_line1, left_eye_line2,
            right_eye_line1, right_eye_line2,
            left_line, top_line, right_line,
            left_antenna_line, left_antenna_ball,
            right_antenna_line, right_antenna_ball
        )


class HumanNeutralFace(FaceBase):
    """
    人类中性表情 mobject
    """
    def __init__(self, size=2, stroke_width=3, stroke_color="#FFFFFF",
                 eye_color="#FFFFFF", fill_opacity=0.5, **kwargs):
        super().__init__(size, stroke_width, stroke_color, eye_color, fill_opacity, **kwargs)

    def create_face(self):
        radius = self.radius

        # 创建脸部轮廓（圆形）
        face_outline = Circle(
            radius=radius,
            color=self.stroke_color,
            stroke_width=self.stroke_width,
            fill_opacity=self.fill_opacity
        )

        # 创建眼睛（小圆点）
        eye_radius = radius * 0.08
        eye_y = radius * 0.3
        eye_x = radius * 0.4

        # 左眼 - 小圆点
        left_eye = Circle(
            radius=eye_radius,
            color=self.eye_color,
            fill_opacity=1
        )
        left_eye.move_to([-eye_x, eye_y, 0])

        # 右眼 - 小圆点
        right_eye = Circle(
            radius=eye_radius,
            color=self.eye_color,
            fill_opacity=1
        )
        right_eye.move_to([eye_x, eye_y, 0])

        # 创建中性嘴巴（一条直线）
        mouth_width = radius * 0.4
        mouth_line = Line(
            [-mouth_width/2, -radius * 0.4, 0],
            [mouth_width/2, -radius * 0.4, 0],
            color=self.eye_color,
            stroke_width=self.stroke_width
        )

        # 组合所有元素
        self.add(face_outline, left_eye, right_eye, mouth_line)


class BotNeutralFace(FaceBase):
    """
    AI中性表情 mobject
    """
    def create_face(self):
        radius = self.radius

        # 创建斜角方形脸部轮廓
        bevel = radius * 0.2
        face_outline = Polygon(
            [-radius + bevel, radius, 0],      # 左上角（有斜角）
            [radius - bevel, radius, 0],       # 右上角（有斜角）
            [radius, radius - bevel, 0],       # 右上斜角
            [radius, -radius + bevel, 0],      # 右下角（有斜角）
            [radius - bevel, -radius, 0],      # 右下角（有斜角）
            [-radius + bevel, -radius, 0],     # 左下角（有斜角）
            [-radius, -radius + bevel, 0],     # 左下斜角
            [-radius, radius - bevel, 0],      # 左上斜角
            color=self.stroke_color,
            stroke_width=self.stroke_width,
            fill_opacity=self.fill_opacity
        )

        # 创建方形眼睛（小一些的方形）
        eye_size = radius * 0.15
        left_eye = Square(side_length=eye_size, color=self.eye_color, fill_opacity=1)
        right_eye = Square(side_length=eye_size, color=self.eye_color, fill_opacity=1)

        # 眼睛位置
        eye_y = radius * 0.3
        eye_x = radius * 0.4
        left_eye.move_to([-eye_x, eye_y, 0])
        right_eye.move_to([eye_x, eye_y, 0])

        # 创建中性嘴巴（一条直线）
        mouth_width = radius * 0.6
        mouth_line = Line(
            [-mouth_width/2, -radius * 0.4, 0],
            [mouth_width/2, -radius * 0.4, 0],
            color=self.eye_color,
            stroke_width=self.stroke_width
        )

        # 创建天线（正常竖直）
        antenna_height = radius * 0.4
        antenna_ball_radius = radius * 0.08

        # 左天线
        left_antenna_line = Line(
            [-radius * 0.3, radius, 0],
            [-radius * 0.3, radius + antenna_height, 0],
            color="#9932CC",  # 紫色，表示中性
            stroke_width=self.stroke_width
        )
        left_antenna_ball = Circle(
            radius=antenna_ball_radius,
            color="#9932CC",
            fill_opacity=1
        ).move_to([-radius * 0.3, radius + antenna_height, 0])

        # 右天线
        right_antenna_line = Line(
            [radius * 0.3, radius, 0],
            [radius * 0.3, radius + antenna_height, 0],
            color="#9932CC",
            stroke_width=self.stroke_width
        )
        right_antenna_ball = Circle(
            radius=antenna_ball_radius,
            color="#9932CC",
            fill_opacity=1
        ).move_to([radius * 0.3, radius + antenna_height, 0])

        # 组合所有元素
        self.add(
            face_outline, left_eye, right_eye, mouth_line,
            left_antenna_line, left_antenna_ball,
            right_antenna_line, right_antenna_ball
        )