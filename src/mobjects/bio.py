from manim import *
from .svg_mobject import CustomSVGMobject

class RunningGirl(CustomSVGMobject):
    """
    跑步女孩SVG mobject，继承自CustomSVGMobject基类
    """
    def __init__(self, **kwargs):
        # 调用父类构造函数，传入跑步女孩SVG路径
        super().__init__(
            svg_path="svgs/girl_running.svg",
            **kwargs
        )

class Brain(CustomSVGMobject):
    """
    大脑SVG mobject，继承自CustomSVGMobject基类
    """
    def __init__(self, **kwargs):
        # 调用父类构造函数，传入大脑SVG路径
        super().__init__(
            svg_path="svgs/brain.svg",
            **kwargs)

class Eye(CustomSVGMobject):
    def __init__(self, **kwargs):
        super().__init__(
            svg_path="svgs/eye.svg",
            **kwargs)
