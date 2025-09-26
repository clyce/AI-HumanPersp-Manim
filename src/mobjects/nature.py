from manim import *
from .svg_mobject import CustomSVGMobject

class Sun(VGroup):
    def __init__(self, **kwargs):
        pass

class Tree(VGroup):
    def __init__(self, **kwargs):
        pass

class Atom(CustomSVGMobject):
    def __init__(self, **kwargs):
        super().__init__(
            svg_path="svgs/atom.svg",
            **kwargs)

class Cloud(CustomSVGMobject):
    def __init__(self, **kwargs):
        super().__init__(
            svg_path="svgs/cloud.svg",
            **kwargs)