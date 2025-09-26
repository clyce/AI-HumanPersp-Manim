from manim import *
from .svg_mobject import CustomSVGMobject

class Binary(CustomSVGMobject):
    def __init__(self, **kwargs):
        super().__init__(
            svg_path="svgs/binary.svg",
            fill_opacity=0.0,
            **kwargs)

class Sound(CustomSVGMobject):
    def __init__(self, **kwargs):
        super().__init__(
            svg_path="svgs/sound.svg",
            fill_opacity=0.0,
            **kwargs)

class Photo(CustomSVGMobject):
    def __init__(self, **kwargs):
        super().__init__(
            svg_path="svgs/photo.svg",
            fill_opacity=0.0,
            **kwargs)

class LightBulb(CustomSVGMobject):
    def __init__(self, **kwargs):
        super().__init__(
            svg_path="svgs/light_bulb.svg",
            fill_opacity=0.0,
            **kwargs)