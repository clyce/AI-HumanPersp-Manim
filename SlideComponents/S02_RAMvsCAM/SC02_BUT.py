from manim import *
from src.SlideFrames import *
from src.configs import *


class BUTSlideComponent(SlideComponent):
    def render(self):
        space = self.get_available_space()

        line1 = Text("但记忆的工作过程", font=font_heading, font_size=36, color="#FF6B6B")
        line2 = Text("并非基于逻辑与控制", font=font_heading, font_size=36, color="#FF6B6B")

        group = VGroup(line1, line2).arrange(DOWN, buff=0.5)
        group.move_to([0, space["center_y"], 0])

        self.context.play(Write(line1), run_time=1.0)
        self.context.wait(0.3)
        self.context.play(Write(line2), run_time=1.0)
        self.context.next_slide()
