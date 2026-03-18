from manim import *
from src.SlideFrames import *
from src.configs import *
from src.mobjects.icons import LightBulb


class CAMSlideComponent(TitleSlideComponent):
    def __init__(self, context):
        super().__init__(context, "CAM - 另一种记忆系统")

    def render_content(self):
        space = self.get_available_space()
        self._content_mobs = []

        left_x = -3.5
        right_x = 3.5
        top_y = space["top"] - 1.0

        self._anim_poetry(left_x, top_y)
        self.context.next_slide()

        self._anim_association(right_x, top_y)
        self.context.next_slide()

        divider_y = space["center_y"] + 0.3
        divider = Line(
            LEFT * 5.5 + UP * divider_y,
            RIGHT * 5.5 + UP * divider_y,
            color="#4A90E2", stroke_width=2
        )
        self.context.play(Create(divider), run_time=0.4)
        self._content_mobs.append(divider)

        bottom_y = divider_y - 1.2
        self._anim_search_vs_evoke(left_x, right_x, bottom_y)
        self.context.next_slide()

        self._anim_expansion(0, space["bottom"] + 1.0)
        self.context.next_slide()

        cam_text = Text(
            "Content Addressable Memory (CAM)",
            font=font_heading, font_size=26, color="#87CEEB"
        )
        cam_sub = Text(
            "又称 Associative Memory (AM)",
            font=font_main_text, font_size=16, color="#A0A0A0"
        )
        cam_group = VGroup(cam_text, cam_sub).arrange(DOWN, buff=0.2)
        cam_group.move_to([0, space["bottom"] + 0.5, 0])

        self.context.play(Write(cam_text), run_time=0.8)
        self.context.play(FadeIn(cam_sub), run_time=0.4)
        self.context.next_slide()

    def _anim_poetry(self, cx, cy):
        poem1 = Text("白日依山尽", font=font_heading, font_size=28, color="#F0F8FF")
        poem1.move_to([cx, cy, 0])

        self.context.play(Write(poem1), run_time=0.8)
        self.context.wait(0.5)

        poem2 = Text("黄河入海流", font=font_heading, font_size=28, color="#FFD700")
        poem2.move_to([cx, cy - 0.8, 0])

        self.context.play(Write(poem2), run_time=0.8)
        self._content_mobs.extend([poem1, poem2])

    def _anim_association(self, cx, cy):
        name = Text("一个熟悉的名字", font=font_main_text, font_size=22, color="#F0F8FF")
        name.move_to([cx, cy, 0])
        self.context.play(Write(name), run_time=0.5)

        person_circle = Circle(radius=0.3, color="#87CEEB", fill_opacity=0.3, stroke_width=2)
        person_label = Text("人物", font=font_main_text, font_size=14, color="#87CEEB")
        person_group = VGroup(person_circle, person_label).arrange(DOWN, buff=0.1)
        person_group.move_to([cx - 0.8, cy - 1.0, 0])

        scene_rect = RoundedRectangle(
            width=0.8, height=0.6, corner_radius=0.08,
            color="#FFD700", fill_opacity=0.2, stroke_width=2
        )
        scene_label = Text("场景", font=font_main_text, font_size=14, color="#FFD700")
        scene_group = VGroup(scene_rect, scene_label).arrange(DOWN, buff=0.1)
        scene_group.move_to([cx + 0.8, cy - 1.0, 0])

        self.context.play(
            FadeIn(person_group, shift=DOWN * 0.3),
            FadeIn(scene_group, shift=DOWN * 0.3),
            run_time=0.5
        )
        self._content_mobs.extend([name, person_group, scene_group])

    def _anim_search_vs_evoke(self, left_x, right_x, cy):
        mag_circle = Circle(radius=0.35, color="#888888", stroke_width=2)
        mag_handle = Line(
            mag_circle.get_corner(DR) + LEFT * 0.05 + UP * 0.05,
            mag_circle.get_corner(DR) + RIGHT * 0.2 + DOWN * 0.2,
            color="#888888", stroke_width=3
        )
        mag = VGroup(mag_circle, mag_handle)
        mag.move_to([left_x, cy, 0])

        search_label = Text("搜寻", font=font_main_text, font_size=20, color="#888888")
        search_label.next_to(mag, DOWN, buff=0.3)

        cross = VGroup(
            Line(LEFT * 0.15 + UP * 0.15, RIGHT * 0.15 + DOWN * 0.15, color="#FF4444", stroke_width=4),
            Line(LEFT * 0.15 + DOWN * 0.15, RIGHT * 0.15 + UP * 0.15, color="#FF4444", stroke_width=4),
        )
        cross.next_to(mag, RIGHT, buff=0.15)

        self.context.play(FadeIn(mag), Write(search_label), run_time=0.4)
        self.context.play(Create(cross), run_time=0.3)
        self._content_mobs.extend([mag, search_label, cross])

        bulb = LightBulb()
        bulb.scale(0.5)
        bulb.set_opacity(0.3)
        bulb.move_to([right_x, cy, 0])

        evoke_label = Text("唤起", font=font_main_text, font_size=20, color="#FFD700")
        evoke_label.next_to(bulb, DOWN, buff=0.3)

        self.context.play(FadeIn(bulb), Write(evoke_label), run_time=0.4)

        self.context.play(
            bulb.animate.set_opacity(1.0).set_color(YELLOW), run_time=0.5
        )
        self._content_mobs.extend([bulb, evoke_label])

    def _anim_expansion(self, cx, cy):
        """片段 → 全貌的概念示意：小矩形扩展为大矩形"""
        fragment = Rectangle(
            width=0.8, height=0.6, color="#87CEEB",
            fill_opacity=0.3, stroke_width=2
        )
        fragment.move_to([cx, cy, 0])
        frag_label = Text("片段", font=font_main_text, font_size=14, color="#87CEEB")
        frag_label.move_to(fragment.get_center())

        self.context.play(FadeIn(fragment), Write(frag_label), run_time=0.4)

        full = Rectangle(
            width=3.0, height=1.2, color="#FFD700",
            fill_opacity=0.15, stroke_width=2
        )
        full.move_to([cx, cy, 0])
        full_label = Text("记忆全貌", font=font_main_text, font_size=18, color="#FFD700")
        full_label.move_to(full.get_center())

        self.context.play(
            Transform(fragment, full),
            Transform(frag_label, full_label),
            run_time=0.8
        )
        self._content_mobs.extend([fragment, frag_label])
