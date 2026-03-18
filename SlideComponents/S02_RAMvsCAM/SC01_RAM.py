from manim import *
from src.SlideFrames import *
from src.configs import *
from src.mobjects.bio import Brain
from src.mobjects.icons import Photo
import random


class RAMSlideComponent(TitleSlideComponent):
    def __init__(self, context):
        super().__init__(context, "我们所知的「存储」")

    def render_content(self):
        space = self.get_available_space()
        self._content_mobs = []

        example_y = space["top"] - 1.3
        col_xs = [-4.0, 0.0, 4.0]

        self._anim_locker(col_xs[0], example_y)
        self.context.next_slide()

        self._anim_door(col_xs[1], example_y)
        self.context.next_slide()

        self._anim_folder(col_xs[2], example_y)
        self.context.next_slide()

        divider_y = space["center_y"] + 0.5
        divider = Line(
            LEFT * 5.5 + UP * divider_y,
            RIGHT * 5.5 + UP * divider_y,
            color="#4A90E2", stroke_width=2
        )
        self.context.play(Create(divider), run_time=0.5)
        self._content_mobs.append(divider)

        ram_text = Text(
            "Random Access Memory (RAM)",
            font=font_heading, font_size=28, color="#87CEEB"
        )
        ram_text.move_to([0, divider_y - 0.6, 0])
        self.context.play(Write(ram_text), run_time=0.8)
        self._content_mobs.append(ram_text)
        self.context.next_slide()

        self._anim_array(0, space["bottom"] + 1.8)

        self._anim_brain(space["center_y"])

    def _anim_locker(self, cx, cy):
        food = Circle(radius=0.25, color="#FF8C00", fill_opacity=0.8, stroke_width=0)
        food.move_to([cx - 0.6, cy, 0])

        locker = Rectangle(width=0.9, height=1.1, color=WHITE, stroke_width=2)
        locker.move_to([cx + 0.5, cy, 0])
        num_label = Text("52", font_size=22, color=WHITE)
        num_label.move_to(locker.get_top() + DOWN * 0.25)
        locker_grp = VGroup(locker, num_label)

        self.context.play(FadeIn(food), run_time=0.3)
        self.context.play(FadeIn(locker_grp), run_time=0.3)
        self.context.play(
            food.animate.move_to(locker.get_center()).set_opacity(0.3),
            run_time=0.5
        )
        self._content_mobs.extend([food, locker_grp])

    def _anim_door(self, cx, cy):
        door = Rectangle(
            width=0.8, height=1.3, color="#8B4513",
            fill_opacity=0.3, stroke_width=2
        )
        door.move_to([cx, cy - 0.05, 0])
        knob = Dot(point=door.get_right() + LEFT * 0.15, radius=0.05, color=YELLOW)
        door_grp = VGroup(door, knob)

        nameplate = Rectangle(
            width=0.9, height=0.25, color="#C0C0C0",
            fill_opacity=0.2, stroke_width=1
        )
        nameplate.next_to(door, UP, buff=0.08)
        plate_text = Text("8-504", font_size=14, color=WHITE)
        plate_text.move_to(nameplate.get_center())

        self.context.play(FadeIn(door_grp), run_time=0.3)
        self.context.play(FadeIn(nameplate), run_time=0.2)
        self.context.play(Write(plate_text), run_time=0.5)
        self._content_mobs.extend([door_grp, nameplate, plate_text])

    def _anim_folder(self, cx, cy):
        body = Rectangle(
            width=1.0, height=0.7, color="#DAA520",
            fill_opacity=0.25, stroke_width=2
        )
        tab = Rectangle(
            width=0.4, height=0.15, color="#DAA520",
            fill_opacity=0.25, stroke_width=2
        )
        tab.next_to(body, UP, buff=0, aligned_edge=LEFT)
        folder = VGroup(body, tab)
        folder.move_to([cx, cy - 0.15, 0])

        label = Text("我的文档", font=font_main_text, font_size=12, color="#F0F8FF")
        label.next_to(folder, DOWN, buff=0.1)

        self.context.play(FadeIn(folder), Write(label), run_time=0.4)

        photo = Photo()
        photo.scale(0.3)
        photo.move_to(body.get_center() + UP * 0.7)
        self.context.play(FadeIn(photo, shift=UP * 0.7), run_time=0.5)

        self._content_mobs.extend([folder, label, photo])

    def _anim_array(self, cx, cy):
        num_cells = 8
        cell_w = 0.65
        total_w = num_cells * cell_w
        start_x = cx - total_w / 2 + cell_w / 2

        cells = []
        for i in range(num_cells):
            cell = Rectangle(width=cell_w, height=0.5, color="#4A90E2", stroke_width=2)
            cell.move_to([start_x + i * cell_w, cy, 0])
            cells.append(cell)

        addrs = []
        for i in range(num_cells):
            addr = Text(f"0x{i:02X}", font=font_code, font_size=9, color=GREY)
            addr.next_to(cells[i], UP, buff=0.08)
            addrs.append(addr)

        array_struct = VGroup(*cells, *addrs)
        self.context.play(Create(array_struct), run_time=0.5)
        self._content_mobs.append(array_struct)
        self.context.next_slide()

        random.seed(42)
        values = []
        for i in range(num_cells):
            v = Text(
                str(random.randint(10, 99)),
                font=font_code, font_size=16, color="#F0F8FF"
            )
            v.move_to(cells[i].get_center())
            values.append(v)

        self.context.play(*[Write(v) for v in values], run_time=0.5)
        self._content_mobs.extend(values)
        self.context.next_slide()

        target_idx = 4
        ptr_arrow = Arrow(
            cells[0].get_bottom() + DOWN * 0.55,
            cells[0].get_bottom() + DOWN * 0.05,
            color="#FF6B6B", stroke_width=3, buff=0
        )
        ptr_text = Text("ptr", font=font_code, font_size=12, color="#FF6B6B")
        ptr_text.next_to(ptr_arrow, DOWN, buff=0.05)
        ptr = VGroup(ptr_arrow, ptr_text)

        self.context.play(FadeIn(ptr), run_time=0.3)

        dest_arrow = Arrow(
            cells[target_idx].get_bottom() + DOWN * 0.55,
            cells[target_idx].get_bottom() + DOWN * 0.05,
            color="#FF6B6B", stroke_width=3, buff=0
        )
        dest_text = Text("ptr", font=font_code, font_size=12, color="#FF6B6B")
        dest_text.next_to(dest_arrow, DOWN, buff=0.05)
        dest_ptr = VGroup(dest_arrow, dest_text)

        self.context.play(Transform(ptr, dest_ptr), run_time=0.6)
        self._content_mobs.append(ptr)
        self.context.next_slide()

        highlight = cells[target_idx].copy().set_stroke(color=YELLOW, width=4)
        self.context.play(Create(highlight), run_time=0.3)
        self._content_mobs.append(highlight)
        self.context.next_slide()

    def _anim_brain(self, center_y):
        self.context.play(
            *[m.animate.set_opacity(0.2) for m in self._content_mobs],
            run_time=0.5
        )

        brain = Brain()
        brain.scale(2.0)
        brain.move_to([0, center_y, 0])

        question = Text("?", font_size=80, color="#FFD700", weight=BOLD)
        question.move_to(brain.get_center())

        self.context.play(FadeIn(brain), run_time=0.5)
        self.context.play(Write(question), run_time=0.5)
        self.context.next_slide()
