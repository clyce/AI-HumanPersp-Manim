from manim import *
from manim_slides import Slide, ThreeDSlide
from configs import *

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

@runtime_checkable
class SlideContext(Protocol):
    """
    定义 SlideComponent 需要的slide上下文接口
    使用Protocol实现接口隔离原则，组件只依赖它真正需要的功能
    """
    def play(self, *animations, **kwargs):
        """播放动画"""
        ...

    def wait(self, duration: float = 1.0):
        """等待指定时间"""
        ...

    def next_slide(self, **kwargs):
        """切换到下一个slide"""
        ...

    def add(self, *mobjects):
        """添加对象到场景"""
        ...

    def remove(self, *mobjects):
        """从场景移除对象"""
        ...

    @property
    def canvas(self):
        """获取canvas对象（包含header_line, footer_line等）"""
        ...

    @property
    def mobjects_without_canvas(self):
        """获取不包含canvas的对象列表"""
        ...


class SlideComponent(ABC):
    """
    Slide组件基类
    采用组合模式，每个具体的slide功能作为独立组件
    """

    def __init__(self, context: SlideContext, auto_clear: bool = True):
        self.context = context
        self.auto_clear = auto_clear  # 是否自动清理前一页内容
        self._initialized = False

    def execute(self):
        """
        执行slide组件的主要逻辑
        包含完整的生命周期：clear -> setup -> render -> cleanup
        """
        try:
            # 自动清理前一页内容（可覆盖）
            if self.auto_clear:
                self.clear_previous_content()

            if not self._initialized:
                self.setup()
                self._initialized = True

            self.render()

        except Exception as e:
            self.on_error(e)
            raise
        finally:
            self.cleanup()

    def clear_previous_content(self, run_time: float = 0.5):
        """
        清理前一页内容的默认逻辑
        子类可以重写此方法来自定义清理行为
        """
        # 切换到下一个slide点
        self.context.next_slide()

        # 检查是否有需要淡出的对象
        mobjects_to_fade_out = []
        if hasattr(self.context, 'mobjects_without_canvas') and self.context.mobjects_without_canvas:
            mobjects_to_fade_out = self.context.mobjects_without_canvas

        if mobjects_to_fade_out:
            self.context.play(FadeOut(*mobjects_to_fade_out), run_time=run_time)

    def setup(self):
        """
        初始化设置，在render之前调用
        子类可以重写此方法进行预处理
        """
        pass

    @abstractmethod
    def render(self):
        """
        渲染slide内容的核心逻辑
        子类必须实现此方法
        """
        pass

    def cleanup(self):
        """
        清理工作，在render之后调用
        子类可以重写此方法进行后处理
        """
        pass

    def on_error(self, error: Exception):
        """
        错误处理
        子类可以重写此方法自定义错误处理逻辑
        """
        print(f"Error in {self.__class__.__name__}: {error}")

    # ========== 通用定位工具方法 ==========

    def get_available_space(self):
        """获取header和footer之间的可用空间"""
        if not hasattr(self.context, 'canvas') or not self.context.canvas:
            return {"top": 3.5, "bottom": -3.5, "height": 7.0, "center_y": 0}

        header_bottom = self.context.canvas.get("header_line")
        footer_top = self.context.canvas.get("footer_line")

        if header_bottom and footer_top:
            top = header_bottom.get_bottom()[1] - 0.3
            bottom = footer_top.get_top()[1] + 0.3
        else:
            top = 3.5
            bottom = -3.5

        return {
            "top": top,
            "bottom": bottom,
            "height": top - bottom,
            "center_y": (top + bottom) / 2
        }

    def get_title_position(self, buff=0.5):
        """获取标题的标准位置（header_line下方）"""
        if hasattr(self.context, 'canvas') and self.context.canvas.get("header_line"):
            return self.context.canvas["header_line"].get_bottom() + DOWN * buff
        return UP * 3.0

    def get_small_title_position(self, buff=0.3):
        """获取小标题的位置（更靠近header）"""
        if hasattr(self.context, 'canvas') and self.context.canvas.get("header_line"):
            return self.context.canvas["header_line"].get_bottom() + DOWN * buff
        return UP * 2.5

    def get_column_positions(self, num_columns=2, width_ratio=0.85):
        """
        获取多栏布局的位置
        Args:
            num_columns: 栏数（2或3）
            width_ratio: 使用屏幕宽度的比例
        Returns:
            dict: 包含各栏的中心位置、宽度和分割线位置
        """
        available_space = self.get_available_space()
        screen_width = config.frame_width * width_ratio

        if num_columns == 2:
            # 两栏布局：更宽的间距，更合理的列宽
            column_width = screen_width * 0.4  # 每栏占总宽度的40%
            spacing = screen_width * 0.25      # 间距占总宽度的25%
            positions = [
                LEFT * spacing + UP * available_space["center_y"],
                RIGHT * spacing + UP * available_space["center_y"]
            ]
            divider_x = 0  # 分割线在中央

        elif num_columns == 3:
            # 三栏布局：更均匀的分布
            column_width = screen_width * 0.28  # 每栏占总宽度的28%
            spacing = screen_width * 0.32       # 外侧间距
            positions = [
                LEFT * spacing + UP * available_space["center_y"],
                ORIGIN + UP * available_space["center_y"],
                RIGHT * spacing + UP * available_space["center_y"]
            ]
            # 分割线位置：栏之间的中点
            divider_positions = [
                -spacing * 0.5,  # 左分割线
                spacing * 0.5    # 右分割线
            ]

        else:
            raise ValueError("Only 2 or 3 columns are supported")

        return {
            "positions": positions,
            "column_width": column_width,
            "divider_positions": divider_positions if num_columns == 3 else [divider_x] if num_columns == 2 else []
        }

    def get_divider_positions(self, num_columns=3):
        """
        获取分割线位置
        注意：此方法已弃用，请使用 get_column_positions() 返回的 divider_positions
        """
        column_info = self.get_column_positions(num_columns)
        available_space = self.get_available_space()

        divider_lines = []
        for x_pos in column_info["divider_positions"]:
            divider_lines.extend([
                LEFT * x_pos + UP * available_space["top"],    # 分割线起点
                LEFT * x_pos + UP * available_space["bottom"]  # 分割线终点
            ])

        return divider_lines

    def center_horizontally(self, mobject):
        """将对象水平居中"""
        mobject.move_to([0, mobject.get_y(), 0])
        return mobject

    def position_below_title(self, mobject, title, buff=0.8):
        """将对象定位在标题下方"""
        mobject.next_to(title, DOWN, buff=buff)
        return self.center_horizontally(mobject)

    def interactive_list(self, list_items, positioning_function,
                         font_size=18, color="#F0F8FF", run_time=0.5,
                         animation="Write", bullet="circle", bullet_size=0.1,
                         bullet_color="#FFD700"):
        """
        创建交互式列表
        从 SlideWithCover 移动到 SlideComponent，作为组件的辅助方法
        """
        bullets = []
        if bullet == "circle":
            bullets = [Circle(radius=bullet_size, color=bullet_color) for _ in range(len(list_items))]
        elif bullet == "square":
            bullets = [Square(side_length=bullet_size, color=bullet_color) for _ in range(len(list_items))]

        list_items = [
            (item[0], Text(item[1], font=font_main_text, font_size=font_size, color=color), item[2])
            for item in list_items]
        positioning_function([item[1] for item in list_items])

        before_item_show = None
        after_item_show = None

        for id, item in enumerate(list_items):
            # 如果 item 是 tuple，解析tuple：
            (before_item_show, text_item, after_item_show) = item
            if before_item_show:
                before_item_show(text_item)
            # if bullet is not None and len(bullets) > 0:
            #     self.context.play(Create(bullets[id]), run_time=run_time)
            if animation == "Write":
                self.context.play(Write(text_item), run_time=run_time)
            elif animation == "Fade":
                self.context.play(FadeIn(text_item), run_time=run_time)
            if after_item_show:
                after_item_show(text_item)
            self.context.wait(0.3)
            self.context.next_slide()


# ========== 标准化布局基类 ==========

class TitleSlideComponent(SlideComponent):
    """
    带标题的slide组件基类
    自动处理标题显示 → 缩小移动 → 添加内容的标准模式
    """

    def __init__(self, context: SlideContext, title_text: str,
                 auto_clear: bool = True, title_font_size: int = 32,
                 small_title_font_size: int = 24):
        super().__init__(context, auto_clear)
        self.title_text = title_text
        self.title_font_size = title_font_size
        self.small_title_font_size = small_title_font_size
        self.title = None
        self.small_title = None

    def render(self):
        """标准的标题处理流程：显示大标题 → 缩小移动 → 添加内容"""
        self._show_main_title()
        self._move_title_to_top()
        self.render_content()

    def _show_main_title(self):
        """显示主标题"""
        self.title = Text(
            self.title_text,
            font=font_heading,
            font_size=self.title_font_size,
            color="#FFD700"
        )
        self.title.move_to(self.get_title_position())

        self.context.play(FadeIn(self.title))
        self.context.wait(0.5)
        self.context.next_slide()

    def _move_title_to_top(self):
        """将标题缩小并移动到上方"""
        self.small_title = Text(
            self.title_text,
            font=font_heading,
            font_size=self.small_title_font_size,
            color="#FFD700"
        )
        self.small_title.move_to(self.get_small_title_position())

        self.context.play(Transform(self.title, self.small_title), run_time=0.5)

    @abstractmethod
    def render_content(self):
        """渲染主要内容，子类必须实现"""
        pass


class ColumnLayoutSlideComponent(TitleSlideComponent):
    """
    多栏布局slide组件基类
    支持2栏或3栏布局，可选分割线
    """

    def __init__(self, context: SlideContext, title_text: str,
                 num_columns: int = 2, show_dividers: bool = False, **kwargs):
        super().__init__(context, title_text, **kwargs)
        self.num_columns = num_columns
        self.show_dividers = show_dividers
        self.column_positions = []
        self.dividers = []

    def render_content(self):
        """渲染多栏布局内容"""
        self._setup_columns()
        if self.show_dividers:
            self._create_dividers()
        self.render_columns()

    def _setup_columns(self):
        """设置栏位位置"""
        self.column_info = self.get_column_positions(self.num_columns)
        self.column_positions = self.column_info["positions"]

    def _create_dividers(self):
        """创建分割线"""
        available_space = self.get_available_space()
        self.dividers = []

        # 使用新的分割线位置计算
        for x_pos in self.column_info["divider_positions"]:
            divider = Line(
                [x_pos, available_space["top"], 0],
                [x_pos, available_space["bottom"], 0],
                color="#4A90E2", stroke_width=1
            )
            self.dividers.append(divider)

        for divider in self.dividers:
            self.context.play(Create(divider), run_time=0.2)

    @abstractmethod
    def render_columns(self):
        """渲染各栏内容，子类必须实现"""
        pass

    def render_column_content(self, column_index: int, content_creator):
        """
        渲染指定栏的内容
        Args:
            column_index: 栏索引（0开始）
            content_creator: 内容创建函数，接收栏位置作为参数
        """
        if 0 <= column_index < len(self.column_positions):
            content_creator(self.column_positions[column_index])


class ListSlideComponent(TitleSlideComponent):
    """
    列表展示slide组件基类
    标准化了interactive_list的使用模式
    """

    def __init__(self, context: SlideContext, title_text: str,
                 list_items: list, **kwargs):
        super().__init__(context, title_text, **kwargs)
        self.list_items = list_items

    def render_content(self):
        """渲染列表内容"""
        # 转换为interactive_list格式
        formatted_items = [(None, item, None) if isinstance(item, str) else item
                          for item in self.list_items]

        # 使用标准定位函数
        def position_items(items):
            content_group = VGroup(*items).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
            content_group = self.position_below_title(content_group, self.small_title)

        # 调用interactive_list
        self.interactive_list(
            formatted_items,
            position_items,
            **self.get_list_style()
        )

    def get_list_style(self):
        """获取列表样式，子类可以重写"""
        return {
            "font_size": 20,
            "color": "#F0F8FF",
            "run_time": 0.8,
            "animation": "Write",
            "bullet": "circle",
            "bullet_size": 0.06,
            "bullet_color": "#FFD700"
        }


class QuestionAnswerSlideComponent(ColumnLayoutSlideComponent):
    """
    问答布局slide组件
    左栏问题，右栏答案的标准布局
    """

    def __init__(self, context: SlideContext, title_text: str,
                 questions: list, answers: list, **kwargs):
        super().__init__(context, title_text, num_columns=2, **kwargs)
        self.questions = questions
        self.answers = answers

    def render_columns(self):
        """渲染问答内容"""
        # 左栏：问题列表
        self.render_column_content(0, self._render_questions)

        # 右栏：答案列表
        self.render_column_content(1, self._render_answers)

    def _render_questions(self, position):
        """渲染问题列表"""
        question_items = [(None, q, None) for q in self.questions]

        def position_questions(items):
            content_group = VGroup(*items).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
            content_group.move_to([position[0], position[1], 0])

        self.interactive_list(
            question_items,
            position_questions,
            font_size=18,
            color="#87CEEB",
            bullet_color="#FFD700"
        )

    def _render_answers(self, position):
        """渲染答案列表"""
        answer_items = [(None, a, None) for a in self.answers]

        def position_answers(items):
            content_group = VGroup(*items).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
            content_group.move_to([position[0], position[1], 0])

        self.interactive_list(
            answer_items,
            position_answers,
            font_size=18,
            color="#F0F8FF",
            bullet_color="#FF6B6B"
        )


class CoverSlideComponent(SlideComponent):
    """
    封面页slide组件
    负责显示标题页和初始化header/footer
    """

    def __init__(self, context: SlideContext, subtitle_text: str, author_text_lines=None):
        # 封面页不需要自动清理（因为是第一页）
        super().__init__(context, auto_clear=False)
        self.subtitle_text = subtitle_text
        self.author_text_lines = author_text_lines or [
            "Bilibili: @Clyce",
            "公众号：酒缸中的玻尔兹曼脑",
            "知乎：Clyce"
        ]
        self.title_text = "人的视角：（重新）理解 AI"

    def render(self):
        """渲染封面页"""
        self._create_title_page()
        self.context.next_slide()
        self._create_header_footer()

    def _create_title_page(self):
        """创建标题页"""
        # 创建标题页元素 - 使用更美观的配色
        self.title = Text(self.title_text, font=font_heading, font_size=48, color="#FFD700")  # 金色
        self.subtitle = Text(self.subtitle_text, font=font_heading, font_size=32, color="#87CEEB")  # 天蓝色
        self.author_lines = [
            Text(line, font=font_main_text, font_size=20, color="#F0F8FF") for line in self.author_text_lines]  # 爱丽丝蓝

        # 设置标题页布局
        self.title.to_edge(UP, buff=1.5)
        self.subtitle.next_to(self.title, DOWN, buff=0.8)

        # 创建作者信息组
        author_group = VGroup(*self.author_lines).arrange(DOWN, buff=0.3)
        author_group.to_edge(DOWN, buff=1.5)

        # 添加装饰性元素 - 使用渐变色彩
        self.title_line = Line(LEFT * 3, RIGHT * 3, color="#FF6B6B", stroke_width=3)
        self.title_line.next_to(self.subtitle, DOWN, buff=0.5)

        # 播放标题页动画 - 改进的动画效果
        # 检查是否有需要淡出的对象
        mobjects_to_fade_out = []
        if hasattr(self.context, 'mobjects_without_canvas') and self.context.mobjects_without_canvas:
            mobjects_to_fade_out = self.context.mobjects_without_canvas

        if mobjects_to_fade_out:
            self.context.play(
                FadeOut(*mobjects_to_fade_out),
                FadeIn(self.title, shift=LEFT),
                FadeIn(self.subtitle, shift=RIGHT),
                run_time=1
            )
        else:
            self.context.play(
                FadeIn(self.title, shift=LEFT),
                FadeIn(self.subtitle, shift=RIGHT),
                run_time=1
            )

        # 装饰线淡入
        self.context.play(FadeIn(self.title_line), run_time=0.3)

        # Authors逐行write效果
        for i, author_line in enumerate(self.author_lines):
            self.context.play(Write(author_line), run_time=0.5)
            if i < len(self.author_lines) - 1:
                self.context.wait(0.1)

    def _create_header_footer(self):
        """创建页眉和页脚，将标题页元素转换为页面布局"""
        # 缩小并移动标题到页眉 - 更小的字体和高度
        small_title = Text(self.title_text, font=font_heading, font_size=12, color="#FFD700")
        small_title.to_edge(UP, buff=0.2).to_edge(LEFT, buff=0.2)

        # 缩小并移动副标题到页眉右侧 - 更小的字体和高度
        small_subtitle = Text(self.subtitle_text, font=font_heading, font_size=12, color="#87CEEB")
        small_subtitle.to_edge(UP, buff=0.2).to_edge(RIGHT, buff=0.2)

        # 创建页眉线 - 在header下方，更细的线条，顶着屏幕两侧
        header_line = Line(LEFT * 6, RIGHT * 6, color="#4A90E2", stroke_width=1)
        # 计算header区域的下边界位置
        header_bottom = max(small_title.get_bottom()[1], small_subtitle.get_bottom()[1])
        header_line.move_to([0, header_bottom - 0.1, 0])
        # 让线条顶着屏幕两侧
        header_line.stretch_to_fit_width(config.frame_width)

        # 创建作者信息的页脚布局
        if self.author_text_lines:
            # 将作者信息等距分布在页脚 - 更小的字体
            footer_authors = []
            for i, line in enumerate(self.author_text_lines):
                author_text = Text(line, font=font_main_text, font_size=10, color="#F0F8FF")
                footer_authors.append(author_text)

            # 等距布局作者信息
            footer_group = VGroup(*footer_authors).arrange(RIGHT, buff=1.2)
            footer_group.to_edge(DOWN, buff=0.2)

        # 创建页脚线 - 在footer上方，更细的线条，顶着屏幕两侧
        footer_line = Line(LEFT * 6, RIGHT * 6, color="#4A90E2", stroke_width=1)
        footer_line.next_to(footer_group, UP, buff=0.1)
        # 让线条顶着屏幕两侧
        footer_line.stretch_to_fit_width(config.frame_width)

        # 播放转换动画
        self.context.play(
            FadeIn(header_line),
            FadeIn(footer_line),
            Transform(self.title, small_title),
            Transform(self.subtitle, small_subtitle),
            FadeOut(self.title_line),
            *[Transform(self.author_lines[i], footer_authors[i]) for i in range(len(self.author_text_lines))],
            run_time=0.5
        )

        # 更新canvas
        self.context.add_to_canvas(
            header_line=header_line,
            footer_line=footer_line,
            title=small_title,
            subtitle=small_subtitle,
            authors=footer_group if self.author_text_lines else None
        )


class SlideManager:
    """
    Slide组件管理器
    负责管理和协调各个slide组件的执行
    """

    def __init__(self, context: SlideContext):
        self.context = context
        self.components = []

    def add_component(self, component_class, *args, **kwargs):
        """添加slide组件"""
        component = component_class(self.context, *args, **kwargs)
        self.components.append(component)
        return component

    def execute_component(self, index: int):
        """执行指定索引的组件"""
        if 0 <= index < len(self.components):
            self.components[index].execute()
        else:
            raise IndexError(f"Component index {index} out of range")

    def execute_all(self):
        """按顺序执行所有组件"""
        for i, component in enumerate(self.components):
            try:
                component.execute()
            except Exception as e:
                print(f"Error executing component {i}: {e}")
                # 可以选择继续执行下一个组件或停止
                continue

    def simple_execute_all(self):
        """
        简化的执行所有组件方法
        现在每个组件都会自动处理清理逻辑，不需要手动调用 clear_and_next
        """
        self.execute_all()

class SlideWithCover(ThreeDSlide):
    """
    3D版本的SlideWithCover，继承自ThreeDSlide以支持3D功能
    实现SlideContext协议，可以作为SlideComponent的上下文
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._canvas = {}  # 存储canvas相关对象（header_line, footer_line等）
        self.slide_manager = SlideManager(self)  # 组件管理器

    def construct(self):
        pass

    def add_cover(self, subtitle_text: str, author_text_lines=None):
        """添加封面页组件的便捷方法"""
        return self.slide_manager.add_component(
            CoverSlideComponent, subtitle_text, author_text_lines)

    @property
    def canvas(self):
        """实现SlideContext协议中的canvas property"""
        return self._canvas

    def add_to_canvas(self, **kwargs):
        """
        添加对象到canvas字典
        用于存储header_line, footer_line, title, subtitle等UI元素
        """
        self._canvas.update(kwargs)

    # Multi-camera methods for SlideWithCover3D class

    def fix_ui_elements_in_frame(self):
        """
        将所有UI元素固定在相机视角中，使其在3D相机移动时保持不�?
        这是 SlideWithCover3D 的核�?multi-camera 功能
        """
        # 收集所有需要固定的UI元素
        ui_elements = []

        # 添加标题和副标题
        if hasattr(self, 'title') and self.title:
            ui_elements.append(self.title)
        if hasattr(self, 'subtitle') and self.subtitle:
            ui_elements.append(self.subtitle)

        # 添加作者信�?
        if hasattr(self, 'author_lines') and self.author_lines:
            ui_elements.extend(self.author_lines)

        # 添加canvas中的UI元素
        if hasattr(self, 'canvas') and self.canvas:
            for key, element in self.canvas.items():
                if element and key in ['header_line', 'footer_line', 'title', 'subtitle', 'authors']:
                    if isinstance(element, (VMobject, VGroup)):
                        ui_elements.append(element)

        # 固定所有UI元素在相机视角中
        if ui_elements:
            self.add_fixed_in_frame_mobjects(*ui_elements)

    def add_fixed_content_title(self, title_text, font_size=32, color="#FFD700"):
        """
        添加固定在相机视角中的内容标题（用于3D场景中）
        """
        title = Text(title_text, font=font_heading, font_size=font_size, color=color)
        if hasattr(self, 'canvas') and 'header_line' in self.canvas:
            title.next_to(self.canvas["header_line"], DOWN, buff=0.5)
        else:
            title.to_edge(UP, buff=1.0)

        # 固定标题在相机视角中
        self.add_fixed_in_frame_mobjects(title)

        return title

    def enable_3d_camera_with_fixed_ui(self, phi=75*DEGREES, theta=45*DEGREES):
        """
        启用3D相机并固定UI元素
        """
        # 设置3D相机方向
        self.set_camera_orientation(phi=phi, theta=theta)

        # 固定现有的UI元素
        self.fix_ui_elements_in_frame()

    def reset_to_2d_camera(self):
        """
        重置相机回到2D模式并清除固定的UI元素
        """
        # 重置相机到默认2D视角
        self.set_camera_orientation(phi=0, theta=-90*DEGREES)

        # 清除所有固定在相机视角中的对象
        if hasattr(self, 'camera') and hasattr(self.camera, 'fixed_in_frame_mobjects'):
            self.camera.fixed_in_frame_mobjects.clear()

        # 重新固定基本的UI元素（header和footer）
        self.fix_basic_ui_elements()

    def fix_basic_ui_elements(self):
        """
        仅固定基本的UI元素（header和footer相关）
        """
        basic_ui_elements = []

        # 添加canvas中的基本UI元素
        if hasattr(self, 'canvas') and self.canvas:
            for key, element in self.canvas.items():
                if element and key in ['header_line', 'footer_line', 'title', 'subtitle', 'authors']:
                    if isinstance(element, (VMobject, VGroup)):
                        basic_ui_elements.append(element)

        # 固定基本UI元素在相机视角中
        if basic_ui_elements:
            self.add_fixed_in_frame_mobjects(*basic_ui_elements)


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

        # 激励语句
        encouragement_text = Text(
            "鼓起勇气，去成为历史的车轮吧",
            font=font_main_text,
            font_size=16,
            color="#F0F8FF"
        )
        encouragement_text.next_to(contact_info, DOWN, buff=0.8)

        # 动画显示
        self.context.play(FadeIn(thanks_text), run_time=1.0)
        self.context.play(FadeIn(contact_info), run_time=1.0)
        self.context.play(FadeIn(encouragement_text), run_time=1.2)
        self.context.next_slide()
