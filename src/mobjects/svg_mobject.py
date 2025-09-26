from manim import *

class CustomSVGMobject(VGroup):
    """
    SVG mobject 基类，提供通用的 SVG 加载和样式设置功能
    """
    def __init__(self, svg_path, size=2, color="#FF6B6B", fill_opacity=0.8,
                 stroke_width=2, stroke_color="#FFD700", **kwargs):
        super().__init__(**kwargs)
        self.svg_path = svg_path
        self.size = size
        self.color = color
        self.fill_opacity = fill_opacity
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color

        # 创建SVG对象
        self.create_svg()

    def create_svg(self):
        """
        从SVG文件创建图形并应用自定义样式
        """
        # 加载SVG文件
        svg_obj = SVGMobject(self.svg_path)

        # 设置填充颜色和透明度
        svg_obj.set_fill(color=self.color, opacity=self.fill_opacity)

        # 设置描边颜色和宽度
        svg_obj.set_stroke(color=self.stroke_color, width=self.stroke_width)

        # 计算缩放比例以适应指定大小
        svg_width = svg_obj.get_width()
        svg_height = svg_obj.get_height()

        # 计算缩放比例，保持宽高比
        scale_factor = min(self.size / svg_width, self.size / svg_height)
        svg_obj.scale(scale_factor)

        # 将SVG对象添加到VGroup中
        self.add(svg_obj)

        # 存储原始SVG对象以便后续操作
        self.svg_obj = svg_obj

    def set_color(self, color):
        """
        设置填充颜色
        """
        self.color = color
        self.svg_obj.set_fill(color=color, opacity=self.fill_opacity)

    def set_stroke_color(self, stroke_color):
        """
        设置描边颜色
        """
        self.stroke_color = stroke_color
        self.svg_obj.set_stroke(color=stroke_color, width=self.stroke_width)

    def set_fill_opacity(self, opacity):
        """
        设置填充透明度
        """
        self.fill_opacity = opacity
        self.svg_obj.set_fill(color=self.color, opacity=opacity)

    def set_stroke_width(self, width):
        """
        设置描边宽度
        """
        self.stroke_width = width
        self.svg_obj.set_stroke(color=self.stroke_color, width=width)

    def resize(self, new_size):
        """
        重新调整SVG大小
        """
        self.size = new_size
        # 重新创建SVG以应用新大小
        self.remove(self.svg_obj)
        self.create_svg()

    def get_svg_bounds(self):
        """
        获取SVG的边界框
        """
        return self.svg_obj.get_boundary_point(UP), self.svg_obj.get_boundary_point(DOWN)

    def center_svg(self):
        """
        将SVG居中
        """
        self.svg_obj.move_to(ORIGIN)
