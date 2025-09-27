from manim import *
from src.SlideFrames import *
from src.configs import *

class SelfOrgSlideComponent(ColumnLayoutSlideComponent):
    def __init__(self, context):
        super().__init__(context, "自组织学习！", num_columns=2)

    def render_columns(self):
        """
        Transcript：
        左侧：
            小标题：我们看到了什么？
            - 没有外部指令，系统通过**内部相互作用**自动调整
            - 每次辩论都是**一次学习**，支持度就是**权重更新**
            - 群体逐渐形成了**分布式记忆**，无需中央存储
        右侧：
            小标题：他们记住了什么？
            - 我们刻意淡化了辩论的 "话题" —— 因为这不重要。
            - 重要的是每次辩论时，我们给定的立场 —— 这是我们真正要编码的信息
            - 让他们构成方阵，发放黑白双色牌，他们便 "记住" 了一副黑白图像
            - **"编码"的魅力！**
        """
        left_pos = self.column_positions[0]
        right_pos = self.column_positions[1]

        # 左栏内容
        left_title = Text("我们看到了什么？", font_size=24, color="#FFD700", weight=BOLD)
        left_title.move_to(left_pos + UP * 2.5)
        self.context.play(Write(left_title), run_time=0.8)
        self.context.wait(0.3)

        left_points = [
            "没有外部指令，系统通过内部相互作用自动调整",
            "每次辩论都是一次学习，支持度就是权重更新",
            "群体逐渐形成了分布式记忆，无需中央存储"
        ]

        for i, point in enumerate(left_points):
            # 创建要点文本，使用合适的字体大小和颜色
            point_text = Text(f"• {point}", font_size=16, color=WHITE)
            point_text.move_to(left_pos + UP * (1.5 - i * 0.8))
            point_text.shift(LEFT * 0.2)  # 稍微左移对齐

            # 限制文本宽度，自动换行
            if point_text.width > 3.5:  # 如果文本太宽
                # 手动分割长文本
                if "内部相互作用" in point:
                    line1 = Text("• 没有外部指令，系统通过", font_size=16, color=WHITE)
                    line2 = Text("  内部相互作用自动调整", font_size=16, color=WHITE)
                    line1.move_to(left_pos + UP * (1.5 - i * 0.8) + LEFT * 0.2)
                    line2.move_to(left_pos + UP * (1.5 - i * 0.8 - 0.3) + LEFT * 0.2)
                    self.context.play(Write(line1), run_time=0.6)
                    self.context.play(Write(line2), run_time=0.6)
                elif "分布式记忆" in point:
                    line1 = Text("• 群体逐渐形成了分布式记忆，", font_size=16, color=WHITE)
                    line2 = Text("  无需中央存储", font_size=16, color=WHITE)
                    line1.move_to(left_pos + UP * (1.5 - i * 0.8) + LEFT * 0.2)
                    line2.move_to(left_pos + UP * (1.5 - i * 0.8 - 0.3) + LEFT * 0.2)
                    self.context.play(Write(line1), run_time=0.6)
                    self.context.play(Write(line2), run_time=0.6)
                else:
                    self.context.play(Write(point_text), run_time=0.8)
            else:
                self.context.play(Write(point_text), run_time=0.8)

            self.context.wait(0.3)

        self.context.wait(0.5)
        self.context.next_slide()

        # 右栏内容
        right_title = Text("他们记住了什么？", font_size=24, color="#FFD700", weight=BOLD)
        right_title.move_to(right_pos + UP * 2.5)
        self.context.play(Write(right_title), run_time=0.8)
        self.context.wait(0.3)

        right_points = [
            "我们刻意淡化了辩论的'话题'——因为这不重要",
            "重要的是每次辩论时，我们给定的立场",
            "这是我们真正要编码的信息",
            "让他们构成方阵，发放黑白双色牌，",
            "他们便'记住'了一副黑白图像"
        ]

        for i, point in enumerate(right_points):
            point_text = Text(f"• {point}", font_size=16, color=WHITE)
            point_text.move_to(right_pos + UP * (1.8 - i * 0.6))
            point_text.shift(LEFT * 0.2)

            # 特殊处理最后两个点，作为一个整体
            if "构成方阵" in point:
                line1 = Text("• 让他们构成方阵，发放黑白双色牌，", font_size=16, color=WHITE)
                line2 = Text("  他们便'记住'了一副黑白图像", font_size=16, color=WHITE)
                line1.move_to(right_pos + UP * (1.8 - i * 0.6) + LEFT * 0.2)
                line2.move_to(right_pos + UP * (1.8 - i * 0.6 - 0.3) + LEFT * 0.2)
                self.context.play(Write(line1), run_time=0.6)
                self.context.play(Write(line2), run_time=0.6)
                break  # 跳过下一个点，因为已经处理了
            elif "记住" not in point:  # 避免重复处理
                self.context.play(Write(point_text), run_time=0.8)
                self.context.wait(0.3)

        self.context.wait(0.8)

        # 最后的高亮文本
        highlight_text = Text("'编码'的魅力！", font_size=20, color="#FF6B9D", weight=BOLD)
        highlight_text.move_to(right_pos + DOWN * 1.5)
        self.context.play(Write(highlight_text), run_time=1.0)

        self.context.wait(2.0)
        self.context.next_slide()
