from manim import *
from src.SlideFrames import *
from src.configs import *

class CAMSlideComponent(TitleSlideComponent):
    def __init__(self, context):
        super().__init__(context, "CAM - 另一种记忆系统")

    def render_content(self):
        """
        Transcript：
        看到 “白日依山尽”，你会自然想到“黄河入海流”【播放动画 A】
        看到一个熟悉的名字，你的脑海中会自然浮现出 ta 的模样，或与她经历的某个场景【播放动画 B】
        【画分割线】
        回忆的过程，相比于”搜寻“【播放动画 C】，更准确的是”唤起“【播放动画 D】
        你通过记忆的一部分，或者和记忆相关的某种线索，自然而然地补充出 “记忆” 的全貌【播放动画 E】
        这样的系统，我们称之为 Content Addressable Memory (CAM) 或 Associative Memory (AM)【写下 CAM 字样】

        ====

        布局：

        白日依山尽 |  xxx
                 |  (人物 | 场景
        黄河入海流 |  图标)
        ---------分割线---------
        动画 C       动画 D
        Content Addressable Memory (CAM)

        ====

        动画定义：
            动画 A：写下 “白日依山尽”，停顿，用另一个颜色写下 “黄河入海流”
            动画 B：写下 “XXX”，然后在下方画一个 “人物” 的图标，然后画一个 “场景” 的图标
            动画 C：
                1. 画一个放大镜图标，然后播放搜索动画
                2. 在放大镜下面打叉
            动画 D：
                1. 画一个灯泡图标（暗色）
                2. 将灯泡点亮
            动画 E:
                显示一个世界名画的中央部分，然后扩充到整个画作
        """
        pass
