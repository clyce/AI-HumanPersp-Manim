from manim import *
from src.SlideFrames import *
from src.configs import *

class RAMSlideComponent(TitleSlideComponent):
    def __init__(self, context):
        super().__init__(context, "我们所知的“存储”")

    def render_content(self):
        """
        Transcript：
        现实中你们一定曾经听过这样的话：
        - 你要的外卖在 52 号柜子【播放动画 A】
        - 你要找的人在 8 号楼 504 室【播放动画 B】
        - 你存储的照片 "我的文档" 文件夹下【播放动画 C】
        这是人类一直以来对于存储的认知 —— 按图索骥【Draw 分割线】。
        在计算机中，这样的存储系统被称为 Random Access Memory (RAM)。 【写下 Random Access Memory 字样】
        比如我们的内存【播放动画 D】，数据会被存储在一个个“地址”里【播放动画 E】，每当我们要访问一个数据，我们需要一个指针【播放动画 F】，指向这个数据所在的地址【播放动画 G】，来获得这个数据。
        这样的思维模式是如此根深蒂固，以致于我常常听人问起，人脑是如何对记忆进行“寻址”的【播放动画 H】。

        ====

        布局：
        动画 A | 动画 B | 动画 C
        ---------分割线---------
        Random Access Memory (RAM)
        动画 D -> E -> F -> G

        ====

        动画定义：
            动画 A：
                1. 画一个食物
                2. 画一个方框代表柜子，上面写 52
                3. 将食物平移进方框中，同时渐隐到 alpha = 0.3

            动画 B：
                1. 画一扇门
                2. 门上方画一个扁方框，代表门牌，上面用 Write 动画书写 8-504

            动画 C：
                1. 画一个文件夹，下书 “我的文档”
                2. 从文件夹中淡入一个照片图标，同时向上移动（模拟弹出）

            动画 DEFG：
                D. 画一个类似数组的结构
                E. 在每个格子里填充一个随机数
                F. 在第一个格子下方画一个箭头，然后平移到第五个格子
                G. 将第五个格子加上高亮边框

            动画 H：
                1. 将A-G的所有内容淡出到 alpha = 0.5 (此时 食物 alpha = 0.15)
                2. 在画面正中央画一个大脑
                3. 在大脑上画一个巨大的问号
        """
        pass

class BUTSlideComponent(SlideComponent):
    def render_content(self):
        f"""
        Transcript：
        【播放书写动画】但记忆的工作本质，并非基于逻辑与控制

        ====

        布局：
        中央大字
        """
        pass

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
        这样的系统，我们称之为 Content Access Memory (CAM)【写下 CAM 字样】

        ====

        布局：

        白日依山尽 |  xxx
                 |  (人物 | 场景
        黄河入海流 |  图标)
        ---------分割线---------
        动画 C       动画 D
        Content Access Memory (CAM)

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
                TODO: ?
        """
        pass

class GangingSlideComponent(SlideComponent):
    def __init__(self, context):
        super().__init__(context, "拉帮结派！")

    def render_content(self):
        """
        """
        pass

class RAMvsCAM(SlideWithCover):
    def construct(self):
        self.add_cover("存储与记忆：“事实”如何被唤起")

        self.slide_manager.add_component(RAMSlideComponent)
        self.slide_manager.add_component(BUTSlideComponent)
        self.slide_manager.add_component(CAMSlideComponent)
        self.slide_manager.add_component(GangingSlideComponent)

        self.slide_manager.add_component(ThanksSlideComponent)

        self.slide_manager.simple_execute_all()