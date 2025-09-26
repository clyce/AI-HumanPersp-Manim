import numpy as np
from manim import *
from src.SlideFrames import *
from src.configs import *
from src.mobjects.faces import HumanHappyFace, HumanSadFace, HumanNeutralFace, BotHappyFace, BotSadFace, BotNeutralFace

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

class GangingUpSlideComponent(TitleSlideComponent):
    def __init__(self, context):
        super().__init__(context, "拉帮结派！")

    def render_content(self):
        """
        Transcript：
        想象这样一个社区【播放动画 A】，
        圈子中的每个人互相之间都有个"支持度"【播放动画 B】。
        两个人互相的支持度为正数时，他们倾向于互相附和【播放动画 C】
        两个人互相的支持度为负数时，他们倾向于互相反对【播放动画 D】

        于此同时，他们之间的支持度也会变化：
        当他们对同一件事持相同态度时，他们的支持度会增加【播放动画 E】
        当他们对同一件事持不同态度时，他们的支持度会减少【播放动画 F】
        ====
        动画定义：
            动画 A: 依次画四对脸（共八张）
            动画 B: 在每一对脸之间画一条白线，上面显示数字 0.0
            动画 C:
                1. 对于左上方的一对脸，让数字上涨到 1.0
                2. 与此同时，白线变成蓝线
                3. 接下来，两张脸一同过渡成笑脸，然后过渡成哭脸，再过渡成笑脸
            动画 D:
                1. 对于左下方的一对脸，让数字下跌到 -1.0
                2. 与此同时，白线变成红线
                3. 接下来，两张脸，左边按照 笑脸 - 哭脸 - 笑脸 的顺序过渡，右边按照 哭脸 - 笑脸 - 哭脸 的顺序过渡
            动画 E:
                1. 对于右上方的一对脸，让他们同时过渡成笑脸
                2. 接下来，数字上涨到 1.0
                3. 与此同时，白线变成蓝线
            动画 F:
                1. 对于右下方的一对脸，让左边脸过渡成哭脸，右边脸过渡成笑脸
                2. 接下来，数字下跌到 -1.0
                3. 与此同时，白线变成红线
        ====
        布局：
            脸 - 脸 | 脸 - 脸
            脸 - 脸 | 脸 - 脸
        """
        # 设置脸的位置参数
        face_size = 1.0
        face_spacing_x = 3.0  # 水平间距
        face_spacing_y = 2.0  # 垂直间距
        pair_spacing = 1.5    # 每对脸之间的距离

        # 计算四对脸的位置（2x2布局）
        positions = [
            # 左上对：左脸、右脸
            [[-face_spacing_x - pair_spacing/2, face_spacing_y/2, 0],
             [-face_spacing_x + pair_spacing/2, face_spacing_y/2, 0]],
            # 右上对：左脸、右脸
            [[face_spacing_x - pair_spacing/2, face_spacing_y/2, 0],
             [face_spacing_x + pair_spacing/2, face_spacing_y/2, 0]],
            # 左下对：左脸、右脸
            [[-face_spacing_x - pair_spacing/2, -face_spacing_y/2, 0],
             [-face_spacing_x + pair_spacing/2, -face_spacing_y/2, 0]],
            # 右下对：左脸、右脸
            [[face_spacing_x - pair_spacing/2, -face_spacing_y/2, 0],
             [face_spacing_x + pair_spacing/2, -face_spacing_y/2, 0]]
        ]

        # 动画 A: 依次画四对脸（共八张）
        self.faces = []
        for pair_idx, pair_positions in enumerate(positions):
            pair_faces = []
            for face_pos in pair_positions:
                # 创建中性表情的脸
                face = HumanNeutralFace(size=face_size)
                face.move_to(face_pos)
                pair_faces.append(face)

            self.faces.append(pair_faces)
            # 依次显示每对脸
            self.context.play(*[FadeIn(face) for face in pair_faces], run_time=0.5)
            self.context.wait(0.3)

        self.context.next_slide()

        # 动画 B: 在每一对脸之间画一条白线，上面显示数字 0.0
        self.connection_lines = []
        self.support_numbers = []

        for pair_idx, pair_faces in enumerate(self.faces):
            left_face, right_face = pair_faces

            # 创建连接线
            line = Line(
                left_face.get_center() + RIGHT * face_size/2,
                right_face.get_center() + LEFT * face_size/2,
                color=WHITE,
                stroke_width=3
            )

            # 创建数字标签
            number = DecimalNumber(
                0.0,
                num_decimal_places=1,
                font_size=20,
                color=WHITE
            )
            number.move_to(line.get_center() + UP * 0.3)

            self.connection_lines.append(line)
            self.support_numbers.append(number)

            # 显示连接线和数字
            self.context.play(Create(line), FadeIn(number), run_time=0.5)

        self.context.wait(0.5)
        self.context.next_slide()

        # 动画 C: 左上方的一对脸 - 正支持度和同步表情
        self._animate_support_and_sync_emotion(0, 1.0, BLUE,
                                                 [("happy", "happy"), ("sad", "sad"), ("happy", "happy")])

        # 动画 D: 左下方的一对脸 - 负支持度和反向表情
        self._animate_support_and_opposite_emotion(2, -1.0, RED,
                                                     [("happy", "sad"), ("sad", "happy"), ("happy", "sad")])

        # 动画 E: 右上方的一对脸 - 先同步表情再正支持度
        self._animate_emotion_then_support(1, 1.0, BLUE, [("happy", "happy")])

        # 动画 F: 右下方的一对脸 - 先反向表情再负支持度
        self._animate_emotion_then_support(3, -1.0, RED, [("sad", "happy")])

    def _create_face_from_emotion(self, emotion_type, size=1.0):
        """根据表情类型创建脸"""
        if emotion_type == "happy":
            return HumanHappyFace(size=size)
        elif emotion_type == "sad":
            return HumanSadFace(size=size)
        else:  # neutral
            return HumanNeutralFace(size=size)

    def _animate_face_transition(self, old_face, new_emotion, position):
        """脸部表情过渡动画"""
        new_face = self._create_face_from_emotion(new_emotion, old_face.size)
        new_face.move_to(position)

        self.context.play(Transform(old_face, new_face), run_time=0.8)
        return new_face

    def _animate_support_and_sync_emotion(self, pair_idx, target_value, target_color, emotion_sequence):
        """支持度变化和同步表情变化"""
        # 1. 数字上涨/下跌 + 线条变色
        self.context.play(
            ChangeDecimalToValue(self.support_numbers[pair_idx], target_value),
            self.connection_lines[pair_idx].animate.set_color(target_color),
            run_time=1.0
        )
        self.context.wait(0.5)

        # 2. 表情同步变化序列
        for left_emotion, right_emotion in emotion_sequence:
            left_pos = self.faces[pair_idx][0].get_center()
            right_pos = self.faces[pair_idx][1].get_center()

            # 同时过渡两张脸
            self.context.play(
                Transform(self.faces[pair_idx][0],
                         self._create_face_from_emotion(left_emotion, 1.0).move_to(left_pos)),
                Transform(self.faces[pair_idx][1],
                         self._create_face_from_emotion(right_emotion, 1.0).move_to(right_pos)),
                run_time=0.8
            )
            self.context.wait(0.5)

        self.context.next_slide()

    def _animate_support_and_opposite_emotion(self, pair_idx, target_value, target_color, emotion_sequence):
        """支持度变化和反向表情变化"""
        # 1. 数字上涨/下跌 + 线条变色
        self.context.play(
            ChangeDecimalToValue(self.support_numbers[pair_idx], target_value),
            self.connection_lines[pair_idx].animate.set_color(target_color),
            run_time=1.0
        )
        self.context.wait(0.5)

        # 2. 表情反向变化序列
        for left_emotion, right_emotion in emotion_sequence:
            left_pos = self.faces[pair_idx][0].get_center()
            right_pos = self.faces[pair_idx][1].get_center()

            # 过渡两张脸到不同表情
            self.context.play(
                Transform(self.faces[pair_idx][0],
                         self._create_face_from_emotion(left_emotion, 1.0).move_to(left_pos)),
                Transform(self.faces[pair_idx][1],
                         self._create_face_from_emotion(right_emotion, 1.0).move_to(right_pos)),
                run_time=0.8
            )
            self.context.wait(0.5)

        self.context.next_slide()

    def _animate_emotion_then_support(self, pair_idx, target_value, target_color, emotion_sequence):
        """先表情变化，再支持度变化"""
        # 1. 表情变化
        for left_emotion, right_emotion in emotion_sequence:
            left_pos = self.faces[pair_idx][0].get_center()
            right_pos = self.faces[pair_idx][1].get_center()

            self.context.play(
                Transform(self.faces[pair_idx][0],
                         self._create_face_from_emotion(left_emotion, 1.0).move_to(left_pos)),
                Transform(self.faces[pair_idx][1],
                         self._create_face_from_emotion(right_emotion, 1.0).move_to(right_pos)),
                run_time=0.8
            )
            self.context.wait(0.5)

        # 2. 数字上涨/下跌 + 线条变色
        self.context.play(
            ChangeDecimalToValue(self.support_numbers[pair_idx], target_value),
            self.connection_lines[pair_idx].animate.set_color(target_color),
            run_time=1.0
        )
        self.context.wait(0.5)
        self.context.next_slide()

class PreSetSlideComponent(ColumnLayoutSlideComponent):
    def __init__(self, context):
        super().__init__(context, "……预设立场", num_columns=2, show_dividers=True)

    def render_columns(self):
        """
        Transcript：
        在这样一个圈子里【播放动画 A】，我们做这样一件事：
        随便丢给他们一些什么话题（这不重要），
        然后给这个圈子中的每个人发一个支持/反对的"预设立场"牌【播放动画 B】。
        让他们以这些预设的立场开始辩论，然后改变互相之间的支持度【播放动画 C】。
        现在我们把这些预设牌放到一边【播放动画 D】，
        然后再给他们另一套预设牌，让他们继续辩论【播放动画 E】。
        我们让他们反复在这两套牌组下进行辩论【播放动画 F】，最终他们就会形成一种网络模式。

        现在我们拿出其中一个牌阵中的部分牌，让他们重新辩论一次。【播放动画 G】
        我们会发现，他们（不准确地）"回忆"起了当初的全体预设立场 【播放动画 H】。

        让我们再尝试另一个实验：【播放动画 I】
        用第二套牌组的中间两张牌作为提示，看看能否唤起对第二套牌组的回忆【播放动画 J】。

        ====

        布局：
            左栏：六张脸组成一个全连接环形 + 牌组
            右栏：权重矩阵显示

        ====

        动画定义：
            动画 A: 依次画六张脸，然后画连接线（注意不要画到圆心），组成一个全连接环形
            动画 B:
                1. 在牌组 1 生成 6 张红/蓝的牌子 (101011)
                2. 把这些牌子，依次快速移动到 6 张脸的位置（顺时针），然后淡出，同时接到红色牌子的变成哭脸，接到蓝色牌子的变成笑脸
            动画 C: 以线的红蓝变化表示，真实地运行一次 Hopfield Network 的学习过程
            动画 D: 从六张脸依次快速收回牌组 1 (脸部表情不变化)
            动画 E: 生成牌组 2 (011000)
            动画 F: 把 牌组 1 和 牌组 2 各复制一份，都依次移动到六张脸的位置（顺时针），淡出，然后真实地运行一次 Hopfield Network 的对于 Batch-size = 2 的学习过程
            动画 G: 用一个红框把牌组1的前三位（101）框起来，然后把这三张牌复制一份快速分发给前三张脸，让六张脸的面部表情根据连接关系自然变化（使用真实的 Hopfield Network 收敛过程进行模拟）
            动画 H: 从六张脸依次快速提取牌组，然后放到牌组 1 下方进行比对
            动画 I: 重置脸部表情为中性，用红框框选牌组2的中间两位（11），分发给对应脸，触发网络收敛
            动画 J: 从六张脸提取结果，与牌组 2 进行比对，显示回忆成功率和结果分析
        """

        # 设置参数
        self.num_faces = 6
        self.face_size = 0.6  # 减小脸的尺寸以适应左栏
        self.circle_radius = 1.8  # 减小圆的半径

        # 初始化Hopfield网络权重矩阵
        self.weights = np.zeros((self.num_faces, self.num_faces))

        # 跟踪每张脸的当前表情状态 (1=开心, 0=悲伤, None=中性)
        self.current_emotions = [None] * self.num_faces

        # 牌组定义
        self.pattern1 = [1, 0, 1, 0, 1, 1]  # 101011 (1代表蓝牌/支持，0代表红牌/反对)
        self.pattern2 = [0, 1, 1, 0, 0, 0]  # 011000

        # 获取栏位位置
        left_pos = self.column_positions[0]
        right_pos = self.column_positions[1]

        # 动画 A: 在左栏创建六张脸的环形网络
        self._create_face_circle(left_pos)
        self._create_connection_network()

        # 在右栏创建权重矩阵显示
        self._create_weight_matrix_display(right_pos)

        # 在左栏底部创建牌组显示区域
        self._setup_card_display_area(left_pos)

        # 动画 B: 生成牌组1并分发
        self._animate_card_distribution(self.pattern1, "牌组 1")

        # 动画 C: 运行Hopfield Network学习过程
        self._animate_hopfield_learning([self.pattern1])

        # 动画 D: 收回牌组1
        self._animate_card_collection("牌组 1")

        # 动画 E: 生成牌组2并分发
        self._animate_card_distribution(self.pattern2, "牌组 2")

        # 动画 F: 批量学习两个模式
        self._animate_hopfield_learning([self.pattern1, self.pattern2])

        # 动画 G: 部分提示和网络收敛（牌组1的前三位）
        self._animate_partial_cue_and_recall()

        # 动画 H: 提取结果并比对（牌组1）
        self._animate_result_comparison()

        # 动画 I: 对牌组2的中间两位进行部分提示和回忆
        self._animate_partial_cue_pattern2()

        # 动画 J: 提取结果并比对（牌组2）
        self._animate_result_comparison_pattern2()

    def _create_face_circle(self, left_pos):
        """创建环形排列的六张脸"""
        self.faces = []
        self.face_positions = []
        self.face_numbers = []  # 存储数字标号

        for i in range(self.num_faces):
            # 计算环形位置，相对于左栏中心
            angle = i * 2 * PI / self.num_faces - PI/2  # 从顶部开始
            x = left_pos[0] + self.circle_radius * np.cos(angle)
            y = left_pos[1] + self.circle_radius * np.sin(angle) + 0.5  # 稍微上移
            position = [x, y, 0]
            self.face_positions.append(position)

            # 创建中性脸
            face = HumanNeutralFace(size=self.face_size)
            face.move_to(position)
            self.faces.append(face)

            # 创建数字标号 (1-6，1是最上方)
            number_label = Text(str(i + 1), font_size=16, color="#FFD700")
            number_label.move_to(position + UP * (self.face_size/2 + 0.3))
            self.face_numbers.append(number_label)

            # 依次显示脸和数字
            self.context.play(FadeIn(face), FadeIn(number_label), run_time=0.3)
            self.context.wait(0.2)

        self.context.next_slide()

    def _create_connection_network(self):
        """创建全连接网络"""
        self.connection_lines = {}

        for i in range(self.num_faces):
            for j in range(i + 1, self.num_faces):
                # 创建连接线
                start_pos = self.face_positions[i]
                end_pos = self.face_positions[j]

                # 避免穿过圆心，计算更短的弧形路径
                line = Line(start_pos, end_pos, color="#404040", stroke_width=2)
                self.connection_lines[(i, j)] = line

                # 显示连接线
                self.context.play(Create(line), run_time=0.1)

        self.context.wait(0.5)
        self.context.next_slide()

    def _create_weight_matrix_display(self, right_pos):
        """在右栏创建权重矩阵显示"""
        # 权重矩阵标题
        matrix_title = Text("权重矩阵", font_size=20, color="#FFD700")
        matrix_title.move_to(right_pos + UP * 2.0)
        self.context.play(Write(matrix_title), run_time=0.5)

        # 创建6x6权重矩阵显示
        self.weight_matrix_display = []
        cell_size = 0.4

        for i in range(self.num_faces):
            row = []
            for j in range(self.num_faces):
                # 计算矩阵单元格位置
                x = right_pos[0] + (j - 2.5) * cell_size
                y = right_pos[1] + (2.5 - i) * cell_size

                # 创建单元格（方形边框）
                cell = Square(side_length=cell_size * 0.9, color=WHITE, stroke_width=1)
                cell.move_to([x, y, 0])

                # 创建权重值文本
                if i == j:
                    value_text = Text("0", font_size=10, color=WHITE)
                else:
                    value_text = DecimalNumber(0.0, num_decimal_places=1, font_size=10, color=WHITE)
                value_text.move_to([x, y, 0])

                cell_group = VGroup(cell, value_text)
                row.append(cell_group)

                # 逐个显示矩阵单元格
                self.context.play(FadeIn(cell_group), run_time=0.05)

            self.weight_matrix_display.append(row)

        self.context.wait(0.5)
        self.context.next_slide()

    def _setup_card_display_area(self, left_pos):
        """设置牌组显示区域"""
        # 获取可用空间
        available_space = self.get_available_space()

        # 牌组标签位置（在左栏底部）
        self.card_area_1 = [left_pos[0] - 1.0, available_space["bottom"] + 0.8, 0]
        self.card_area_2 = [left_pos[0] + 1.0, available_space["bottom"] + 0.8, 0]

    def _create_card(self, value, label=""):
        """创建牌子（红色=0/反对，蓝色=1/支持）"""
        if value == 1:
            card = Square(side_length=0.4, color=BLUE, fill_opacity=0.8)
            if label:
                text = Text(label, font_size=12, color=WHITE)
                text.move_to(card.get_center())
                card = VGroup(card, text)
        else:
            card = Square(side_length=0.4, color=RED, fill_opacity=0.8)
            if label:
                text = Text(label, font_size=12, color=WHITE)
                text.move_to(card.get_center())
                card = VGroup(card, text)
        return card

    def _animate_card_distribution(self, pattern, group_name):
        """动画：分发牌组"""
        # 显示牌组标签
        group_label = Text(group_name, font_size=16, color="#FFD700")
        base_pos = self.card_area_1 if "1" in group_name else self.card_area_2
        group_label.move_to(base_pos)
        group_label.shift(UP * 0.4)
        self.context.play(Write(group_label), run_time=0.5)

        # 创建牌组
        cards = []

        for i, value in enumerate(pattern):
            card = self._create_card(value, str(value))
            # 水平排列牌组
            pos = [base_pos[0] + (i - 2.5) * 0.3, base_pos[1], 0]
            card.move_to(pos)
            cards.append(card)

            self.context.play(FadeIn(card), run_time=0.2)

        self.context.wait(0.5)

        # 分发给脸
        for i, (card, value) in enumerate(zip(cards, pattern)):
            target_pos = self.face_positions[i]
            card_copy = card.copy()

            # 移动牌子到脸的位置
            self.context.play(card_copy.animate.move_to(target_pos + UP * 0.3), run_time=0.5)

            # 牌子淡出，脸变表情
            new_face = HumanHappyFace(size=self.face_size) if value == 1 else HumanSadFace(size=self.face_size)
            new_face.move_to(target_pos)

            # 更新表情状态跟踪
            self.current_emotions[i] = value

            self.context.play(
                FadeOut(card_copy),
                Transform(self.faces[i], new_face),
                run_time=0.5
            )

        self.context.wait(0.5)
        self.context.next_slide()

    def _animate_hopfield_learning(self, patterns):
        """动画：Hopfield网络学习过程"""
        # 更新权重矩阵
        for pattern in patterns:
            # 转换为bipolar (-1, 1)
            bipolar_pattern = [2*p - 1 for p in pattern]

            # Hebbian学习规则：W_ij += xi * xj
            for i in range(self.num_faces):
                for j in range(self.num_faces):
                    if i != j:
                        self.weights[i][j] += bipolar_pattern[i] * bipolar_pattern[j]

        # 可视化权重变化（连接线和矩阵同时更新）
        self._visualize_weight_changes()

        self.context.wait(1.0)
        self.context.next_slide()

    def _visualize_weight_changes(self):
        """可视化权重变化"""
        line_animations = []
        matrix_animations = []

        # 更新连接线
        for (i, j), line in self.connection_lines.items():
            weight = self.weights[i][j]

            # 根据权重调整线条颜色和粗细
            if weight > 0:
                color = interpolate_color(WHITE, BLUE, min(abs(weight) / 4, 1))
            elif weight < 0:
                color = interpolate_color(WHITE, RED, min(abs(weight) / 4, 1))
            else:
                color = WHITE

            stroke_width = max(1, min(abs(weight) + 1, 5))

            line_animations.append(line.animate.set_color(color).set_stroke_width(stroke_width))

        # 更新权重矩阵显示
        for i in range(self.num_faces):
            for j in range(self.num_faces):
                if i != j:  # 跳过对角线元素
                    weight = self.weights[i][j]
                    cell_group = self.weight_matrix_display[i][j]

                    # 更新数值
                    new_value = DecimalNumber(weight, num_decimal_places=1, font_size=10, color=WHITE)
                    new_value.move_to(cell_group[1].get_center())

                    # 更新背景颜色
                    if weight > 0:
                        bg_color = interpolate_color(WHITE, BLUE, min(abs(weight) / 4, 0.5))
                    elif weight < 0:
                        bg_color = interpolate_color(WHITE, RED, min(abs(weight) / 4, 0.5))
                    else:
                        bg_color = WHITE

                    matrix_animations.append(Transform(cell_group[1], new_value))
                    matrix_animations.append(cell_group[0].animate.set_fill(bg_color, opacity=0.3))

        # 同时播放所有动画
        all_animations = line_animations + matrix_animations
        if all_animations:
            self.context.play(*all_animations, run_time=1.5)

    def _animate_card_collection(self, group_name):
        """动画：收回牌组"""
        # 从脸收回牌子（脸的表情保持不变）
        base_pos = self.card_area_1 if "1" in group_name else self.card_area_2

        cards_to_fade = []  # 收集需要淡出的牌子

        for i in range(self.num_faces):
            # 使用跟踪的表情状态
            value = self.current_emotions[i] if self.current_emotions[i] is not None else 0

            card = self._create_card(value, str(value))
            card.move_to(self.face_positions[i] + UP * 0.3)

            # 水平排列收回的牌组
            target_pos = [base_pos[0] + (i - 2.5) * 0.3, base_pos[1], 0]

            self.context.play(
                FadeIn(card),
                card.animate.move_to(target_pos),
                run_time=0.3
            )

            cards_to_fade.append(card)

        self.context.wait(0.5)

        # 收回完成后，将所有牌子淡出
        self.context.play(*[FadeOut(card) for card in cards_to_fade], run_time=0.5)

        self.context.next_slide()

    def _animate_partial_cue_and_recall(self):
        """动画：部分提示和回忆"""
        # 创建红框框选前三张牌
        frame = Rectangle(
            width=1.0, height=0.4,
            color=RED, stroke_width=3, fill_opacity=0
        )
        frame.move_to([self.card_area_1[0] - 0.45, self.card_area_1[1], 0])

        self.context.play(Create(frame), run_time=0.5)
        self.context.wait(0.5)

        # 分发前三张牌
        partial_pattern = self.pattern1[:3]  # [1, 0, 1]

        for i, value in enumerate(partial_pattern):
            card = self._create_card(value, str(value))
            start_pos = [self.card_area_1[0] + (i - 2.5) * 0.3, self.card_area_1[1], 0]
            target_pos = self.face_positions[i] + UP * 0.3

            card.move_to(start_pos)
            self.context.play(card.animate.move_to(target_pos), run_time=0.3)

            # 设置对应脸的表情
            new_face = HumanHappyFace(size=self.face_size) if value == 1 else HumanSadFace(size=self.face_size)
            new_face.move_to(self.face_positions[i])

            # 更新表情状态跟踪
            self.current_emotions[i] = value

            self.context.play(
                FadeOut(card),
                Transform(self.faces[i], new_face),
                run_time=0.3
            )

        # 让后三张脸根据网络权重自然演化
        self._animate_network_convergence([1, 0, 1, None, None, None])

        self.context.next_slide()

    def _animate_network_convergence(self, initial_state):
        """动画：网络收敛过程"""
        current_state = initial_state.copy()

        for iteration in range(5):  # 最多5次迭代
            changed = False

            for i in range(self.num_faces):
                if current_state[i] is None:  # 只更新未固定的神经元
                    # 计算net input
                    net_input = sum(self.weights[i][j] * (2 * current_state[j] - 1)
                                  for j in range(self.num_faces)
                                  if current_state[j] is not None)

                    # 激活函数
                    new_state = 1 if net_input > 0 else 0

                    if current_state[i] != new_state:
                        current_state[i] = new_state
                        changed = True

                        # 更新脸的表情
                        new_face = HumanHappyFace(size=self.face_size) if new_state == 1 else HumanSadFace(size=self.face_size)
                        new_face.move_to(self.face_positions[i])

                        # 更新表情状态跟踪
                        self.current_emotions[i] = new_state

                        self.context.play(Transform(self.faces[i], new_face), run_time=0.5)
                        self.context.wait(0.3)

            if not changed:
                break

        self.context.wait(1.0)

    def _animate_result_comparison(self):
        """动画：结果比对"""
        # 从脸提取当前状态
        extracted_pattern = []
        for i in range(self.num_faces):
            # 使用跟踪的表情状态
            value = self.current_emotions[i] if self.current_emotions[i] is not None else 0
            extracted_pattern.append(value)

            # 创建提取的牌子
            card = self._create_card(value, str(value))
            card.move_to(self.face_positions[i] + UP * 0.3)

            # 移动到比对区域（在牌组1下方）
            target_pos = [self.card_area_1[0] + (i - 2.5) * 0.3, self.card_area_1[1] - 0.6, 0]

            self.context.play(
                FadeIn(card),
                card.animate.move_to(target_pos),
                run_time=0.3
            )

        # 显示比对标签
        comparison_label = Text("回忆结果:", font_size=14, color="#FFD700")
        comparison_label.move_to([self.card_area_1[0], self.card_area_1[1] - 1.0, 0])
        self.context.play(Write(comparison_label), run_time=0.5)

        # 计算准确率
        accuracy = sum(1 for i in range(len(self.pattern1))
                      if extracted_pattern[i] == self.pattern1[i]) / len(self.pattern1)

        accuracy_text = Text(f"准确率: {accuracy:.1%}", font_size=14, color="#90EE90")
        accuracy_text.move_to([self.card_area_1[0], self.card_area_1[1] - 1.3, 0])
        self.context.play(Write(accuracy_text), run_time=0.5)

        self.context.wait(2.0)
        self.context.next_slide()

    def _animate_partial_cue_pattern2(self):
        """动画：对牌组2中间两位进行部分提示和回忆"""
        # 首先清除所有脸的表情，重置为中性
        for i in range(self.num_faces):
            neutral_face = HumanNeutralFace(size=self.face_size)
            neutral_face.move_to(self.face_positions[i])
            self.context.play(Transform(self.faces[i], neutral_face), run_time=0.3)
            self.current_emotions[i] = None

        self.context.wait(0.5)

        # 创建红框框选牌组2的中间两张牌（索引2-3）
        frame = Rectangle(
            width=0.7, height=0.4,
            color=RED, stroke_width=3, fill_opacity=0
        )
        frame.move_to([self.card_area_2[0] - 0.15, self.card_area_2[1], 0])  # 框选中间两张牌

        self.context.play(Create(frame), run_time=0.5)
        self.context.wait(0.5)

        # 添加提示文本
        cue_text = Text("牌组2中间两位提示", font_size=14, color="#FFD700")
        cue_text.move_to([self.card_area_2[0], self.card_area_2[1] + 0.6, 0])
        self.context.play(Write(cue_text), run_time=0.5)

        # 分发牌组2的中间两张牌（索引2-3）
        partial_pattern_indices = [2, 3]  # 索引2和3
        partial_pattern_values = [self.pattern2[i] for i in partial_pattern_indices]  # [1, 1]

        for idx, face_idx in enumerate(partial_pattern_indices):
            value = partial_pattern_values[idx]
            card = self._create_card(value, str(value))
            start_pos = [self.card_area_2[0] + (idx - 0.5) * 0.3, self.card_area_2[1], 0]
            target_pos = self.face_positions[face_idx] + UP * 0.3

            card.move_to(start_pos)
            self.context.play(card.animate.move_to(target_pos), run_time=0.3)

            # 设置对应脸的表情
            new_face = HumanHappyFace(size=self.face_size) if value == 1 else HumanSadFace(size=self.face_size)
            new_face.move_to(self.face_positions[face_idx])

            # 更新表情状态跟踪
            self.current_emotions[face_idx] = value

            self.context.play(
                FadeOut(card),
                Transform(self.faces[face_idx], new_face),
                run_time=0.3
            )

        # 构建初始状态数组：只有索引2和3有值，其他为None
        initial_state = [None] * self.num_faces
        for idx, face_idx in enumerate(partial_pattern_indices):
            initial_state[face_idx] = partial_pattern_values[idx]

        # 让其他脸根据网络权重自然演化
        self._animate_network_convergence(initial_state)

        self.context.next_slide()

    def _animate_result_comparison_pattern2(self):
        """动画：牌组2的结果比对"""
        # 从脸提取当前状态
        extracted_pattern = []
        for i in range(self.num_faces):
            # 使用跟踪的表情状态
            value = self.current_emotions[i] if self.current_emotions[i] is not None else 0
            extracted_pattern.append(value)

            # 创建提取的牌子
            card = self._create_card(value, str(value))
            card.move_to(self.face_positions[i] + UP * 0.3)

            # 移动到比对区域（在牌组2下方）
            target_pos = [self.card_area_2[0] + (i - 2.5) * 0.3, self.card_area_2[1] - 0.6, 0]

            self.context.play(
                FadeIn(card),
                card.animate.move_to(target_pos),
                run_time=0.3
            )

        # 显示比对标签
        comparison_label = Text("回忆结果:", font_size=14, color="#FFD700")
        comparison_label.move_to([self.card_area_2[0], self.card_area_2[1] - 1.0, 0])
        self.context.play(Write(comparison_label), run_time=0.5)

        # 计算准确率（与牌组2比较）
        accuracy = sum(1 for i in range(len(self.pattern2))
                      if extracted_pattern[i] == self.pattern2[i]) / len(self.pattern2)

        accuracy_text = Text(f"准确率: {accuracy:.1%}", font_size=14, color="#90EE90")
        accuracy_text.move_to([self.card_area_2[0], self.card_area_2[1] - 1.3, 0])
        self.context.play(Write(accuracy_text), run_time=0.5)

        # 添加对比说明
        if accuracy > 0.8:
            result_text = Text("成功回忆牌组2！", font_size=12, color="#90EE90")
        elif accuracy > 0.5:
            result_text = Text("部分回忆成功", font_size=12, color="#FFD700")
        else:
            result_text = Text("可能回忆到牌组1", font_size=12, color="#FF6B6B")

        result_text.move_to([self.card_area_2[0], self.card_area_2[1] - 1.6, 0])
        self.context.play(Write(result_text), run_time=0.5)

        self.context.wait(2.0)
        self.context.next_slide()

class RAMvsCAM(SlideWithCover):
    def construct(self):
        self.add_cover("存储与记忆：“事实”如何被唤起")

        #self.slide_manager.add_component(RAMSlideComponent)
        #self.slide_manager.add_component(BUTSlideComponent)
        #self.slide_manager.add_component(CAMSlideComponent)
        self.slide_manager.add_component(GangingUpSlideComponent)
        self.slide_manager.add_component(PreSetSlideComponent)

        self.slide_manager.add_component(ThanksSlideComponent)

        self.slide_manager.simple_execute_all()