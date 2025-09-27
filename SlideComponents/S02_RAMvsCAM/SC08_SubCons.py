import numpy as np
from manim import *
from src.SlideFrames import *
from src.configs import *
from src.hopfield_tools import StandardBoltzmannMachine, BoltzmannMachineVisualizer

class SubConsSlideComponent(TitleSlideComponent):
    def __init__(self, context):
        super().__init__(context, "……把观众拉下水")

    def render_content(self):
        """
        Transcript：
            【播放动画 A】
            那么解决这个问题的办法，其实是有的 —— 也就是把观众拉下水。
            【播放动画 B】
            什么意思呢，我们让观众席上的所有人和台上的人一同进行辩论。
            （也就是说，我们赋予它潜意识的部分）
            注意这里会让他们携带一个初始的关系 —— 以避免无效的"辩论"
            【播放动画 C】
            而后，我们重新尝试输入较多的数据，让观众席与台上的人一同改变彼此的关系模式
            【播放动画 D】
            接下来我们看看，在潜意识浪花翻腾起来的时候，他们记住了哪些吧
            【播放动画 E】
            【播放动画 F】
            "人脑只开发了 10%" —— 这是一句谬论。
        动画定义：
            动画 A: 在屏幕中央画一个六个节点环形作为"visible node"（此时不要画脸，而是使用对应颜色的 dot 表示）
            动画 B:
                1. 旋转摄像机，进入 3D 模式
                2. 将原来的六个节点的网络整体沿 Z 向上平移
                3. 在下面加入五层每层六个（共30个）节点，同样使用 dot 表示，整体形成一个圆柱面
            动画 C: 根据 B 的配置初始化 Boltzmann Machine 权重，而后显示连接关系
            动画 D：将整个柱面+连接缩小并移动到 y 轴左侧，然后在右侧生成24个牌组（6 个一列，沿 Z 轴叠 4 列）
            动画 E: 对比动画，将右侧的牌组依次移动到左侧的柱面中，而后显示对比结果
            动画 F: 摄像机归位，并且在中央写上最终准确率
        """
        # 动画 A: 创建二维的六个节点环形网络（使用中性圆点）
        self._create_2d_visible_network()
        self.context.next_slide()

        # 动画 B: 转换到3D并创建圆柱面结构
        self._transform_to_3d_cylinder()
        self.context.next_slide()

        # 动画 C: 初始化Boltzmann Machine并显示连接
        self._initialize_boltzmann_machine()
        self.context.next_slide()

        # 动画 D: 缩放网络并创建训练数据显示
        self._create_training_setup()
        self.context.next_slide()

        # 动画 E: 进行训练和对比测试
        self._run_training_and_testing()
        self.context.next_slide()

        # 动画 F: 摄像机归位并显示最终结果
        self._show_final_results()

    def _create_2d_visible_network(self):
        """动画 A: 创建2D的六个节点环形网络"""
        self.visible_nodes_2d = []
        self.visible_positions_2d = []

        # 创建六个中性节点
        radius = 1.5
        for i in range(6):
            angle = i * 2 * PI / 6 - PI/2
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            position = [x, y, 0]
            self.visible_positions_2d.append(position)

            # 创建中性圆点
            node = Dot(radius=0.1, color=GRAY, fill_opacity=0.8)
            node.move_to(position)
            self.visible_nodes_2d.append(node)

            self.context.play(FadeIn(node), run_time=0.2)

        # 创建完整连接（仅为视觉效果）
        self.connections_2d = []
        for i in range(6):
            for j in range(i + 1, 6):
                line = Line(self.visible_positions_2d[i], self.visible_positions_2d[j],
                           color=GRAY, stroke_width=1)
                self.connections_2d.append(line)
                self.context.play(Create(line), run_time=0.1)

        self.context.wait(0.5)

    def _transform_to_3d_cylinder(self):
        """动画 B: 转换到3D模式并创建圆柱面结构"""
        # 设置3D摄像机
        self.context.set_camera_orientation(phi=60 * DEGREES, theta=30 * DEGREES, distance=8)

        # 移除2D连接线
        self.context.play(*[FadeOut(line) for line in self.connections_2d], run_time=0.5)

        # 创建Boltzmann机实例
        self.boltzmann_machine = StandardBoltzmannMachine(
            context=self.context,
            n_visible=6,
            n_invisible=30  # 5层 × 6个节点
        )

        # 创建可视化工具
        self.visualizer = BoltzmannMachineVisualizer(
            context=self.context,
            boltzmann_machine=self.boltzmann_machine,
            visible_radius=1.5,
            hidden_radius=1.2,
            layer_spacing=0.8
        )

        # 移除2D节点
        self.context.play(*[FadeOut(node) for node in self.visible_nodes_2d], run_time=0.5)

        # 创建3D圆柱面结构
        self.visualizer.create_3d_network_structure(center_pos=[0, 0, 0], show_animation=True)

        self.context.wait(1.0)

    def _initialize_boltzmann_machine(self):
        """动画 C: 初始化Boltzmann Machine权重并显示连接"""
        # 使用小的随机权重初始化（已经在构造函数中完成）

        # 显示连接关系（只显示权重较大的连接以避免过于密集）
        self.context.play(
            Write(Text("初始化权重连接...", font_size=20, color="#FFD700").move_to([0, -3, 0])),
            run_time=0.5
        )

        self.visualizer.create_connections(show_animation=True, connection_threshold=0.05)

        self.context.wait(1.0)

    def _create_training_setup(self):
        """动画 D: 缩放网络并创建训练数据布局"""
        # 缩放并移动网络到左侧
        self.visualizer.scale_and_move_network(
            scale_factor=0.6,
            target_position=[-3, 0, 0],
            show_animation=True
        )

        # 生成24个训练模式（6个一列，4列深度）
        self.training_patterns = self._generate_training_patterns()

        # 在右侧创建3D模式显示
        self.pattern_objects = self.visualizer.create_pattern_display_3d(
            patterns=self.training_patterns,
            base_pos=[3, 2, 0],
            show_animation=True
        )

        # 添加标题
        training_title = Text("训练数据", font_size=18, color="#FFD700")
        training_title.move_to([3, 3, 0])
        self.context.play(Write(training_title), run_time=0.5)

        self.context.wait(1.0)

    def _generate_training_patterns(self):
        """生成24个训练模式"""
        patterns = [
            # 第一组模式（相似模式组1）
            [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 1],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 1, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0],

            # 第二组模式（相似模式组2）
            [0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1],
            [0, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 1],

            # 第三组模式（混合模式1）
            [1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 1],
            [0, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],

            # 第四组模式（混合模式2）
            [1, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 1, 1],
        ]
        return patterns

    def _run_training_and_testing(self):
        """动画 E: 进行训练和测试"""
        # 训练Boltzmann机
        training_text = Text("正在辩论...", font_size=16, color="#FFD700")
        training_text.move_to([0, -3.5, 0])
        self.context.play(Write(training_text), run_time=0.5)

        # 使用较少的训练轮数以加快演示
        loss_history = self.boltzmann_machine.train(
            training_data=self.training_patterns,
            epochs=50,
            batch_size=4,
            cd_steps=1,
            verbose=False
        )

        # 更新连接显示（重新创建以反映训练后的权重）
        self.context.play(FadeOut(training_text), run_time=0.3)

        # 移除旧连接并创建新连接
        for line in self.visualizer.connection_lines:
            self.context.remove(line)

        self.visualizer.create_connections(show_animation=False, connection_threshold=0.1)

        # 进行测试：选择几个模式进行不完整输入测试
        self.test_results = []
        test_patterns = self.training_patterns[:6]  # 测试前6个模式

        for i, pattern in enumerate(test_patterns):
            # 创建不完整的输入（只给前3个值）
            incomplete_pattern = pattern[:3] + [None, None, None]

            # 设置可见层状态
            self.visualizer.set_visible_pattern(incomplete_pattern[:3] + [0, 0, 0], show_animation=True)

            # 运行推理
            for _ in range(5):  # 5步推理
                self.visualizer.run_inference_step(show_animation=True)

            # 获取结果
            result_pattern = list(self.boltzmann_machine.visible_states)
            accuracy = self._calculate_test_accuracy(pattern, result_pattern)
            self.test_results.append(accuracy)

            self.context.wait(0.3)

        self.context.wait(1.0)

    def _calculate_test_accuracy(self, original, reconstructed):
        """计算测试准确率"""
        if len(original) != len(reconstructed):
            return 0.0

        # 只计算后3位的准确率（前3位是给定的）
        correct = sum(1 for i in range(3, 6) if original[i] == reconstructed[i])
        return correct / 3

    def _show_final_results(self):
        """动画 F: 摄像机归位并显示最终结果"""
        # 摄像机归位
        self.context.set_camera_orientation(phi=0, theta=0, distance=6)

        # 清除当前显示内容
        self.context.clear()

        # 计算平均准确率
        if self.test_results:
            avg_accuracy = np.mean(self.test_results)
        else:
            avg_accuracy = 0.75  # 默认值以防没有测试结果

        # 显示最终准确率
        final_text = Text(
            f"深度记忆准确率: {avg_accuracy:.1%}",
            font_size=36,
            color="#FF6B9D",
            weight=BOLD
        )
        final_text.move_to(ORIGIN)

        self.context.play(Write(final_text), run_time=2.0)

        # 显示解释文本
        explanation1 = Text(
            "通过引入\"潜意识\"（隐藏层）",
            font_size=20,
            color="#FFD700"
        )
        explanation1.move_to(UP * 1.5)

        explanation2 = Text(
            "网络获得了更强的模式记忆能力",
            font_size=20,
            color="#FFD700"
        )
        explanation2.move_to(DOWN * 1.5)

        self.context.play(
            Write(explanation1),
            Write(explanation2),
            run_time=1.5
        )

        # 最终的启发性文字
        final_wisdom = Text(
            "\"人脑只开发了10%\"——这是一句谬论",
            font_size=24,
            color="#90EE90"
        )
        final_wisdom.move_to(DOWN * 2.5)

        self.context.play(Write(final_wisdom), run_time=1.5)

        self.context.wait(2.0)