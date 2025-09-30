import numpy as np
from manim import *
from src.SlideFrames import *
from src.configs import *
from src.hopfield_tools import StandardBoltzmannMachine, BoltzmannMachineVisualizer
from SlideComponents.S02_RAMvsCAM.shared import get_patterns_for_compare, get_pattern_names, get_cue_indices, calculate_cued_accuracy

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
                3. 在下面加入五层每层六个（共18个）节点，同样使用 dot 表示，整体形成一个圆柱面
            动画 C: 根据 B 的配置初始化 Boltzmann Machine 权重，而后显示连接关系
            动画 D：将整个柱面+连接缩小并移动到 y 轴左侧，然后在右侧生成9个牌组（9 个一列）
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
        self.context.move_camera(phi=60 * DEGREES, theta=-30 * DEGREES, distance=8)

        # 移除2D连接线
        self.context.play(*[FadeOut(line) for line in self.connections_2d], run_time=0.5)

        # 创建Boltzmann机实例（优化参数以提高收敛性）
        self.boltzmann_machine = StandardBoltzmannMachine(
            context=self.context,
            n_visible=6,
            n_invisible=6  # 减少到6个节点，与训练样本数量相匹配
        )

        # 优化学习参数以提高收敛
        self.boltzmann_machine.learning_rate = 0.01  # 降低学习率提高稳定性
        self.boltzmann_machine.temperature = 0.0     # 温度=0，完全确定性

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
        self.visualizer.create_3d_network_structure(
            center_pos=[0, 0, 0], show_animation=True)

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

        recall_title = Text("深度回忆", font_size=18, color="#90EE90")
        recall_title.move_to([4.5, 3, 0])

        self.context.play(Write(training_title), Write(recall_title), run_time=0.5)

        self.context.wait(1.0)

    def _generate_training_patterns(self):
        """生成9个训练模式（使用共享配置）"""
        return get_patterns_for_compare()

    def _run_training_and_testing(self):
        """动画 E: 进行训练和测试"""
        # 训练Boltzmann机
        #training_text = Text("正在深度辩论...", font_size=18, color="#FFD700")
        #training_text.move_to([0, -3.5, 0])
        #self.context.play(Write(training_text), run_time=0.5)

        # 使用基于收敛的智能训练
        training_result = self.boltzmann_machine.train_with_convergence(
            training_data=self.training_patterns,
            max_epochs=500,       # 最大训练轮数
            batch_size=3,         # 小批次大小
            cd_steps=5,           # 对比散度步数
            patience=30,          # 早停耐心值
            min_improvement=1e-5, # 最小改善阈值
            verbose=False
        )

        # 更新连接显示（重新创建以反映训练后的权重）
        #self.context.play(FadeOut(training_text), run_time=0.3)

        # 显示智能收敛信息
        converged = training_result['converged']
        final_epoch = training_result['final_epoch']
        final_loss = training_result['final_loss']

        if converged:
            convergence_text = Text(
                f"智能收敛 (第{final_epoch}轮, 损失: {final_loss:.4f})",
                font_size=14, color="#90EE90"
            )
        else:
            convergence_text = Text(
                f"达到最大轮数 (第{final_epoch}轮, 损失: {final_loss:.4f})",
                font_size=14, color="#FFD700"
            )

        convergence_text.move_to([0, -3.5, 0])
        self.context.play(Write(convergence_text), run_time=0.5)
        self.context.wait(1.0)
        self.context.play(FadeOut(convergence_text), run_time=0.3)

        # 移除旧连接并创建新连接
        for line in self.visualizer.connection_lines:
            self.context.remove(line)

        self.visualizer.create_connections(
            show_animation=False, connection_threshold=0.1)

        # 进行测试：模仿SC07的可视化流程
        self.test_results = []
        self.recall_patterns = []  # 存储回忆结果用于显示

        for i, pattern in enumerate(self.training_patterns):
            accuracy, recall_pattern = self._test_pattern_with_visualization(pattern, i)
            self.test_results.append(accuracy)
            self.recall_patterns.append(recall_pattern)

        self.context.wait(1.0)

    def _test_pattern_with_visualization(self, original_pattern, pattern_idx):
        """使用可视化测试单个模式（模仿SC07风格）"""
        # 使用共享配置获取提示位组合（统一使用3位掩码）
        cue_indices = get_cue_indices(pattern_idx)

        # 创建提示模式
        cue_pattern = [None] * 6
        for idx in cue_indices:
            cue_pattern[idx] = original_pattern[idx]

        # 在右侧牌组中高亮提示位置
        pattern_y_offset = pattern_idx * 0.6
        pattern_pos = [3, 2.5 - pattern_y_offset, 0]

        # 创建高亮箭头（指向对应位置的小球上方）
        dot_spacing = 0.15
        arrows = []
        for idx in cue_indices:
            # 箭头位置：在小球上方
            arrow_start = [pattern_pos[0] + (idx - 2.5) * dot_spacing, pattern_pos[1] + 0.2, 0]
            arrow_end = [pattern_pos[0] + (idx - 2.5) * dot_spacing, pattern_pos[1] + 0.1, 0]

            arrow = Arrow(
                start=arrow_start,
                end=arrow_end,
                color=RED,
                stroke_width=4,
                tip_length=0.05,
                max_tip_length_to_length_ratio=0.5
            )
            arrows.append(arrow)

        self.context.play(*[Create(arrow) for arrow in arrows], run_time=0.3)

        # 将提示部分移动到网络（创建临时圆点进行移动）
        from src.hopfield_tools import HopfieldNetworkTools
        temp_dots = []
        for idx in cue_indices:
            value = original_pattern[idx]
            # 创建牌组样式的圆点
            temp_dot = Dot(
                radius=0.08,
                color=BLUE if value == 1 else RED,
                fill_opacity=0.8
            )

            start_pos = [pattern_pos[0] + (idx - 2.5) * dot_spacing, pattern_pos[1], 0]
            # 网络位置在左侧
            target_pos = [-3, 0, idx * 0.2]  # 简化的目标位置

            temp_dot.move_to(start_pos)
            temp_dots.append(temp_dot)

            self.context.play(temp_dot.animate.move_to(target_pos), run_time=0.3)
            self.context.play(FadeOut(temp_dot), run_time=0.1)

        # 设置Boltzmann机的可见层状态并固定提示位置
        visible_states = [0] * 6
        for idx in cue_indices:
            visible_states[idx] = original_pattern[idx]

        # 使用新方法：设置模式并固定提示节点
        self.visualizer.set_visible_pattern_with_fixed(
            pattern=visible_states,
            fixed_indices=list(cue_indices),
            show_animation=True
        )

        # 使用智能收敛推理（基于多数投票）
        inference_result = self.visualizer.run_inference_until_convergence(
            max_steps=150,              # 最大推理步数
            voting_window=30,           # 多数投票窗口（演示中使用更长窗口）
            early_stop_threshold=0.7,   # 早停阈值（演示中要求更高稳定性）
            check_interval=5,           # 检查间隔
            show_animation=True,        # 显示最终状态
            verbose=False
        )

        # 获取推理结果
        result_pattern = list(self.boltzmann_machine.visible_states)

        # 验证固定的提示部分没有改变
        for idx in cue_indices:
            if result_pattern[idx] != original_pattern[idx]:
                # 这不应该发生，如果发生了说明固定机制有问题
                print(f"WARNING: Fixed cue at index {idx} changed from {original_pattern[idx]} to {result_pattern[idx]}")
                # 强制恢复正确值
                result_pattern[idx] = original_pattern[idx]

        # 清除固定节点设置以便下次测试
        self.boltzmann_machine.clear_fixed_visible_indices()

        # 显示推理收敛信息（快速显示，不影响主流程）
        if inference_result['converged']:
            steps_info = Text(
                f"推理收敛({inference_result['steps_taken']}步)",
                font_size=8, color="#90EE90"
            )
        else:
            steps_info = Text(
                f"推理({inference_result['steps_taken']}步)",
                font_size=8, color="#FFD700"
            )

        steps_info.move_to([4.5, 2.5 - pattern_y_offset - 0.3, 0])
        self.context.play(Write(steps_info), run_time=0.2)

        # 将结果移动到右侧结果区域（与输入数据模式保持一致）
        result_pos = [4.5, 2.5 - pattern_y_offset, 0]

        # 创建与输入数据一样模式的小球
        from src.hopfield_tools import HopfieldNetworkTools
        temp_hopfield = HopfieldNetworkTools(self.context, 6, 0.08, 1.0)  # 用于创建牌组样式

        for i, value in enumerate(result_pattern):
            # 使用与输入数据相同的牌组样式
            result_card = temp_hopfield.create_card(value)

            pos = [result_pos[0] + (i - 2.5) * dot_spacing, result_pos[1], 0]
            result_card.move_to(pos)
            self.context.play(FadeIn(result_card), run_time=0.1)

        # 计算准确率（只计算未提示的部分）
        accuracy = calculate_cued_accuracy(original_pattern, result_pattern, cue_indices)

        # 显示准确率
        acc_text = Text(
            f"{accuracy:.1%}",
            font_size=10,
            color="#90EE90" if accuracy > 0.8 else "#FFD700" if accuracy > 0.5 else "#FF6B6B"
        )
        acc_text.move_to([result_pos[0] + 1.0, result_pos[1], 0])
        self.context.play(Write(acc_text), run_time=0.3)

        # 清除箭头
        self.context.play(*[FadeOut(arrow) for arrow in arrows], run_time=0.3)

        self.context.wait(0.3)
        return accuracy, result_pattern

        # _calculate_cued_accuracy 方法已移至 Shared.py，这里不再需要

    def _show_final_results(self):
        """动画 F: 摄像机归位并显示最终结果"""
        self.context.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES, distance=6)

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