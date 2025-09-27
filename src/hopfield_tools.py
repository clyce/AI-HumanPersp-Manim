"""
HopfieldNetworkTools - Hopfield网络可视化工具类

包含以下功能：
1. 全连接脸谱网络的绘制和管理
2. 牌组（模式）的创建和显示
3. Hopfield网络的训练和推导过程可视化
4. 权重矩阵的可视化
"""
import numpy as np
from manim import *
from src.mobjects.faces import HumanHappyFace, HumanSadFace, HumanNeutralFace

class StandardBoltzmannMachine:
    """
    标准玻尔兹曼机实现

    标准玻尔兹曼机是一种随机递归神经网络，包含可见单元和隐藏单元。
    与受限玻尔兹曼机(RBM)不同，标准BM允许任意连接，包括层内连接。

    特点：
    1. 双向连接：可见-隐藏、隐藏-隐藏、可见-可见
    2. 对称权重：w_ij = w_ji
    3. 无自连接：w_ii = 0
    4. 基于能量的模型
    """

    def __init__(self, context, n_visible=6, n_invisible=30, on_weight_change=None, on_node_value_change=None):
        """
        初始化标准玻尔兹曼机

        Args:
            context: Manim场景上下文，用于可视化
            n_visible: 可见单元数量
            n_invisible: 隐藏单元数量
            on_weight_change: 权重变化回调函数 (context, i, j) => void
            on_node_value_change: 节点值变化回调函数 (context, i, value) => void
        """
        self.context = context
        self.n_visible = n_visible
        self.n_invisible = n_invisible
        self.n_total = n_visible + n_invisible

        # 回调函数
        self.on_weight_change = on_weight_change
        self.on_node_value_change = on_node_value_change

        # 初始化权重矩阵（对称，无自连接）
        self.weights = self._initialize_weights()

        # 初始化单元状态（二进制：0或1）
        self.visible_states = np.zeros(n_visible, dtype=int)
        self.invisible_states = np.zeros(n_invisible, dtype=int)

        # 偏置项
        self.visible_biases = np.zeros(n_visible)
        self.invisible_biases = np.zeros(n_invisible)

        # 学习参数
        self.learning_rate = 0.1
        self.temperature = 1.0

        # 训练统计
        self.energy_history = []

    def _initialize_weights(self):
        """
        初始化权重矩阵

        使用小随机值初始化，确保对称性且无自连接

        Returns:
            np.ndarray: 初始化的权重矩阵
        """
        # 创建对称的随机权重矩阵
        weights = np.random.normal(0, 0.1, (self.n_total, self.n_total))

        # 确保对称性
        weights = (weights + weights.T) / 2

        # 去除自连接
        np.fill_diagonal(weights, 0)

        return weights

    def get_all_states(self):
        """
        获取所有单元的状态

        Returns:
            np.ndarray: 合并的状态向量 [可见状态, 隐藏状态]
        """
        return np.concatenate([self.visible_states, self.invisible_states])

    def set_visible_states(self, states):
        """
        设置可见单元状态

        Args:
            states: 可见单元状态数组
        """
        self.visible_states = np.array(states, dtype=int)

        # 触发可视化回调
        if self.on_node_value_change:
            for i, state in enumerate(states):
                self.on_node_value_change(self.context, i, state)

    def set_invisible_states(self, states):
        """
        设置隐藏单元状态

        Args:
            states: 隐藏单元状态数组
        """
        self.invisible_states = np.array(states, dtype=int)

        # 触发可视化回调
        if self.on_node_value_change:
            for i, state in enumerate(states):
                self.on_node_value_change(self.context, i + self.n_visible, state)

    def compute_energy(self, visible_states=None, invisible_states=None):
        """
        计算系统能量

        能量函数：E = -∑∑ w_ij * s_i * s_j - ∑ b_i * s_i
        其中 s_i, s_j 是单元状态，w_ij 是权重，b_i 是偏置

        Args:
            visible_states: 可见单元状态（可选，默认使用当前状态）
            invisible_states: 隐藏单元状态（可选，默认使用当前状态）

        Returns:
            float: 系统能量值
        """
        if visible_states is None:
            visible_states = self.visible_states
        if invisible_states is None:
            invisible_states = self.invisible_states

        all_states = np.concatenate([visible_states, invisible_states])

        # 计算权重项：-∑∑ w_ij * s_i * s_j
        weight_energy = 0
        for i in range(self.n_total):
            for j in range(i + 1, self.n_total):  # 避免重复计算
                weight_energy += self.weights[i, j] * all_states[i] * all_states[j]

        # 计算偏置项：-∑ b_i * s_i
        all_biases = np.concatenate([self.visible_biases, self.invisible_biases])
        bias_energy = np.dot(all_biases, all_states)

        return -(weight_energy + bias_energy)

    def compute_activation_probability(self, unit_index, states=None):
        """
        计算单元激活概率

        使用 sigmoid 函数：P(s_i = 1) = σ(∑ w_ij * s_j + b_i)

        Args:
            unit_index: 要计算的单元索引
            states: 其他单元的状态（可选，默认使用当前状态）

        Returns:
            float: 激活概率 [0, 1]
        """
        if states is None:
            states = self.get_all_states()

        # 计算净输入
        net_input = 0
        for j in range(self.n_total):
            if j != unit_index:  # 不包括自身
                net_input += self.weights[unit_index, j] * states[j]

        # 添加偏置
        if unit_index < self.n_visible:
            net_input += self.visible_biases[unit_index]
        else:
            net_input += self.invisible_biases[unit_index - self.n_visible]

        # 应用温度参数和 sigmoid 函数
        return 1.0 / (1.0 + np.exp(-net_input / self.temperature))

    def sample_unit(self, unit_index, states=None):
        """
        对单个单元进行采样

        Args:
            unit_index: 要采样的单元索引
            states: 其他单元的状态（可选）

        Returns:
            int: 采样结果 (0 或 1)
        """
        prob = self.compute_activation_probability(unit_index, states)
        return 1 if np.random.random() < prob else 0

    def gibbs_sampling_step(self, update_visible=True, update_invisible=True):
        """
        执行一步Gibbs采样

        依次更新每个单元的状态

        Args:
            update_visible: 是否更新可见单元
            update_invisible: 是否更新隐藏单元
        """
        all_states = self.get_all_states()

        # 更新可见单元
        if update_visible:
            for i in range(self.n_visible):
                new_state = self.sample_unit(i, all_states)
                all_states[i] = new_state
                self.visible_states[i] = new_state

                # 触发可视化回调
                if self.on_node_value_change:
                    self.on_node_value_change(self.context, i, new_state)

        # 更新隐藏单元
        if update_invisible:
            for i in range(self.n_invisible):
                unit_index = i + self.n_visible
                new_state = self.sample_unit(unit_index, all_states)
                all_states[unit_index] = new_state
                self.invisible_states[i] = new_state

                # 触发可视化回调
                if self.on_node_value_change:
                    self.on_node_value_change(self.context, unit_index, new_state)

    def run_gibbs_chain(self, steps=100, update_visible=True, update_invisible=True):
        """
        运行Gibbs采样链

        Args:
            steps: 采样步数
            update_visible: 是否更新可见单元
            update_invisible: 是否更新隐藏单元

        Returns:
            list: 每步的能量值历史
        """
        energy_history = []

        for step in range(steps):
            self.gibbs_sampling_step(update_visible, update_invisible)
            energy = self.compute_energy()
            energy_history.append(energy)

            # 每10步记录一次
            if step % 10 == 0:
                print(f"Step {step}, Energy: {energy:.4f}")

        return energy_history

    def mean_field_inference(self, max_iterations=50, tolerance=1e-4):
        """
        平均场推理

        使用确定性的平均场近似代替随机采样

        Args:
            max_iterations: 最大迭代次数
            tolerance: 收敛容差

        Returns:
            tuple: (是否收敛, 迭代次数, 最终概率)
        """
        # 初始化平均场概率
        visible_probs = np.random.random(self.n_visible)
        invisible_probs = np.random.random(self.n_invisible)

        for iteration in range(max_iterations):
            old_visible_probs = visible_probs.copy()
            old_invisible_probs = invisible_probs.copy()

            # 更新可见单元概率
            for i in range(self.n_visible):
                net_input = self.visible_biases[i]

                # 来自其他可见单元的贡献
                for j in range(self.n_visible):
                    if i != j:
                        net_input += self.weights[i, j] * visible_probs[j]

                # 来自隐藏单元的贡献
                for j in range(self.n_invisible):
                    net_input += self.weights[i, j + self.n_visible] * invisible_probs[j]

                visible_probs[i] = 1.0 / (1.0 + np.exp(-net_input / self.temperature))

            # 更新隐藏单元概率
            for i in range(self.n_invisible):
                unit_idx = i + self.n_visible
                net_input = self.invisible_biases[i]

                # 来自可见单元的贡献
                for j in range(self.n_visible):
                    net_input += self.weights[unit_idx, j] * visible_probs[j]

                # 来自其他隐藏单元的贡献
                for j in range(self.n_invisible):
                    if i != j:
                        net_input += self.weights[unit_idx, j + self.n_visible] * invisible_probs[j]

                invisible_probs[i] = 1.0 / (1.0 + np.exp(-net_input / self.temperature))

            # 检查收敛
            visible_change = np.max(np.abs(visible_probs - old_visible_probs))
            invisible_change = np.max(np.abs(invisible_probs - old_invisible_probs))

            if max(visible_change, invisible_change) < tolerance:
                return True, iteration + 1, (visible_probs, invisible_probs)

        return False, max_iterations, (visible_probs, invisible_probs)

    def contrastive_divergence_step(self, data_batch, cd_steps=1):
        """
        对比散度学习步骤

        Args:
            data_batch: 训练数据批次 (batch_size, n_visible)
            cd_steps: CD步数（CD-k中的k）

        Returns:
            float: 平均重构误差
        """
        batch_size = len(data_batch)
        total_error = 0

        # 权重梯度累积
        weight_gradients = np.zeros_like(self.weights)
        visible_bias_gradients = np.zeros_like(self.visible_biases)
        invisible_bias_gradients = np.zeros_like(self.invisible_biases)

        for data_point in data_batch:
            # 正相位：设置可见单元为数据
            self.set_visible_states(data_point)

            # 采样隐藏单元
            for i in range(self.n_invisible):
                unit_idx = i + self.n_visible
                self.invisible_states[i] = self.sample_unit(unit_idx)

            # 记录正相位统计
            positive_visible = self.visible_states.copy()
            positive_invisible = self.invisible_states.copy()

            # 负相位：运行CD-k步
            for _ in range(cd_steps):
                self.gibbs_sampling_step(update_visible=True, update_invisible=True)

            # 记录负相位统计
            negative_visible = self.visible_states.copy()
            negative_invisible = self.invisible_states.copy()

            # 计算梯度
            all_positive = np.concatenate([positive_visible, positive_invisible])
            all_negative = np.concatenate([negative_visible, negative_invisible])

            # 权重梯度
            for i in range(self.n_total):
                for j in range(i + 1, self.n_total):
                    positive_corr = all_positive[i] * all_positive[j]
                    negative_corr = all_negative[i] * all_negative[j]
                    weight_gradients[i, j] += (positive_corr - negative_corr)
                    weight_gradients[j, i] += (positive_corr - negative_corr)

            # 偏置梯度
            visible_bias_gradients += (positive_visible - negative_visible)
            invisible_bias_gradients += (positive_invisible - negative_invisible)

            # 计算重构误差
            reconstruction_error = np.mean((data_point - negative_visible) ** 2)
            total_error += reconstruction_error

        # 更新参数
        self.weights += self.learning_rate * weight_gradients / batch_size
        self.visible_biases += self.learning_rate * visible_bias_gradients / batch_size
        self.invisible_biases += self.learning_rate * invisible_bias_gradients / batch_size

        # 确保权重对称性和无自连接
        self.weights = (self.weights + self.weights.T) / 2
        np.fill_diagonal(self.weights, 0)

        # 触发权重变化回调
        if self.on_weight_change:
            for i in range(self.n_total):
                for j in range(i + 1, self.n_total):
                    self.on_weight_change(self.context, i, j)

        return total_error / batch_size

    def train(self, training_data, epochs=100, batch_size=10, cd_steps=1, verbose=True):
        """
        训练玻尔兹曼机

        Args:
            training_data: 训练数据 (n_samples, n_visible)
            epochs: 训练轮数
            batch_size: 批次大小
            cd_steps: 对比散度步数
            verbose: 是否打印训练进度

        Returns:
            list: 训练损失历史
        """
        training_data = np.array(training_data)
        n_samples = len(training_data)
        loss_history = []

        for epoch in range(epochs):
            # 随机打乱数据
            shuffled_indices = np.random.permutation(n_samples)
            epoch_loss = 0
            n_batches = 0

            # 批次训练
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = shuffled_indices[start_idx:end_idx]
                batch = training_data[batch_indices]

                batch_loss = self.contrastive_divergence_step(batch, cd_steps)
                epoch_loss += batch_loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")

        return loss_history

    def generate_samples(self, n_samples=10, gibbs_steps=1000, burn_in=100):
        """
        生成样本

        Args:
            n_samples: 要生成的样本数
            gibbs_steps: 每个样本的Gibbs采样步数
            burn_in: 燃尽期步数

        Returns:
            np.ndarray: 生成的样本 (n_samples, n_visible)
        """
        samples = []

        for _ in range(n_samples):
            # 随机初始化
            self.visible_states = np.random.randint(0, 2, self.n_visible)
            self.invisible_states = np.random.randint(0, 2, self.n_invisible)

            # 燃尽期
            for _ in range(burn_in):
                self.gibbs_sampling_step()

            # 采样期
            for _ in range(gibbs_steps):
                self.gibbs_sampling_step()

            samples.append(self.visible_states.copy())

        return np.array(samples)

    def get_network_statistics(self):
        """
        获取网络统计信息

        Returns:
            dict: 包含各种网络统计的字典
        """
        current_energy = self.compute_energy()

        # 计算权重统计
        weight_stats = {
            'mean': np.mean(self.weights),
            'std': np.std(self.weights),
            'min': np.min(self.weights),
            'max': np.max(self.weights),
            'sparsity': np.mean(np.abs(self.weights) < 1e-6)
        }

        return {
            'current_energy': current_energy,
            'visible_states': self.visible_states.copy(),
            'invisible_states': self.invisible_states.copy(),
            'weight_statistics': weight_stats,
            'n_visible': self.n_visible,
            'n_invisible': self.n_invisible,
            'learning_rate': self.learning_rate,
            'temperature': self.temperature
        }

class BoltzmannMachineVisualizer:
    """
    标准玻尔兹曼机的 3D 可视化工具

    专门用于可视化具有可见层和多个隐藏层的标准玻尔兹曼机
    支持 3D 圆柱面布局和动态连接显示
    """

    def __init__(self, context, boltzmann_machine, visible_radius=1.5, hidden_radius=1.2, layer_spacing=0.8):
        """
        初始化可视化工具

        Args:
            context: Manim场景上下文
            boltzmann_machine: StandardBoltzmannMachine实例
            visible_radius: 可见层圆形半径
            hidden_radius: 隐藏层圆形半径
            layer_spacing: 层间距离
        """
        self.context = context
        self.bm = boltzmann_machine
        self.visible_radius = visible_radius
        self.hidden_radius = hidden_radius
        self.layer_spacing = layer_spacing

        # 计算隐藏层数量（假设每层6个节点）
        self.nodes_per_hidden_layer = 6
        self.num_hidden_layers = self.bm.n_invisible // self.nodes_per_hidden_layer

        # 存储节点对象
        self.visible_nodes = []
        self.hidden_nodes = []
        self.connection_lines = []

        # 存储节点位置
        self.visible_positions = []
        self.hidden_positions = []

        # 可视化组
        self.network_group = VGroup()

    def create_card_3d(self, value, radius=0.08):
        """创建3D空间中的牌子（圆点表示）"""
        if value == 1:
            return Dot3D(radius=radius, color=BLUE, fill_opacity=1.0)
        else:
            return Dot3D(radius=radius, color=RED, fill_opacity=1.0)

    def create_neutral_node_3d(self, radius=0.08):
        """创建中性节点"""
        return Dot3D(radius=radius, color=GRAY, fill_opacity=0.8)

    def create_3d_network_structure(self, center_pos=[0, 0, 0], show_animation=True):
        """
        创建3D圆柱面网络结构

        Args:
            center_pos: 网络中心位置
            show_animation: 是否显示创建动画
        """
        # 创建可见层（顶层，环形排列）
        visible_layer_z = center_pos[2] + (self.num_hidden_layers * self.layer_spacing) / 2

        for i in range(self.bm.n_visible):
            angle = i * 2 * PI / self.bm.n_visible - PI/2
            x = center_pos[0] + self.visible_radius * np.cos(angle)
            y = center_pos[1] + self.visible_radius * np.sin(angle)
            z = visible_layer_z

            position = [x, y, z]
            self.visible_positions.append(position)

            # 创建节点
            node = self.create_neutral_node_3d()
            node.move_to(position)
            self.visible_nodes.append(node)

            if show_animation:
                self.context.play(FadeIn(node), run_time=0.1)
            else:
                self.context.add(node)

        # 创建隐藏层（下面的层，每层环形排列）
        for layer_idx in range(self.num_hidden_layers):
            layer_z = center_pos[2] + (self.num_hidden_layers - 1 - layer_idx) * self.layer_spacing - (self.num_hidden_layers * self.layer_spacing) / 2

            layer_nodes = []
            layer_positions = []

            for i in range(self.nodes_per_hidden_layer):
                angle = i * 2 * PI / self.nodes_per_hidden_layer - PI/2
                x = center_pos[0] + self.hidden_radius * np.cos(angle)
                y = center_pos[1] + self.hidden_radius * np.sin(angle)
                z = layer_z

                position = [x, y, z]
                layer_positions.append(position)

                # 创建节点
                node = self.create_neutral_node_3d()
                node.move_to(position)
                layer_nodes.append(node)

                if show_animation:
                    self.context.play(FadeIn(node), run_time=0.05)
                else:
                    self.context.add(node)

            self.hidden_nodes.append(layer_nodes)
            self.hidden_positions.append(layer_positions)

        # 将所有节点添加到组中
        all_nodes = self.visible_nodes + [node for layer in self.hidden_nodes for node in layer]
        self.network_group.add(*all_nodes)

        if show_animation:
            self.context.wait(0.5)

    def create_connections(self, show_animation=True, connection_threshold=0.1):
        """
        创建网络连接线

        Args:
            show_animation: 是否显示动画
            connection_threshold: 权重阈值，只显示绝对值大于此阈值的连接
        """
        self.connection_lines = []

        # 获取所有位置
        all_positions = self.visible_positions.copy()
        for layer_positions in self.hidden_positions:
            all_positions.extend(layer_positions)

        connections_to_draw = []

        # 创建连接线
        for i in range(self.bm.n_total):
            for j in range(i + 1, self.bm.n_total):
                weight = self.bm.weights[i, j]

                # 只显示权重足够大的连接
                if abs(weight) > connection_threshold:
                    start_pos = all_positions[i]
                    end_pos = all_positions[j]

                    # 根据权重设置颜色和粗细
                    if weight > 0:
                        color = interpolate_color(GRAY, BLUE, min(abs(weight) / 1.0, 1))
                    else:
                        color = interpolate_color(GRAY, RED, min(abs(weight) / 1.0, 1))

                    stroke_width = max(0.5, min(abs(weight) * 2, 3))

                    line = Line3D(start_pos, end_pos, color=color, stroke_width=stroke_width)
                    self.connection_lines.append(line)
                    connections_to_draw.append(line)

        # 添加连接线到组中
        self.network_group.add(*self.connection_lines)

        if show_animation and connections_to_draw:
            # 分批显示连接线以避免动画过长
            batch_size = 20
            for i in range(0, len(connections_to_draw), batch_size):
                batch = connections_to_draw[i:i + batch_size]
                self.context.play(*[Create(line) for line in batch], run_time=0.3)
        elif not show_animation:
            self.context.add(*connections_to_draw)

    def update_node_states(self, show_animation=True):
        """根据当前状态更新节点显示"""
        animations = []

        # 更新可见节点
        for i, value in enumerate(self.bm.visible_states):
            new_node = self.create_card_3d(value)
            new_node.move_to(self.visible_positions[i])

            if show_animation:
                animations.append(Transform(self.visible_nodes[i], new_node))
            else:
                self.context.remove(self.visible_nodes[i])
                self.visible_nodes[i] = new_node
                self.context.add(new_node)

        # 更新隐藏节点
        for layer_idx in range(self.num_hidden_layers):
            for node_idx in range(self.nodes_per_hidden_layer):
                global_idx = layer_idx * self.nodes_per_hidden_layer + node_idx
                if global_idx < len(self.bm.invisible_states):
                    value = self.bm.invisible_states[global_idx]
                    new_node = self.create_card_3d(value)
                    new_node.move_to(self.hidden_positions[layer_idx][node_idx])

                    if show_animation:
                        animations.append(Transform(self.hidden_nodes[layer_idx][node_idx], new_node))
                    else:
                        self.context.remove(self.hidden_nodes[layer_idx][node_idx])
                        self.hidden_nodes[layer_idx][node_idx] = new_node
                        self.context.add(new_node)

        if show_animation and animations:
            self.context.play(*animations, run_time=0.5)

    def set_visible_pattern(self, pattern, show_animation=True):
        """设置可见层模式"""
        self.bm.set_visible_states(pattern)

        # 只更新可见节点
        animations = []
        for i, value in enumerate(pattern):
            new_node = self.create_card_3d(value)
            new_node.move_to(self.visible_positions[i])

            if show_animation:
                animations.append(Transform(self.visible_nodes[i], new_node))
            else:
                self.context.remove(self.visible_nodes[i])
                self.visible_nodes[i] = new_node
                self.context.add(new_node)

        if show_animation and animations:
            self.context.play(*animations, run_time=0.3)

    def run_inference_step(self, show_animation=True):
        """运行一步推理并可视化"""
        # 执行一步Gibbs采样
        self.bm.gibbs_sampling_step(update_visible=True, update_invisible=True)

        # 更新可视化
        self.update_node_states(show_animation)

    def scale_and_move_network(self, scale_factor=0.5, target_position=[-3, 0, 0], show_animation=True):
        """缩放并移动整个网络"""
        if show_animation:
            self.context.play(
                self.network_group.animate.scale(scale_factor).move_to(target_position),
                run_time=1.0
            )
        else:
            self.network_group.scale(scale_factor).move_to(target_position)

    def create_pattern_display_3d(self, patterns, base_pos=[3, 2, 0], show_animation=True):
        """
        创建3D空间中的模式显示（沿Z轴堆叠）

        Args:
            patterns: 模式列表
            base_pos: 基准位置
            show_animation: 是否显示动画
        """
        pattern_objects = []
        dot_spacing = 0.15
        z_spacing = 0.3

        # 计算每列的模式数（6个一列）
        patterns_per_column = 6
        num_columns = (len(patterns) + patterns_per_column - 1) // patterns_per_column

        for pattern_idx, pattern in enumerate(patterns):
            # 计算3D位置
            column = pattern_idx // patterns_per_column
            row = pattern_idx % patterns_per_column

            pattern_y = base_pos[1] - row * 0.4
            pattern_z = base_pos[2] + column * z_spacing

            # 创建模式标签
            label = Text(f"模式{pattern_idx+1}", font_size=10, color="#FFD700")
            label.move_to([base_pos[0] - 0.8, pattern_y, pattern_z])

            if show_animation:
                self.context.play(FadeIn(label), run_time=0.1)
            else:
                self.context.add(label)

            # 创建模式圆点
            pattern_dots = []
            for i, value in enumerate(pattern):
                dot = self.create_card_3d(value, radius=0.06)
                pos = [base_pos[0] + (i - 2.5) * dot_spacing, pattern_y, pattern_z]
                dot.move_to(pos)
                pattern_dots.append(dot)

                if show_animation:
                    self.context.play(FadeIn(dot), run_time=0.05)
                else:
                    self.context.add(dot)

            pattern_objects.append({
                'label': label,
                'dots': pattern_dots,
                'pattern': pattern,
                'position': [base_pos[0], pattern_y, pattern_z]
            })

        return pattern_objects

class HopfieldNetworkTools:
    """Hopfield网络可视化工具类"""

    def __init__(self, context, num_faces=6, face_size=0.6, circle_radius=1.8):
        """
        初始化Hopfield网络工具

        Args:
            context: Manim场景上下文
            num_faces: 脸的数量（神经元数量）
            face_size: 脸的尺寸
            circle_radius: 环形排列半径
        """
        self.context = context
        self.num_faces = num_faces
        self.face_size = face_size
        self.circle_radius = circle_radius
        self.base_connection_color = GRAY

        # 初始化网络状态
        self.weights = np.zeros((self.num_faces, self.num_faces))
        self.faces = []
        self.face_positions = []
        self.face_numbers = []
        self.connection_lines = {}
        self.current_emotions = [None] * self.num_faces

        # 权重矩阵显示
        self.weight_matrix_display = []

    def create_face_circle(self, center_pos, show_animation=True):
        """
        创建环形排列的脸谱网络

        Args:
            center_pos: 网络中心位置
            show_animation: 是否显示创建动画
        """
        self.faces = []
        self.face_positions = []
        self.face_numbers = []

        for i in range(self.num_faces):
            # 计算环形位置
            angle = i * 2 * PI / self.num_faces - PI/2  # 从顶部开始
            x = center_pos[0] + self.circle_radius * np.cos(angle)
            y = center_pos[1] + self.circle_radius * np.sin(angle) + 0.5
            position = [x, y, 0]
            self.face_positions.append(position)

            # 创建中性脸
            face = HumanNeutralFace(size=self.face_size)
            face.move_to(position)
            self.faces.append(face)

            # 创建数字标号
            number_label = Text(str(i + 1), font_size=16, color="#FFD700")
            number_label.move_to(position + UP * (self.face_size/2 + 0.3))
            self.face_numbers.append(number_label)

            if show_animation:
                self.context.play(FadeIn(face), FadeIn(number_label), run_time=0.15)
            else:
                self.context.add(face, number_label)

        # 创建连接线（全连接）
        self.connection_lines = {}
        for i in range(self.num_faces):
            for j in range(i + 1, self.num_faces):
                start_pos = self.face_positions[i]
                end_pos = self.face_positions[j]

                line = Line(start_pos, end_pos, color=self.base_connection_color, stroke_width=2)
                self.connection_lines[(i, j)] = line

                if show_animation:
                    self.context.play(Create(line), run_time=0.05)
                else:
                    self.context.add(line)

        if show_animation:
            self.context.wait(0.3)

    def create_weight_matrix_display(self, center_pos, show_animation=True):
        """
        创建权重矩阵显示（下三角 + 对角线）

        Args:
            center_pos: 矩阵显示中心位置
            show_animation: 是否显示创建动画
        """
        # 权重矩阵标题
        matrix_title = Text("权重矩阵（对称）", font_size=20, color="#FFD700")
        matrix_title.move_to(center_pos + UP * 2.0)

        if show_animation:
            self.context.play(Write(matrix_title), run_time=0.5)
        else:
            self.context.add(matrix_title)

        # 创建矩阵显示
        self.weight_matrix_display = []
        cell_size = 0.4

        for i in range(self.num_faces):
            row = []
            for j in range(self.num_faces):
                x = center_pos[0] + (j - 2.5) * cell_size
                y = center_pos[1] + (2.5 - i) * cell_size

                if i > j:  # 下三角部分
                    cell = Square(side_length=cell_size * 0.9, color=WHITE, stroke_width=1)
                    cell.move_to([x, y, 0])

                    value_text = DecimalNumber(0.0, num_decimal_places=1, font_size=10, color=WHITE)
                    value_text.move_to([x, y, 0])

                    cell_group = VGroup(cell, value_text)
                    row.append(cell_group)

                    if show_animation:
                        self.context.play(FadeIn(cell_group), run_time=0.05)
                    else:
                        self.context.add(cell_group)

                elif i == j:  # 对角线
                    cell = Square(side_length=cell_size * 0.9, color=GRAY, stroke_width=1, fill_opacity=0.3)
                    cell.move_to([x, y, 0])

                    value_text = Text("0", font_size=10, color=GRAY)
                    value_text.move_to([x, y, 0])

                    cell_group = VGroup(cell, value_text)
                    row.append(cell_group)

                    if show_animation:
                        self.context.play(FadeIn(cell_group), run_time=0.05)
                    else:
                        self.context.add(cell_group)
                else:
                    row.append(None)

            self.weight_matrix_display.append(row)

        if show_animation:
            self.context.wait(0.5)

    def create_card(self, value):
        """
        创建牌子（圆点表示）

        Args:
            value: 0（红色/反对）或1（蓝色/支持）

        Returns:
            Dot对象
        """
        if value == 1:
            return Dot(radius=0.08, color=BLUE, fill_opacity=1.0)
        else:
            return Dot(radius=0.08, color=RED, fill_opacity=1.0)

    def create_pattern_display(self, patterns, base_pos, group_names, show_animation=True):
        """
        创建多个模式的显示

        Args:
            patterns: 模式列表，每个模式是一个0/1数组
            base_pos: 基准位置
            group_names: 组名列表
            show_animation: 是否显示动画
        """
        self.pattern_displays = []
        dot_spacing = 0.25

        for pattern_idx, (pattern, group_name) in enumerate(zip(patterns, group_names)):
            # 计算该模式的显示位置
            pattern_y = base_pos[1] - pattern_idx * 0.8

            # 创建组标签
            group_label = Text(group_name, font_size=14, color="#FFD700")
            group_label.move_to([base_pos[0] - 1.5, pattern_y, 0])

            if show_animation:
                self.context.play(Write(group_label), run_time=0.3)
            else:
                self.context.add(group_label)

            # 创建牌组
            pattern_dots = []
            for i, value in enumerate(pattern):
                dot = self.create_card(value)
                pos = [base_pos[0] + (i - 2.5) * dot_spacing, pattern_y, 0]
                dot.move_to(pos)
                pattern_dots.append(dot)

                if show_animation:
                    self.context.play(FadeIn(dot), run_time=0.1)
                else:
                    self.context.add(dot)

            self.pattern_displays.append({
                'label': group_label,
                'dots': pattern_dots,
                'pattern': pattern,
                'base_pos': [base_pos[0], pattern_y, 0]
            })

    def distribute_pattern_to_faces(self, pattern, show_animation=True):
        """
        将模式分发到脸上

        Args:
            pattern: 要分发的模式（0/1数组）
            show_animation: 是否显示动画
        """
        for i, value in enumerate(pattern):
            # 创建临时圆点
            dot = self.create_card(value)
            dot.move_to(self.face_positions[i] + UP * 0.3)

            if show_animation:
                self.context.play(FadeIn(dot), run_time=0.2)

            # 更新脸的表情
            new_face = HumanHappyFace(size=self.face_size) if value == 1 else HumanSadFace(size=self.face_size)
            new_face.move_to(self.face_positions[i])
            self.current_emotions[i] = value

            if show_animation:
                self.context.play(
                    FadeOut(dot),
                    Transform(self.faces[i], new_face),
                    run_time=0.3
                )
            else:
                self.context.remove(dot)
                self.context.remove(self.faces[i])
                self.faces[i] = new_face
                self.context.add(new_face)

    def train_hopfield_network(self, patterns, show_animation=True):
        """
        训练Hopfield网络

        Args:
            patterns: 训练模式列表
            show_animation: 是否显示权重更新动画
        """
        for pattern in patterns:
            # 转换为双极性 (-1, 1)
            bipolar_pattern = [2*p - 1 for p in pattern]

            # Hebbian学习规则
            for i in range(self.num_faces):
                for j in range(i + 1, self.num_faces):
                    weight_delta = bipolar_pattern[i] * bipolar_pattern[j]
                    self.weights[i][j] += weight_delta
                    self.weights[j][i] += weight_delta  # 保持对称性

        if show_animation:
            self.visualize_weight_changes()

    def visualize_weight_changes(self):
        """可视化权重变化"""
        line_animations = []
        matrix_animations = []

        # 更新连接线
        for (i, j), line in self.connection_lines.items():
            weight = self.weights[i][j]

            # 根据权重调整颜色和粗细
            if weight > 0:
                color = interpolate_color(self.base_connection_color, BLUE, min(abs(weight) / 2, 1))
            elif weight < 0:
                color = interpolate_color(self.base_connection_color, RED, min(abs(weight) / 2, 1))
            else:
                color = self.base_connection_color

            stroke_width = max(1, min(abs(weight) + 1, 5))
            line_animations.append(line.animate.set_color(color).set_stroke_width(stroke_width))

        # 更新权重矩阵显示
        for i in range(self.num_faces):
            for j in range(self.num_faces):
                if self.weight_matrix_display and i > j and i < len(self.weight_matrix_display) and j < len(self.weight_matrix_display[i]):  # 只更新下三角部分，并添加边界检查
                    cell_group = self.weight_matrix_display[i][j]

                    # 确保cell_group不是None
                    if cell_group is not None:
                        weight = self.weights[i][j]

                        # 更新数值
                        new_value = DecimalNumber(weight, num_decimal_places=1, font_size=10, color=WHITE)
                        new_value.move_to(cell_group[1].get_center())

                        # 更新背景颜色
                        if weight > 0:
                            bg_color = interpolate_color(WHITE, BLUE, min(abs(weight) / 2, 0.5))
                        elif weight < 0:
                            bg_color = interpolate_color(WHITE, RED, min(abs(weight) / 2, 0.5))
                        else:
                            bg_color = WHITE

                        matrix_animations.append(Transform(cell_group[1], new_value))
                        matrix_animations.append(cell_group[0].animate.set_fill(bg_color, opacity=0.3))

        # 播放所有动画
        all_animations = line_animations + matrix_animations
        if all_animations:
            self.context.play(*all_animations, run_time=1.0)

    def network_recall(self, cue_pattern, max_iterations=10, show_animation=True):
        """
        网络回忆过程（异步更新）

        Args:
            cue_pattern: 线索模式，None表示未设置的位置
            max_iterations: 最大迭代次数
            show_animation: 是否显示收敛过程

        Returns:
            最终收敛的模式
        """
        current_state = cue_pattern.copy()

        # 设置线索部分的脸部表情
        for i, value in enumerate(current_state):
            if value is not None:
                new_face = HumanHappyFace(size=self.face_size) if value == 1 else HumanSadFace(size=self.face_size)
                new_face.move_to(self.face_positions[i])
                self.current_emotions[i] = value

                if show_animation:
                    self.context.play(Transform(self.faces[i], new_face), run_time=0.1)
                else:
                    # 直接替换脸对象，不进行场景操作
                    self.faces[i] = new_face

        # 迭代更新
        for iteration in range(max_iterations):
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
                        self.current_emotions[i] = new_state

                        # 更新脸的表情
                        new_face = HumanHappyFace(size=self.face_size) if new_state == 1 else HumanSadFace(size=self.face_size)
                        new_face.move_to(self.face_positions[i])

                        if show_animation:
                            self.context.play(Transform(self.faces[i], new_face), run_time=0.2)
                            #self.context.wait(0.1)
                        else:
                            # 直接替换脸对象，不进行场景操作
                            self.faces[i] = new_face

            if not changed:
                break

        return current_state

    def extract_pattern_from_faces(self, target_pos, label="", show_animation=True):
        """
        从脸部表情中提取模式并显示

        Args:
            target_pos: 提取结果的显示位置
            label: 结果标签
            show_animation: 是否显示动画

        Returns:
            提取的模式
        """
        extracted_pattern = []
        dot_spacing = 0.25

        for i in range(self.num_faces):
            value = self.current_emotions[i] if self.current_emotions[i] is not None else 0
            extracted_pattern.append(value)

            # 创建结果圆点
            dot = self.create_card(value)
            dot.move_to(self.face_positions[i] + UP * 0.3)

            target_dot_pos = [target_pos[0] + (i - 2.5) * dot_spacing, target_pos[1], 0]

            if show_animation:
                self.context.play(
                    FadeIn(dot),
                    dot.animate.move_to(target_dot_pos),
                    run_time=0.3
                )

        # 显示标签
        if label:
            label_text = Text(label, font_size=14, color="#FFD700")
            label_text.move_to([target_pos[0] - 1.5, target_pos[1], 0])
            if show_animation:
                self.context.play(Write(label_text), run_time=0.3)
            else:
                self.context.add(label_text)

        return extracted_pattern

    def reset_faces_to_neutral(self, show_animation=True):
        """重置所有脸为中性表情"""
        for i in range(self.num_faces):
            neutral_face = HumanNeutralFace(size=self.face_size)
            neutral_face.move_to(self.face_positions[i])
            self.current_emotions[i] = None

            if show_animation:
                self.context.play(Transform(self.faces[i], neutral_face), run_time=0.3)
            else:
                # 直接替换脸对象，不进行场景操作
                self.faces[i] = neutral_face

    def calculate_accuracy(self, pattern1, pattern2):
        """计算两个模式之间的准确率"""
        if len(pattern1) != len(pattern2):
            return 0.0

        correct = sum(1 for i in range(len(pattern1)) if pattern1[i] == pattern2[i])
        return correct / len(pattern1)

    def show_accuracy_text(self, accuracy, position, show_animation=True):
        """显示准确率文本"""
        accuracy_text = Text(f"准确率: {accuracy:.1%}", font_size=14, color="#90EE90")
        accuracy_text.move_to(position)

        if show_animation:
            self.context.play(Write(accuracy_text), run_time=0.5)
        else:
            self.context.add(accuracy_text)

        return accuracy_text
