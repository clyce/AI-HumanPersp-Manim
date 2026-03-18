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

DEFAULT_SEED = 42

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

    def __init__(self, context, n_visible=6, n_invisible=30, on_weight_change=None, on_node_value_change=None, seed=DEFAULT_SEED):
        """
        初始化标准玻尔兹曼机

        Args:
            context: Manim场景上下文，用于可视化
            n_visible: 可见单元数量
            n_invisible: 隐藏单元数量
            on_weight_change: 权重变化回调函数 (context, i, j) => void
            on_node_value_change: 节点值变化回调函数 (context, i, value) => void
            seed: 随机种子（仅在初始化时设置一次）
        """
        self.context = context
        self.n_visible = n_visible
        self.n_invisible = n_invisible
        self.n_total = n_visible + n_invisible

        # 实例级 RNG，避免污染全局状态且保证可重现
        self.rng = np.random.default_rng(seed)

        self.on_weight_change = on_weight_change
        self.on_node_value_change = on_node_value_change

        self.weights = self._initialize_weights()

        self.visible_states = np.zeros(n_visible, dtype=int)
        self.invisible_states = np.zeros(n_invisible, dtype=int)

        self.visible_biases = np.zeros(n_visible)
        self.invisible_biases = np.zeros(n_invisible)

        self.learning_rate = 0.1
        self.temperature = 1.0

        # 推理时的退火参数
        self.inference_initial_temp = 2.0
        self.inference_final_temp = 0.1

        self.energy_history = []

        self.fixed_visible_indices = set()

    def _initialize_weights(self):
        """
        初始化权重矩阵

        使用小随机值初始化，确保对称性且无自连接
        """
        weights = self.rng.normal(0, 0.1, (self.n_total, self.n_total))
        weights = (weights + weights.T) / 2
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

    def set_fixed_visible_indices(self, indices):
        """
        设置固定的可见单元索引（推理时这些单元不会改变）

        Args:
            indices: 固定单元的索引列表
        """
        self.fixed_visible_indices = set(indices) if indices else set()

    def clear_fixed_visible_indices(self):
        """清除所有固定的可见单元索引"""
        self.fixed_visible_indices = set()

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
        # 使用矩阵运算优化计算效率
        # 对于对称权重矩阵：E_weight = 0.5 * s^T * W * s （排除对角线）
        weight_energy = 0.5 * np.dot(all_states, np.dot(self.weights, all_states))

        # 减去对角线项（自连接项），因为理论上应该为0，但数值计算可能有误差
        diagonal_energy = 0.5 * np.sum(np.diag(self.weights) * all_states * all_states)
        weight_energy -= diagonal_energy

        # 计算偏置项：-∑ b_i * s_i
        all_biases = np.concatenate([self.visible_biases, self.invisible_biases])
        bias_energy = np.dot(all_biases, all_states)

        return -(weight_energy + bias_energy)

    def compute_activation_probability(self, unit_index, states=None, temperature=None):
        """
        计算单元激活概率

        使用 sigmoid 函数：P(s_i = 1) = sigma(net_input / T)

        Args:
            unit_index: 要计算的单元索引
            states: 其他单元的状态（可选，默认使用当前状态）
            temperature: 覆盖实例温度（用于退火）
        """
        if states is None:
            states = self.get_all_states()

        T = temperature if temperature is not None else self.temperature

        # 向量化计算净输入
        net_input = np.dot(self.weights[unit_index], states) - self.weights[unit_index, unit_index] * states[unit_index]

        if unit_index < self.n_visible:
            net_input += self.visible_biases[unit_index]
        else:
            net_input += self.invisible_biases[unit_index - self.n_visible]

        if T <= 1e-8:
            return 1.0 if net_input > 0 else (0.5 if net_input == 0 else 0.0)
        else:
            exponent = -net_input / T
            exponent = np.clip(exponent, -500, 500)
            return 1.0 / (1.0 + np.exp(exponent))

    def sample_unit(self, unit_index, states=None, temperature=None):
        """
        对单个单元进行采样（使用实例级 RNG，不重置种子）

        Args:
            unit_index: 要采样的单元索引
            states: 其他单元的状态（可选）
            temperature: 覆盖实例温度（用于退火）
        """
        prob = self.compute_activation_probability(unit_index, states, temperature)

        T = temperature if temperature is not None else self.temperature
        if T <= 1e-8:
            return 1 if prob > 0.5 else 0
        else:
            return 1 if self.rng.random() < prob else 0

    def gibbs_sampling_step(self, update_visible=True, update_invisible=True, temperature=None):
        """
        执行一步Gibbs采样，随机更新顺序以避免系统性偏差

        Args:
            update_visible: 是否更新可见单元
            update_invisible: 是否更新隐藏单元
            temperature: 覆盖实例温度（用于退火）
        """
        all_states = self.get_all_states()

        # 构建需要更新的单元索引列表，并随机打乱
        units_to_update = []
        if update_visible:
            units_to_update.extend([i for i in range(self.n_visible) if i not in self.fixed_visible_indices])
        if update_invisible:
            units_to_update.extend(range(self.n_visible, self.n_total))

        self.rng.shuffle(units_to_update)

        for unit_index in units_to_update:
            new_state = self.sample_unit(unit_index, all_states, temperature)
            all_states[unit_index] = new_state

            if unit_index < self.n_visible:
                self.visible_states[unit_index] = new_state
            else:
                self.invisible_states[unit_index - self.n_visible] = new_state

            if self.on_node_value_change:
                self.on_node_value_change(self.context, unit_index, new_state)

    def run_gibbs_chain(self, steps=100, update_visible=True, update_invisible=True):
        """
        运行Gibbs采样链
        """
        energy_history = []
        for step in range(steps):
            self.gibbs_sampling_step(update_visible, update_invisible)
            energy = self.compute_energy()
            energy_history.append(energy)
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
        visible_probs = self.rng.random(self.n_visible)
        invisible_probs = self.rng.random(self.n_invisible)

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

                if self.temperature == 0:
                    # 温度=0时，使用确定性激活
                    visible_probs[i] = 1.0 if net_input > 0 else 0.0
                else:
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

                if self.temperature == 0:
                    # 温度=0时，使用确定性激活
                    invisible_probs[i] = 1.0 if net_input > 0 else 0.0
                else:
                    invisible_probs[i] = 1.0 / (1.0 + np.exp(-net_input / self.temperature))

            # 检查收敛
            visible_change = np.max(np.abs(visible_probs - old_visible_probs))
            invisible_change = np.max(np.abs(invisible_probs - old_invisible_probs))

            if max(visible_change, invisible_change) < tolerance:
                return True, iteration + 1, (visible_probs, invisible_probs)

        return False, max_iterations, (visible_probs, invisible_probs)

    def contrastive_divergence_step(self, data_batch, cd_steps=1):
        """
        对比散度学习步骤（使用实例级 RNG，无种子重置）
        """
        batch_size = len(data_batch)
        total_error = 0

        weight_gradients = np.zeros_like(self.weights)
        visible_bias_gradients = np.zeros_like(self.visible_biases)
        invisible_bias_gradients = np.zeros_like(self.invisible_biases)

        for data_point in data_batch:
            self.set_visible_states(data_point)

            for i in range(self.n_invisible):
                unit_idx = i + self.n_visible
                self.invisible_states[i] = self.sample_unit(unit_idx)

            positive_visible = self.visible_states.copy()
            positive_invisible = self.invisible_states.copy()

            for _ in range(cd_steps):
                self.gibbs_sampling_step(update_visible=True, update_invisible=True)

            negative_visible = self.visible_states.copy()
            negative_invisible = self.invisible_states.copy()

            # 向量化梯度计算
            all_positive = np.concatenate([positive_visible, positive_invisible]).astype(float)
            all_negative = np.concatenate([negative_visible, negative_invisible]).astype(float)

            weight_gradients += np.outer(all_positive, all_positive) - np.outer(all_negative, all_negative)

            visible_bias_gradients += (positive_visible - negative_visible)
            invisible_bias_gradients += (positive_invisible - negative_invisible)

            reconstruction_error = np.mean((data_point - negative_visible) ** 2)
            total_error += reconstruction_error

        self.weights += self.learning_rate * weight_gradients / batch_size
        self.visible_biases += self.learning_rate * visible_bias_gradients / batch_size
        self.invisible_biases += self.learning_rate * invisible_bias_gradients / batch_size

        self.weights = (self.weights + self.weights.T) / 2
        np.fill_diagonal(self.weights, 0)

        if self.on_weight_change:
            for i in range(self.n_total):
                for j in range(i + 1, self.n_total):
                    self.on_weight_change(self.context, i, j)

        return total_error / batch_size

    def train(self, training_data, epochs=100, batch_size=10, cd_steps=1, verbose=True):
        """
        训练玻尔兹曼机（使用实例级 RNG，无种子重置）
        """
        training_data = np.array(training_data)
        n_samples = len(training_data)
        loss_history = []

        for epoch in range(epochs):
            shuffled_indices = self.rng.permutation(n_samples)
            epoch_loss = 0
            n_batches = 0

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

    def train_with_convergence(self, training_data, max_epochs=1000, batch_size=10, cd_steps=1,
                             patience=20, min_improvement=1e-6, verbose=True):
        """
        基于收敛条件的训练（使用实例级 RNG，无种子重置）
        """
        training_data = np.array(training_data)
        n_samples = len(training_data)
        loss_history = []
        best_loss = float('inf')
        best_weights = self.weights.copy()
        best_vb = self.visible_biases.copy()
        best_hb = self.invisible_biases.copy()
        epochs_without_improvement = 0

        for epoch in range(max_epochs):
            shuffled_indices = self.rng.permutation(n_samples)
            epoch_loss = 0
            n_batches = 0

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = shuffled_indices[start_idx:end_idx]
                batch = training_data[batch_indices]

                batch_loss = self.contrastive_divergence_step(batch, cd_steps)
                epoch_loss += batch_loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)

            if avg_loss < best_loss - min_improvement:
                best_loss = avg_loss
                best_weights = self.weights.copy()
                best_vb = self.visible_biases.copy()
                best_hb = self.invisible_biases.copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, Best: {best_loss:.6f}")

            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}, no improvement for {patience} epochs")
                # 恢复到最优权重
                self.weights = best_weights
                self.visible_biases = best_vb
                self.invisible_biases = best_hb
                return {
                    'loss_history': loss_history,
                    'converged': True,
                    'final_epoch': epoch,
                    'final_loss': avg_loss,
                    'best_loss': best_loss
                }

        self.weights = best_weights
        self.visible_biases = best_vb
        self.invisible_biases = best_hb
        return {
            'loss_history': loss_history,
            'converged': False,
            'final_epoch': max_epochs - 1,
            'final_loss': loss_history[-1] if loss_history else 0,
            'best_loss': best_loss
        }

    def run_inference_until_convergence(self, max_steps=1000, voting_window=20,
                                      early_stop_threshold=0.6, check_interval=10, verbose=False):
        """
        运行推理直到收敛，使用温度退火 + 多数投票

        高温阶段让网络充分探索，低温阶段使网络收敛到能量极小值。
        """
        from collections import Counter

        # 初始化隐藏层
        self.invisible_states = (self.rng.random(self.n_invisible) > 0.5).astype(int)

        # 对未固定的可见单元随机初始化
        for i in range(self.n_visible):
            if i not in self.fixed_visible_indices:
                self.visible_states[i] = int(self.rng.random() > 0.5)

        all_states = []
        early_converged = False
        convergence_step = max_steps

        T_start = self.inference_initial_temp
        T_end = self.inference_final_temp

        for step in range(max_steps):
            # 温度退火：线性从 T_start 降到 T_end
            progress = step / max(max_steps - 1, 1)
            current_temp = T_start + (T_end - T_start) * progress

            self.gibbs_sampling_step(update_visible=True, update_invisible=True, temperature=current_temp)

            current_state = tuple(self.visible_states)
            all_states.append(current_state)

            if step > voting_window and step % check_interval == 0:
                recent_states = all_states[-voting_window:]
                state_counts = Counter(recent_states)
                most_common_state, most_common_count = state_counts.most_common(1)[0]
                dominance_ratio = most_common_count / len(recent_states)

                if verbose and step % (check_interval * 5) == 0:
                    print(f"Step {step}, T={current_temp:.3f}, Dominance: {dominance_ratio:.3f}, State: {most_common_state}")

                if dominance_ratio >= early_stop_threshold:
                    if verbose:
                        print(f"Early convergence at step {step}, dominance: {dominance_ratio:.3f}")
                    self.visible_states = np.array(most_common_state, dtype=int)
                    early_converged = True
                    convergence_step = step
                    break

        if not early_converged and len(all_states) >= voting_window:
            voting_states = all_states[-voting_window:]
            state_counts = Counter(voting_states)
            most_voted_state, vote_count = state_counts.most_common(1)[0]
            self.visible_states = np.array(most_voted_state, dtype=int)
            dominance_ratio = vote_count / len(voting_states)
            if verbose:
                print(f"Final voting: {most_voted_state}, votes: {vote_count}/{len(voting_states)} ({dominance_ratio:.3f})")
        else:
            dominance_ratio = 1.0 if early_converged else 0.0

        return {
            'converged': early_converged,
            'steps_taken': convergence_step,
            'final_dominance': dominance_ratio,
            'visible_states': self.visible_states.copy(),
            'invisible_states': self.invisible_states.copy()
        }


    def generate_samples(self, n_samples=10, gibbs_steps=1000, burn_in=100):
        """
        从学到的分布中自由生成样本。

        生成前自动清除 fixed_visible_indices，使用训练温度采样，
        生成后恢复原有的 fixed 状态。

        Args:
            n_samples: 生成样本数
            gibbs_steps: 每个样本的 Gibbs 采样步数
            burn_in: 丢弃的预热步数

        Returns:
            np.ndarray: shape (n_samples, n_visible)
        """
        saved_fixed = self.fixed_visible_indices.copy()
        self.fixed_visible_indices = set()

        samples = []
        for _ in range(n_samples):
            self.visible_states = self.rng.integers(0, 2, self.n_visible)
            self.invisible_states = self.rng.integers(0, 2, self.n_invisible)

            for _ in range(burn_in):
                self.gibbs_sampling_step(temperature=self.temperature)

            for _ in range(gibbs_steps):
                self.gibbs_sampling_step(temperature=self.temperature)

            samples.append(self.visible_states.copy())

        self.fixed_visible_indices = saved_fixed
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


class RestrictedBoltzmannMachine:
    """
    受限玻尔兹曼机 (RBM) 实现

    与 StandardBoltzmannMachine 的关键区别：
    - 仅有可见-隐藏层间连接，无层内连接（二部图结构）
    - 给定一层时另一层条件独立 → 可并行采样
    - CD 训练高效收敛

    纯计算类，不依赖 Manim context。
    """

    def __init__(self, n_visible, n_hidden, seed=DEFAULT_SEED):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.rng = np.random.default_rng(seed)

        self.W = self.rng.normal(0, 0.01, (n_visible, n_hidden))
        self.vbias = np.zeros(n_visible)
        self.hbias = np.zeros(n_hidden)

        self.lr = 0.1
        self.momentum = 0.5
        self.weight_cost = 1e-4

        self._dW = np.zeros_like(self.W)
        self._dvb = np.zeros_like(self.vbias)
        self._dhb = np.zeros_like(self.hbias)

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def h_given_v(self, v):
        """P(h_j=1 | v) for all hidden units in parallel."""
        return self._sigmoid(v @ self.W + self.hbias)

    def v_given_h(self, h):
        """P(v_i=1 | h) for all visible units in parallel."""
        return self._sigmoid(h @ self.W.T + self.vbias)

    def sample_h(self, v):
        """Sample hidden states given visible."""
        ph = self.h_given_v(v)
        return (self.rng.random(ph.shape) < ph).astype(np.float64), ph

    def sample_v(self, h):
        """Sample visible states given hidden."""
        pv = self.v_given_h(h)
        return (self.rng.random(pv.shape) < pv).astype(np.float64), pv

    def train(self, data, epochs=500, cd_steps=1, batch_size=None, verbose=False):
        """
        CD-k 训练，含 momentum 退火。

        Returns:
            dict with loss_history, converged, final_epoch, best_loss
        """
        data = np.array(data, dtype=np.float64)
        n = len(data)
        if batch_size is None:
            batch_size = n

        loss_history = []
        best_loss = float('inf')
        best_W = self.W.copy()
        best_vb = self.vbias.copy()
        best_hb = self.hbias.copy()
        patience = 80
        no_improve = 0

        for epoch in range(epochs):
            mom = 0.5 if epoch < 5 else 0.9
            idx = self.rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch = data[idx[start:start + batch_size]]
                bs = len(batch)

                # Positive phase: use probabilities for hidden
                ph_data = self.h_given_v(batch)
                pos_grad_W = batch.T @ ph_data / bs
                pos_grad_vb = batch.mean(axis=0)
                pos_grad_hb = ph_data.mean(axis=0)

                # Negative phase (CD-k): sample hidden, reconstruct visible
                h_sample = (self.rng.random(ph_data.shape) < ph_data).astype(np.float64)
                for _ in range(cd_steps):
                    pv_recon = self.v_given_h(h_sample)
                    v_recon = (self.rng.random(pv_recon.shape) < pv_recon).astype(np.float64)
                    ph_recon = self.h_given_v(v_recon)
                    h_sample = (self.rng.random(ph_recon.shape) < ph_recon).astype(np.float64)

                neg_grad_W = v_recon.T @ ph_recon / bs
                neg_grad_vb = v_recon.mean(axis=0)
                neg_grad_hb = ph_recon.mean(axis=0)

                self._dW = mom * self._dW + self.lr * (pos_grad_W - neg_grad_W - self.weight_cost * self.W)
                self._dvb = mom * self._dvb + self.lr * (pos_grad_vb - neg_grad_vb)
                self._dhb = mom * self._dhb + self.lr * (pos_grad_hb - neg_grad_hb)

                self.W += self._dW
                self.vbias += self._dvb
                self.hbias += self._dhb

                # Reconstruction error using probabilities from first CD step
                pv_from_data = self.v_given_h(
                    (self.rng.random(ph_data.shape) < ph_data).astype(np.float64)
                )
                epoch_loss += np.mean((batch - pv_from_data) ** 2)
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)

            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                best_W = self.W.copy()
                best_vb = self.vbias.copy()
                best_hb = self.hbias.copy()
                no_improve = 0
            else:
                no_improve += 1

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: loss={avg_loss:.6f}")

            if no_improve >= patience and best_loss < 0.01:
                self.W = best_W
                self.vbias = best_vb
                self.hbias = best_hb
                return {
                    'loss_history': loss_history,
                    'converged': True,
                    'final_epoch': epoch,
                    'best_loss': best_loss,
                }

        self.W = best_W
        self.vbias = best_vb
        self.hbias = best_hb
        return {
            'loss_history': loss_history,
            'converged': best_loss < 0.01,
            'final_epoch': epochs - 1,
            'best_loss': best_loss,
        }

    def generate_samples(self, n_samples=10, gibbs_steps=1000, burn_in=200):
        """Free generation via Gibbs sampling from random initialisation."""
        samples = []
        for _ in range(n_samples):
            v = (self.rng.random(self.n_visible) > 0.5).astype(np.float64)
            for _ in range(burn_in + gibbs_steps):
                h, _ = self.sample_h(v)
                v, _ = self.sample_v(h)
            samples.append(v.astype(int))
        return np.array(samples)

    def conditional_sample(self, clamped_vals, clamped_idx, n_samples=100,
                           gibbs_steps=500, burn_in=100):
        """
        Sample visible units with some positions clamped.

        Args:
            clamped_vals: dict {visible_index: value} for clamped units
            clamped_idx: list of clamped visible indices (kept for API compat)
            n_samples: number of samples to collect
            gibbs_steps: Gibbs steps per sample
            burn_in: burn-in steps discarded
        """
        samples = []
        for _ in range(n_samples):
            v = (self.rng.random(self.n_visible) > 0.5).astype(np.float64)
            for idx in clamped_idx:
                v[idx] = clamped_vals[idx]

            for step in range(burn_in + gibbs_steps):
                h, _ = self.sample_h(v)
                v, _ = self.sample_v(h)
                for idx in clamped_idx:
                    v[idx] = clamped_vals[idx]

            samples.append(v.astype(int))
        return np.array(samples)


class StandardHopfieldNetwork:
    """
    标准Hopfield网络实现（纯算法，无可视化）

    经典的Hopfield网络是一种递归神经网络，能够存储和回忆模式。
    特点：
    1. 对称权重：w_ij = w_ji
    2. 无自连接：w_ii = 0
    3. 异步更新：每次只更新一个神经元
    4. 能量函数单调下降
    """

    def __init__(self, n_neurons=6):
        """
        初始化Hopfield网络

        Args:
            n_neurons: 神经元数量
        """
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))
        self.states = np.zeros(n_neurons, dtype=int)

    def train(self, patterns):
        """
        使用Hebbian学习规则训练网络

        Args:
            patterns: 训练模式列表，每个模式是0/1数组
        """
        # 重置权重
        self.weights = np.zeros((self.n_neurons, self.n_neurons))

        for pattern in patterns:
            # 转换为双极性 (-1, 1)
            bipolar_pattern = np.array([2*p - 1 for p in pattern])

            # Hebbian学习规则：w_ij += s_i * s_j
            # 使用外积计算所有权重更新
            weight_update = np.outer(bipolar_pattern, bipolar_pattern)

            # 去除对角线（无自连接）
            np.fill_diagonal(weight_update, 0)

            # 累加权重
            self.weights += weight_update

        # 确保权重矩阵对称性（Hopfield网络的关键要求）
        self.weights = (self.weights + self.weights.T) / 2

        # 标准化权重（除以模式数量）
        self.weights = self.weights / len(patterns)

    def set_states(self, pattern):
        """
        设置网络状态

        Args:
            pattern: 状态模式，None表示未设置的位置
        """
        for i, value in enumerate(pattern):
            if value is not None:
                self.states[i] = value

    def recall(self, cue_pattern, max_iterations=50, verbose=False):
        """
        网络回忆过程（异步更新）

        Args:
            cue_pattern: 线索模式，None表示未设置的位置
            max_iterations: 最大迭代次数
            verbose: 是否打印详细信息

        Returns:
            dict: 包含最终状态和收敛信息
        """
        # 初始化状态
        current_state = np.zeros(self.n_neurons, dtype=int)
        fixed_indices = set()

        # 设置线索部分
        for i, value in enumerate(cue_pattern):
            if value is not None:
                current_state[i] = value
                fixed_indices.add(i)  # 固定线索位置

        # 确定性初始化未设置的位置（使用0作为默认值）
        for i in range(self.n_neurons):
            if i not in fixed_indices:
                current_state[i] = 0  # 使用确定性初始化，避免随机性

        # 迭代更新
        energy_history = []
        for iteration in range(max_iterations):
            old_state = current_state.copy()

            # 异步更新（只更新未固定的神经元）
            for i in range(self.n_neurons):
                if i in fixed_indices:
                    continue  # 跳过固定的线索位置

                # 计算第i个神经元的净输入
                net_input = 0
                for j in range(self.n_neurons):
                    if i != j:
                        # 将0/1状态转换为-1/1进行计算
                        bipolar_j = 2 * current_state[j] - 1
                        net_input += self.weights[i][j] * bipolar_j

                # 更新状态：net_input > 0 -> state = 1, else state = 0
                current_state[i] = 1 if net_input > 0 else 0

            # 计算能量
            energy = self._compute_energy(current_state)
            energy_history.append(energy)

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}, Energy: {energy:.4f}")

            # 检查收敛
            if np.array_equal(current_state, old_state):
                if verbose:
                    print(f"Converged at iteration {iteration}")
                return {
                    'converged': True,
                    'iterations': iteration,
                    'final_state': current_state.tolist(),
                    'energy_history': energy_history,
                    'final_energy': energy
                }

        # 未收敛
        return {
            'converged': False,
            'iterations': max_iterations,
            'final_state': current_state.tolist(),
            'energy_history': energy_history,
            'final_energy': energy_history[-1] if energy_history else 0
        }

    def _compute_energy(self, state):
        """
        计算网络能量

        Args:
            state: 网络状态 (0/1数组)

        Returns:
            float: 网络能量
        """
        # 转换为双极性
        bipolar_state = np.array([2*s - 1 for s in state])

        # 计算能量: E = -0.5 * sum(w_ij * s_i * s_j) （排除自连接）
        # 使用矩阵运算优化：E = -0.5 * s^T * W * s + 0.5 * diag(W) * s^T * s
        # 由于无自连接，diag(W) = 0，所以简化为：E = -0.5 * s^T * W * s
        energy = -0.5 * np.dot(bipolar_state, np.dot(self.weights, bipolar_state))

        return energy

    def get_network_statistics(self):
        """
        获取网络统计信息

        Returns:
            dict: 网络统计信息
        """
        return {
            'n_neurons': self.n_neurons,
            'weights': self.weights.copy(),
            'current_states': self.states.copy(),
            'weight_stats': {
                'mean': np.mean(self.weights),
                'std': np.std(self.weights),
                'min': np.min(self.weights),
                'max': np.max(self.weights)
            }
        }


class BoltzmannMachineVisualizer:
    """
    标准玻尔兹曼机的 3D 可视化工具

    专门用于可视化具有可见层和多个隐藏层的标准玻尔兹曼机
    支持 3D 圆柱面布局和动态连接显示
    使用 Manim Graph 类进行实现
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

        # 存储节点位置
        self.visible_positions = []
        self.hidden_positions = []

        # Graph对象
        self.graph = None

        # 节点和边的映射
        self.vertex_ids = {}  # node_index -> vertex_id
        self.vertex_positions = {}  # vertex_id -> position

        # 当前显示的连接阈值
        self.current_connection_threshold = 0.1

        # 兼容性属性（为了向后兼容）
        self.connection_lines = []

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
        创建3D圆柱面网络结构，使用Manim Graph

        Args:
            center_pos: 网络中心位置
            show_animation: 是否显示创建动画
        """
        # 计算所有节点位置
        self._calculate_node_positions(center_pos)

        # 创建顶点列表和布局字典
        vertices = []
        layout_dict = {}

        # 添加可见层节点
        for i in range(self.bm.n_visible):
            vertex_id = f"v{i}"
            vertices.append(vertex_id)
            self.vertex_ids[i] = vertex_id
            layout_dict[vertex_id] = self.visible_positions[i]
            self.vertex_positions[vertex_id] = self.visible_positions[i]

        # 添加隐藏层节点
        for layer_idx in range(self.num_hidden_layers):
            for node_idx in range(self.nodes_per_hidden_layer):
                global_idx = self.bm.n_visible + layer_idx * self.nodes_per_hidden_layer + node_idx
                if global_idx < self.bm.n_total:
                    vertex_id = f"h{layer_idx}_{node_idx}"
                    vertices.append(vertex_id)
                    self.vertex_ids[global_idx] = vertex_id
                    layout_dict[vertex_id] = self.hidden_positions[layer_idx][node_idx]
                    self.vertex_positions[vertex_id] = self.hidden_positions[layer_idx][node_idx]

        # 创建初始的空图（暂不添加边）
        self.graph = Graph(
            vertices=vertices,
            edges=[],  # 开始时不创建边
            layout=layout_dict,
            vertex_type=Dot3D,
            vertex_config={'radius': 0.08, 'color': GRAY, 'fill_opacity': 0.8}
        )

        if show_animation:
            self.context.play(Create(self.graph), run_time=1.0)
        else:
            self.context.add(self.graph)

    def _calculate_node_positions(self, center_pos):
        """计算所有节点的3D位置"""
        # 清空位置列表
        self.visible_positions = []
        self.hidden_positions = []

        # 计算可见层位置（顶层，环形排列）
        visible_layer_z = center_pos[2] + (self.num_hidden_layers * self.layer_spacing) / 2

        for i in range(self.bm.n_visible):
            angle = i * 2 * PI / self.bm.n_visible - PI/2
            x = center_pos[0] + self.visible_radius * np.cos(angle)
            y = center_pos[1] + self.visible_radius * np.sin(angle)
            z = visible_layer_z

            self.visible_positions.append([x, y, z])

        # 计算隐藏层位置（下面的层，每层环形排列）
        for layer_idx in range(self.num_hidden_layers):
            layer_z = center_pos[2] + (self.num_hidden_layers - 1 - layer_idx) * self.layer_spacing - (self.num_hidden_layers * self.layer_spacing) / 2

            layer_positions = []
            for i in range(self.nodes_per_hidden_layer):
                angle = i * 2 * PI / self.nodes_per_hidden_layer - PI/2
                x = center_pos[0] + self.hidden_radius * np.cos(angle)
                y = center_pos[1] + self.hidden_radius * np.sin(angle)
                z = layer_z

                layer_positions.append([x, y, z])

            self.hidden_positions.append(layer_positions)

    def create_connections(self, show_animation=True, connection_threshold=0.1):
        """
        创建网络连接，使用Graph的边功能

        Args:
            show_animation: 是否显示动画
            connection_threshold: 权重阈值，只显示绝对值大于此阈值的连接
        """
        self.current_connection_threshold = connection_threshold

        # 收集要创建的边
        edges_to_add = []
        edge_configs = {}

        # 遍历权重矩阵创建边
        for i in range(self.bm.n_total):
            for j in range(i + 1, self.bm.n_total):
                weight = self.bm.weights[i, j]

                # 只显示权重足够大的连接
                if abs(weight) > connection_threshold:
                    vertex_i = self.vertex_ids[i]
                    vertex_j = self.vertex_ids[j]
                    edge = (vertex_i, vertex_j)
                    edges_to_add.append(edge)

                    # 根据权重设置颜色和粗细
                    if weight > 0:
                        color = interpolate_color(GRAY, BLUE, min(abs(weight) / 1.0, 1))
                    else:
                        color = interpolate_color(GRAY, RED, min(abs(weight) / 1.0, 1))

                    stroke_width = max(0.5, min(abs(weight) * 2, 3))

                    edge_configs[edge] = {
                        'stroke_color': color,
                        'stroke_width': stroke_width
                    }

        # 批量添加边到图中
        if edges_to_add:
            if show_animation:
                # 分批添加边以避免动画过长
                batch_size = 20
                for batch_start in range(0, len(edges_to_add), batch_size):
                    batch_end = min(batch_start + batch_size, len(edges_to_add))
                    batch_edges = edges_to_add[batch_start:batch_end]
                    batch_configs = {edge: edge_configs[edge] for edge in batch_edges}

                    self.graph.add_edges(*batch_edges, edge_config=batch_configs)
                    self.context.play(*[Create(self.graph.edges[edge]) for edge in batch_edges], run_time=0.3)
            else:
                self.graph.add_edges(*edges_to_add, edge_config=edge_configs)
                self.context.add(*[self.graph.edges[edge] for edge in edges_to_add])

        # 更新兼容性属性
        self._update_connection_lines_compatibility()

    def _update_connection_lines_compatibility(self):
        """更新兼容性属性 connection_lines"""
        if self.graph and hasattr(self.graph, 'edges'):
            self.connection_lines = list(self.graph.edges.values())
        else:
            self.connection_lines = []

    def update_node_states(self, show_animation=True):
        """根据当前状态更新节点显示"""
        animations = []
        vertex_updates = {}

        # 更新可见节点
        for i, value in enumerate(self.bm.visible_states):
            vertex_id = self.vertex_ids[i]

            if value == 1:
                new_config = {'color': BLUE, 'fill_opacity': 1.0}
            else:
                new_config = {'color': RED, 'fill_opacity': 1.0}

            vertex_updates[vertex_id] = new_config

            if show_animation:
                old_vertex = self.graph.vertices[vertex_id]
                animations.append(old_vertex.animate.set_color(new_config['color']).set_fill(opacity=new_config['fill_opacity']))
            else:
                self.graph.vertices[vertex_id].set_color(new_config['color']).set_fill(opacity=new_config['fill_opacity'])

        # 更新隐藏节点
        for layer_idx in range(self.num_hidden_layers):
            for node_idx in range(self.nodes_per_hidden_layer):
                global_idx = self.bm.n_visible + layer_idx * self.nodes_per_hidden_layer + node_idx
                if global_idx < self.bm.n_total and global_idx - self.bm.n_visible < len(self.bm.invisible_states):
                    value = self.bm.invisible_states[global_idx - self.bm.n_visible]
                    vertex_id = self.vertex_ids[global_idx]

                    if value == 1:
                        new_config = {'color': BLUE, 'fill_opacity': 1.0}
                    else:
                        new_config = {'color': RED, 'fill_opacity': 1.0}

                    vertex_updates[vertex_id] = new_config

                    if show_animation:
                        old_vertex = self.graph.vertices[vertex_id]
                        animations.append(old_vertex.animate.set_color(new_config['color']).set_fill(opacity=new_config['fill_opacity']))
                    else:
                        self.graph.vertices[vertex_id].set_color(new_config['color']).set_fill(opacity=new_config['fill_opacity'])

        if show_animation and animations:
            self.context.play(*animations, run_time=0.5)

    def set_visible_pattern(self, pattern, show_animation=True):
        """设置可见层模式"""
        self.bm.set_visible_states(pattern)

        # 只更新可见节点
        animations = []
        for i, value in enumerate(pattern):
            vertex_id = self.vertex_ids[i]

            if value == 1:
                new_config = {'color': BLUE, 'fill_opacity': 1.0}
            else:
                new_config = {'color': RED, 'fill_opacity': 1.0}

            if show_animation:
                old_vertex = self.graph.vertices[vertex_id]
                animations.append(old_vertex.animate.set_color(new_config['color']).set_fill(opacity=new_config['fill_opacity']))
            else:
                self.graph.vertices[vertex_id].set_color(new_config['color']).set_fill(opacity=new_config['fill_opacity'])

        if show_animation and animations:
            self.context.play(*animations, run_time=0.3)

    def set_visible_pattern_with_fixed(self, pattern, fixed_indices, show_animation=True):
        """
        设置可见层模式并固定某些节点

        Args:
            pattern: 可见层模式
            fixed_indices: 需要固定的节点索引列表
            show_animation: 是否显示动画
        """
        # 设置模式
        self.set_visible_pattern(pattern, show_animation)

        # 设置固定节点
        self.bm.set_fixed_visible_indices(fixed_indices)

    def clear_connections(self):
        """清除所有现有连接"""
        if self.graph and hasattr(self.graph, 'edges'):
            # 移除所有边
            edges_to_remove = list(self.graph.edges.keys())
            if edges_to_remove:
                self.graph.remove_edges(*edges_to_remove)

        # 更新兼容性属性
        self._update_connection_lines_compatibility()

    def recreate_connections(self, show_animation=False, connection_threshold=0.1):
        """重新创建连接（用于训练后更新权重显示）"""
        # 先清除现有连接
        self.clear_connections()

        # 重新创建连接
        self.create_connections(show_animation=show_animation, connection_threshold=connection_threshold)

    def run_inference_step(self, show_animation=True):
        """运行一步推理并可视化"""
        # 执行一步Gibbs采样
        self.bm.gibbs_sampling_step(update_visible=True, update_invisible=True)

        # 更新可视化
        self.update_node_states(show_animation)

    def run_inference_until_convergence(self, max_steps=1000, voting_window=20,
                                      early_stop_threshold=0.6, check_interval=10, show_animation=True, verbose=False):
        """
        运行推理直到收敛并可视化

        Args:
            max_steps: 最大推理步数
            voting_window: 多数投票窗口大小
            early_stop_threshold: 早停阈值（某状态占比达到此值时提前停止）
            check_interval: 检查收敛的间隔
            show_animation: 是否显示动画
            verbose: 是否打印过程信息

        Returns:
            dict: 收敛信息
        """
        # 使用Boltzmann机的收敛推理
        result = self.bm.run_inference_until_convergence(
            max_steps=max_steps,
            voting_window=voting_window,
            early_stop_threshold=early_stop_threshold,
            check_interval=check_interval,
            verbose=verbose
        )

        # 更新最终可视化
        if show_animation:
            self.update_node_states(show_animation=True)

        return result

    def scale_and_move_network(self, scale_factor=0.5, target_position=[-3, 0, 0], show_animation=True):
        """缩放并移动整个网络"""
        if show_animation:
            self.context.play(
                self.graph.animate.scale(scale_factor).move_to(target_position),
                run_time=1.0
            )
        else:
            self.graph.scale(scale_factor).move_to(target_position)

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
        patterns_per_column = 9
        num_columns = (len(patterns) + patterns_per_column - 1) // patterns_per_column

        for pattern_idx, pattern in enumerate(patterns):
            animations = []
            # 计算3D位置
            column = pattern_idx // patterns_per_column
            row = pattern_idx % patterns_per_column

            pattern_y = base_pos[1] - row * 0.4
            pattern_z = base_pos[2] + column * z_spacing

            # 创建模式标签
            label = Text(f"模式{pattern_idx+1}", font_size=10, color="#FFD700")
            label.move_to([base_pos[0] - 0.8, pattern_y, pattern_z])

            if show_animation:
                animations.append(FadeIn(label))
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
                    animations.append(FadeIn(dot))
                else:
                    self.context.add(dot)

            if show_animation and animations:
                self.context.play(*animations, run_time=0.1)

            pattern_objects.append({
                'label': label,
                'dots': pattern_dots,
                'pattern': pattern,
                'position': [base_pos[0], pattern_y, pattern_z]
            })

        return pattern_objects


class HopfieldNetworkVisualizer:
    """
    Hopfield网络可视化器

    负责处理Hopfield网络的所有可视化相关功能，
    与算法逻辑解耦，类似于BoltzmannMachineVisualizer的设计。
    """

    def __init__(self, context, hopfield_network, face_size=0.6, circle_radius=1.8):
        """
        初始化Hopfield网络可视化器

        Args:
            context: Manim场景上下文
            hopfield_network: StandardHopfieldNetwork实例
            face_size: 脸的尺寸
            circle_radius: 环形排列半径
        """
        self.context = context
        self.hopfield_network = hopfield_network
        self.face_size = face_size
        self.circle_radius = circle_radius
        self.base_connection_color = GRAY

        # 可视化元素
        self.faces = []
        self.face_positions = []
        self.face_numbers = []
        self.connection_lines = {}
        self.current_emotions = [None] * hopfield_network.n_neurons
        self.weight_matrix_display = []

    def create_face_circle(self, center_pos, show_animation=True):
        """创建环形排列的脸谱网络"""
        for i in range(self.hopfield_network.n_neurons):
            angle = i * 2 * PI / self.hopfield_network.n_neurons - PI/2
            x = center_pos[0] + self.circle_radius * np.cos(angle)
            y = center_pos[1] + self.circle_radius * np.sin(angle)
            position = [x, y, 0]
            self.face_positions.append(position)

            # 创建中性脸
            face = HumanNeutralFace(size=self.face_size)
            face.move_to(position)
            self.faces.append(face)

            # 创建编号标签
            number = Text(str(i), font_size=16, color=WHITE)
            number.move_to(position + DOWN * (self.face_size/2 + 0.3))
            self.face_numbers.append(number)

            if show_animation:
                self.context.play(FadeIn(face), Write(number), run_time=0.3)
            else:
                self.context.add(face, number)

        # 创建全连接网络的连接线（隐藏）
        for i in range(self.hopfield_network.n_neurons):
            for j in range(i + 1, self.hopfield_network.n_neurons):
                line = Line(
                    self.face_positions[i],
                    self.face_positions[j],
                    color=self.base_connection_color,
                    stroke_width=1,
                    stroke_opacity=0.3
                )
                self.connection_lines[(i, j)] = line

                if show_animation:
                    self.context.play(Create(line), run_time=0.1)
                else:
                    self.context.add(line)

    def create_weight_matrix_display(self, center_pos, show_animation=True):
        """创建权重矩阵显示"""
        n = self.hopfield_network.n_neurons
        cell_size = 0.4
        matrix_width = n * cell_size
        matrix_height = n * cell_size

        # 创建矩阵框架
        matrix_frame = Rectangle(
            width=matrix_width, height=matrix_height,
            color=WHITE, stroke_width=2, fill_opacity=0
        )
        matrix_frame.move_to(center_pos)
        self.weight_matrix_display.append(matrix_frame)

        # 创建矩阵标题
        title = Text("权重矩阵", font_size=16, color="#FFD700")
        title.move_to(center_pos + UP * (matrix_height/2 + 0.5))
        self.weight_matrix_display.append(title)

        # 创建网格和数字
        for i in range(n):
            for j in range(n):
                # 网格单元
                cell = Rectangle(
                    width=cell_size, height=cell_size,
                    color=GRAY, stroke_width=1, fill_opacity=0
                )
                cell_pos = [
                    center_pos[0] - matrix_width/2 + cell_size/2 + j*cell_size,
                    center_pos[1] + matrix_height/2 - cell_size/2 - i*cell_size,
                    0
                ]
                cell.move_to(cell_pos)

                # 权重数值
                weight_text = DecimalNumber(
                    0.0, num_decimal_places=1, font_size=10, color=WHITE
                )
                weight_text.move_to(cell_pos)

                self.weight_matrix_display.extend([cell, weight_text])

        if show_animation:
            self.context.play(*[Create(elem) for elem in self.weight_matrix_display], run_time=1.0)
        else:
            self.context.add(*self.weight_matrix_display)

    def update_weight_matrix_display(self, show_animation=True):
        """更新权重矩阵显示"""
        n = self.hopfield_network.n_neurons
        weights = self.hopfield_network.weights

        # 更新矩阵中的数值 (跳过框架和标题，从索引2开始)
        # weight_matrix_display结构: [matrix_frame, title, cell0, weight_text0, cell1, weight_text1, ...]
        weight_idx = 2
        animations = []

        for i in range(n):
            for j in range(n):
                # 每个单元包含cell和weight_text，所以weight_text在cell的下一个位置
                if weight_idx + 1 < len(self.weight_matrix_display):
                    weight_text = self.weight_matrix_display[weight_idx + 1]
                    new_value = weights[i][j]

                    if show_animation:
                        animations.append(ChangeDecimalToValue(weight_text, new_value))
                    else:
                        weight_text.set_value(new_value)

                weight_idx += 2

        if show_animation and animations:
            self.context.play(*animations, run_time=1.0)

    def update_connection_colors(self, show_animation=True):
        """根据权重更新连接线颜色"""
        weights = self.hopfield_network.weights
        animations = []

        for (i, j), line in self.connection_lines.items():
            weight = weights[i][j]

            if weight > 0.1:
                color = BLUE
                opacity = min(1.0, weight / np.max(np.abs(weights)) if np.max(np.abs(weights)) > 0 else 0.5)
            elif weight < -0.1:
                color = RED
                opacity = min(1.0, abs(weight) / np.max(np.abs(weights)) if np.max(np.abs(weights)) > 0 else 0.5)
            else:
                color = self.base_connection_color
                opacity = 0.3

            if show_animation:
                animations.append(line.animate.set_color(color).set_stroke(opacity=opacity))
            else:
                line.set_color(color).set_stroke(opacity=opacity)

        if show_animation and animations:
            self.context.play(*animations, run_time=1.0)

    def distribute_pattern_to_faces(self, pattern, show_animation=True):
        """分发模式到脸部"""
        animations = []

        # 安全检查：确保模式长度不超过可用的脸部数量
        max_faces = min(len(pattern), len(self.faces), len(self.face_positions), len(self.current_emotions))
        if max_faces == 0:
            print("Warning: No faces available for pattern distribution")
            return

        for i in range(max_faces):
            value = pattern[i] if i < len(pattern) else 0

            if value == 1:
                new_face = HumanHappyFace(size=self.face_size)
                self.current_emotions[i] = "happy"
            elif value == 0:
                new_face = HumanSadFace(size=self.face_size)
                self.current_emotions[i] = "sad"
            else:
                new_face = HumanNeutralFace(size=self.face_size)
                self.current_emotions[i] = None

            new_face.move_to(self.face_positions[i])

            if show_animation:
                animations.append(Transform(self.faces[i], new_face))
            else:
                self.faces[i] = new_face

        if show_animation and animations:
            self.context.play(*animations, run_time=1.0)

    def visualize_recall_process(self, cue_pattern, show_animation=True):
        """可视化回忆过程"""
        # 确保cue_pattern长度正确
        if len(cue_pattern) != self.hopfield_network.n_neurons:
            print(f"Warning: cue_pattern length {len(cue_pattern)} != n_neurons {self.hopfield_network.n_neurons}")
            return [0] * self.hopfield_network.n_neurons

        # 设置初始状态
        self.hopfield_network.set_states(cue_pattern)

        # 检查states长度
        current_states = self.hopfield_network.states.tolist()
        if len(current_states) != self.hopfield_network.n_neurons:
            print(f"Warning: states length mismatch {len(current_states)} != {self.hopfield_network.n_neurons}")
            current_states = current_states[:self.hopfield_network.n_neurons] + [0] * max(0, self.hopfield_network.n_neurons - len(current_states))

        self.distribute_pattern_to_faces(current_states, show_animation)

        if show_animation:
            self.context.wait(0.5)

        # 执行回忆过程
        result = self.hopfield_network.recall(cue_pattern, verbose=False)

        # 显示最终回忆结果
        if 'final_state' in result and result['final_state'] is not None:
            final_pattern = result['final_state']
            # 确保final_pattern长度正确
            if len(final_pattern) != self.hopfield_network.n_neurons:
                print(f"Warning: final_pattern length {len(final_pattern)} != n_neurons {self.hopfield_network.n_neurons}")
                final_pattern = final_pattern[:self.hopfield_network.n_neurons] + [0] * max(0, self.hopfield_network.n_neurons - len(final_pattern))

            self.distribute_pattern_to_faces(final_pattern, show_animation)
            if show_animation:
                self.context.wait(0.5)
            return final_pattern

        # 如果没有final_state，返回当前状态
        final_states = self.hopfield_network.states.tolist()
        if len(final_states) != self.hopfield_network.n_neurons:
            final_states = final_states[:self.hopfield_network.n_neurons] + [0] * max(0, self.hopfield_network.n_neurons - len(final_states))
        return final_states

    def reset_faces_to_neutral(self, show_animation=True):
        """重置所有脸为中性表情"""
        animations = []

        for i in range(self.hopfield_network.n_neurons):
            neutral_face = HumanNeutralFace(size=self.face_size)
            neutral_face.move_to(self.face_positions[i])
            self.current_emotions[i] = None

            if show_animation:
                animations.append(Transform(self.faces[i], neutral_face))
            else:
                self.faces[i] = neutral_face

        if show_animation and animations:
            self.context.play(*animations, run_time=0.5)

    def create_card(self, value):
        """创建牌组卡片"""
        if value == 1:
            return Dot(radius=0.08, color=BLUE, fill_opacity=0.8)
        else:
            return Dot(radius=0.08, color=RED, fill_opacity=0.8)

    def extract_pattern_from_faces(self, target_pos, label="", show_animation=True):
        """从脸部提取模式并显示"""
        extracted_pattern = []
        dot_spacing = 0.15

        for i in range(self.hopfield_network.n_neurons):
            if self.current_emotions[i] == "happy":
                value = 1
            elif self.current_emotions[i] == "sad":
                value = 0
            else:
                value = 0  # 默认为0

            extracted_pattern.append(value)

            # 创建结果显示
            dot = self.create_card(value)
            pos = [target_pos[0] + (i - (self.hopfield_network.n_neurons-1)/2) * dot_spacing, target_pos[1], 0]
            dot.move_to(pos)

            if show_animation:
                self.context.play(FadeIn(dot), run_time=0.1)
            else:
                self.context.add(dot)

        # 添加标签
        if label:
            label_text = Text(label, font_size=12, color="#FFD700")
            label_text.move_to([target_pos[0] - 1.0, target_pos[1], 0])

            if show_animation:
                self.context.play(Write(label_text), run_time=0.3)
            else:
                self.context.add(label_text)

        return extracted_pattern

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


class HopfieldNetworkTools:
    """
    Hopfield网络工具类（高级API）

    采用解耦架构，组合使用StandardHopfieldNetwork（算法）和HopfieldNetworkVisualizer（可视化）
    提供与之前相同的接口，保持向后兼容性
    """

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

        # 创建解耦的组件
        self.hopfield_network = StandardHopfieldNetwork(n_neurons=num_faces)
        self.visualizer = HopfieldNetworkVisualizer(
            context=context,
            hopfield_network=self.hopfield_network,
            face_size=face_size,
            circle_radius=circle_radius
        )

    # 向后兼容的属性访问
    @property
    def weights(self):
        return self.hopfield_network.weights

    @property
    def faces(self):
        return self.visualizer.faces

    @property
    def face_positions(self):
        return self.visualizer.face_positions

    @property
    def face_numbers(self):
        return self.visualizer.face_numbers

    @property
    def connection_lines(self):
        return self.visualizer.connection_lines

    @property
    def current_emotions(self):
        return self.visualizer.current_emotions

    @property
    def weight_matrix_display(self):
        return self.visualizer.weight_matrix_display

    def create_face_circle(self, center_pos, show_animation=True):
        """
        创建环形排列的脸谱网络

        Args:
            center_pos: 网络中心位置
            show_animation: 是否显示创建动画
        """
        self.visualizer.create_face_circle(center_pos, show_animation)

    def create_weight_matrix_display(self, center_pos, show_animation=True):
        """
        创建权重矩阵显示

        Args:
            center_pos: 矩阵显示中心位置
            show_animation: 是否显示创建动画
        """
        self.visualizer.create_weight_matrix_display(center_pos, show_animation)

    def create_card(self, value):
        """
        创建牌子（圆点表示）

        Args:
            value: 0（红色/反对）或1（蓝色/支持）

        Returns:
            Dot对象
        """
        return self.visualizer.create_card(value)

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
        self.visualizer.distribute_pattern_to_faces(pattern, show_animation)

    def train_hopfield_network(self, patterns, show_animation=True):
        """
        训练Hopfield网络

        Args:
            patterns: 训练模式列表
            show_animation: 是否显示权重更新动画
        """
        # 训练底层网络
        self.hopfield_network.train(patterns)

        # 可视化权重变化
        if show_animation:
            self.visualizer.update_weight_matrix_display(show_animation)
            self.visualizer.update_connection_colors(show_animation)

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
        # 使用可视化器进行回忆过程
        return self.visualizer.visualize_recall_process(cue_pattern, show_animation)

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
        return self.visualizer.extract_pattern_from_faces(target_pos, label, show_animation)

    def reset_faces_to_neutral(self, show_animation=True):
        """重置所有脸为中性表情"""
        self.visualizer.reset_faces_to_neutral(show_animation)

    def calculate_accuracy(self, pattern1, pattern2):
        """计算两个模式之间的准确率"""
        return self.visualizer.calculate_accuracy(pattern1, pattern2)

    def show_accuracy_text(self, accuracy, position, show_animation=True):
        """显示准确率文本"""
        return self.visualizer.show_accuracy_text(accuracy, position, show_animation)
