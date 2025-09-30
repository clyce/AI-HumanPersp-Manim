"""
对比 Hopfield Network (SC07) 和 Boltzmann Machine (SC08) 的训练和推理性能
不生成 Manim 动画，纯粹进行算法性能对比
"""

import numpy as np
from src.hopfield_tools import StandardHopfieldNetwork, StandardBoltzmannMachine, HashExtendedHopfield
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'SlideComponents', 'S02_RAMvsCAM'))
from shared import (
    get_patterns_for_compare, get_cue_indices, calculate_cued_accuracy
)
import time


class NetworkComparison:
    def __init__(self):
        # 使用共享配置获取训练数据
        self.training_patterns = get_patterns_for_compare()

        print("=" * 80)
        print("🧠 Hopfield Network vs Boltzmann Machine 性能对比")
        print("=" * 80)
        print(f"训练数据: {len(self.training_patterns)} 个模式，每个 6 位")
        print(f"测试策略: 前4个模式给3位提示，后8个模式给2位提示")
        print("-" * 80)

    def get_cue_setup(self, pattern_idx):
        """获取提示设置（使用共享配置）"""
        return get_cue_indices(pattern_idx)

    def calculate_cued_accuracy(self, original_pattern, recalled_pattern, cue_indices):
        """计算仅针对未被提示部分的准确率（使用共享配置）"""
        return calculate_cued_accuracy(original_pattern, recalled_pattern, cue_indices)

    def test_hopfield_network(self):
        """测试 Hopfield Network (SC07 方法)"""
        print("🔵 测试 Hopfield Network...")

        # 创建纯算法的 Hopfield 网络
        hopfield = StandardHopfieldNetwork(n_neurons=6)

        # 训练
        start_time = time.time()
        hopfield.train(self.training_patterns)
        training_time = time.time() - start_time

        # 测试每个模式
        accuracies = []
        inference_times = []
        convergence_info = []
        detailed_results = []

        for pattern_idx, original_pattern in enumerate(self.training_patterns):
            cue_indices = self.get_cue_setup(pattern_idx)

            # 创建提示模式
            cue_pattern = [None] * 6
            for idx in cue_indices:
                cue_pattern[idx] = original_pattern[idx]

            # 推理
            start_time = time.time()
            recall_result = hopfield.recall(cue_pattern, max_iterations=50, verbose=False)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # 获取结果
            recalled_pattern = recall_result['final_state']
            convergence_info.append(recall_result['iterations'])

            # 计算准确率
            accuracy = self.calculate_cued_accuracy(original_pattern, recalled_pattern, cue_indices)
            accuracies.append(accuracy)

            # 存储详细结果
            detailed_result = {
                'pattern_idx': pattern_idx,
                'original_pattern': original_pattern.copy(),
                'cue_indices': cue_indices.copy(),
                'cue_pattern': [original_pattern[i] if i in cue_indices else None for i in range(6)],
                'result_pattern': recalled_pattern.copy(),
                'accuracy': accuracy,
                'convergence_steps': recall_result['iterations'],
                'converged': recall_result['converged']
            }
            detailed_results.append(detailed_result)

            convergence_status = "收敛" if recall_result['converged'] else "未收敛"
            print(f"  模式 {pattern_idx+1}: 提示位 {list(cue_indices)} -> 准确率 {accuracy:.1%} "
                  f"({convergence_status}, {recall_result['iterations']}步, {inference_time*1000:.1f}ms)")

        avg_accuracy = np.mean(accuracies)
        avg_inference_time = np.mean(inference_times)
        avg_convergence_steps = np.mean(convergence_info)

        return {
            'name': 'Hopfield Network',
            'training_time': training_time,
            'avg_accuracy': avg_accuracy,
            'avg_inference_time': avg_inference_time,
            'avg_convergence_steps': avg_convergence_steps,
            'accuracies': accuracies,
            'detailed_results': detailed_results
        }

    def test_boltzmann_machine(self):
        """测试 Boltzmann Machine (SC08 方法)"""
        print("🔴 测试 Boltzmann Machine...")

        # 创建一个 Mock Context
        class MockContext:
            def play(self, *args, **kwargs):
                pass
            def add(self, *args):
                pass
            def remove(self, *args):
                pass

        mock_context = MockContext()

        # 创建 Boltzmann Machine
        boltzmann_machine = StandardBoltzmannMachine(
            context=mock_context,
            n_visible=6,
            n_invisible=6  # 与SC08保持一致
        )

        # 优化参数（与SC08保持一致）
        boltzmann_machine.learning_rate = 0.01
        boltzmann_machine.temperature = 0.0  # 温度=0，完全确定性

        # 训练
        start_time = time.time()
        training_result = boltzmann_machine.train_with_convergence(
            training_data=self.training_patterns,
            max_epochs=500,
            batch_size=3,
            cd_steps=5,
            patience=30,
            min_improvement=1e-5,
            verbose=False
        )
        training_time = time.time() - start_time

        print(f"  训练结果: {'收敛' if training_result['converged'] else '达到最大轮数'} "
              f"(第{training_result['final_epoch']}轮, 损失: {training_result['final_loss']:.6f})")

        # 测试每个模式
        accuracies = []
        inference_times = []
        convergence_info = []
        detailed_results = []

        for pattern_idx, original_pattern in enumerate(self.training_patterns):
            cue_indices = self.get_cue_setup(pattern_idx)

            # 设置可见层状态
            visible_states = [0] * 6
            for idx in cue_indices:
                visible_states[idx] = original_pattern[idx]

            boltzmann_machine.set_visible_states(visible_states)
            boltzmann_machine.set_fixed_visible_indices(list(cue_indices))

            # 推理（使用多数投票方法）
            start_time = time.time()
            inference_result = boltzmann_machine.run_inference_until_convergence(
                max_steps=100,              # 减少推理步数
                voting_window=20,           # 多数投票窗口
                early_stop_threshold=0.6,   # 早停阈值（60%主导即可）
                check_interval=10,          # 检查间隔
                verbose=pattern_idx < 2     # 前两个模式显示调试信息
            )
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # 获取结果
            result_pattern = list(boltzmann_machine.visible_states)

            # 验证固定节点没有改变
            for idx in cue_indices:
                if result_pattern[idx] != original_pattern[idx]:
                    print(f"    WARNING: 固定节点 {idx} 被改变!")
                    result_pattern[idx] = original_pattern[idx]

            # 清除固定设置
            boltzmann_machine.clear_fixed_visible_indices()

            # 计算准确率
            accuracy = self.calculate_cued_accuracy(original_pattern, result_pattern, cue_indices)
            accuracies.append(accuracy)

            # 存储详细结果
            detailed_result = {
                'pattern_idx': pattern_idx,
                'original_pattern': original_pattern.copy(),
                'cue_indices': cue_indices.copy(),
                'cue_pattern': [original_pattern[i] if i in cue_indices else None for i in range(6)],
                'result_pattern': result_pattern.copy(),
                'accuracy': accuracy,
                'convergence_steps': inference_result['steps_taken'],
                'converged': inference_result['converged']
            }
            detailed_results.append(detailed_result)

            convergence_status = "收敛" if inference_result['converged'] else "未收敛"
            convergence_info.append(inference_result['steps_taken'])

            print(f"  模式 {pattern_idx+1}: 提示位 {list(cue_indices)} -> 准确率 {accuracy:.1%} "
                  f"({convergence_status}, {inference_result['steps_taken']}步, {inference_time*1000:.1f}ms)")

        avg_accuracy = np.mean(accuracies)
        avg_inference_time = np.mean(inference_times)
        avg_convergence_steps = np.mean(convergence_info)

        return {
            'name': 'Boltzmann Machine',
            'training_time': training_time,
            'avg_accuracy': avg_accuracy,
            'avg_inference_time': avg_inference_time,
            'avg_convergence_steps': avg_convergence_steps,
            'accuracies': accuracies,
            'training_converged': training_result['converged'],
            'training_epochs': training_result['final_epoch'],
            'detailed_results': detailed_results
        }

    def test_hash_extended_hopfield_mode_a(self):
        """测试 HashExtendedHopfield 模式A (完全扩展模式)"""
        print("🟢 测试 HashExtendedHopfield 模式A (完全扩展)...")

        # 创建哈希扩展Hopfield网络
        hash_hopfield = HashExtendedHopfield(n_original=6)

        # 训练
        start_time = time.time()
        hash_hopfield.train(self.training_patterns)
        training_time = time.time() - start_time

        # 测试每个模式
        accuracies = []
        inference_times = []
        convergence_info = []
        detailed_results = []

        for pattern_idx, original_pattern in enumerate(self.training_patterns):
            cue_indices = self.get_cue_setup(pattern_idx)

            # 创建提示模式
            cue_pattern = [None] * 6
            for idx in cue_indices:
                cue_pattern[idx] = original_pattern[idx]

            # 推理
            start_time = time.time()
            recall_result = hash_hopfield.recall_mode_a(cue_pattern, max_iterations=50, verbose=False)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # 获取结果
            recalled_pattern = recall_result['final_state']
            convergence_info.append(recall_result['iterations'])

            # 计算准确率
            accuracy = self.calculate_cued_accuracy(original_pattern, recalled_pattern, cue_indices)
            accuracies.append(accuracy)

            # 存储详细结果
            detailed_result = {
                'pattern_idx': pattern_idx,
                'original_pattern': original_pattern.copy(),
                'cue_indices': cue_indices.copy(),
                'cue_pattern': [original_pattern[i] if i in cue_indices else None for i in range(6)],
                'result_pattern': recalled_pattern.copy(),
                'accuracy': accuracy,
                'convergence_steps': recall_result['iterations'],
                'converged': recall_result['converged']
            }
            detailed_results.append(detailed_result)

            convergence_status = "收敛" if recall_result['converged'] else "未收敛"
            print(f"  模式 {pattern_idx+1}: 提示位 {list(cue_indices)} -> 准确率 {accuracy:.1%} "
                  f"({convergence_status}, {recall_result['iterations']}步, {inference_time*1000:.1f}ms)")

        avg_accuracy = np.mean(accuracies)
        avg_inference_time = np.mean(inference_times)
        avg_convergence_steps = np.mean(convergence_info)

        return {
            'name': 'HashExtendedHopfield 模式A',
            'training_time': training_time,
            'avg_accuracy': avg_accuracy,
            'avg_inference_time': avg_inference_time,
            'avg_convergence_steps': avg_convergence_steps,
            'accuracies': accuracies,
            'detailed_results': detailed_results
        }

    def test_hash_extended_hopfield_mode_b(self):
        """测试 HashExtendedHopfield 模式B (部分锁定模式)"""
        print("🟡 测试 HashExtendedHopfield 模式B (部分锁定)...")

        # 创建哈希扩展Hopfield网络
        hash_hopfield = HashExtendedHopfield(n_original=6)

        # 训练
        start_time = time.time()
        hash_hopfield.train(self.training_patterns)
        training_time = time.time() - start_time

        # 测试每个模式
        accuracies = []
        inference_times = []
        convergence_info = []
        detailed_results = []

        for pattern_idx, original_pattern in enumerate(self.training_patterns):
            cue_indices = self.get_cue_setup(pattern_idx)

            # 创建提示模式
            cue_pattern = [None] * 6
            for idx in cue_indices:
                cue_pattern[idx] = original_pattern[idx]

            # 推理
            start_time = time.time()
            recall_result = hash_hopfield.recall_mode_b(cue_pattern, max_iterations=50, verbose=False)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # 获取结果
            recalled_pattern = recall_result['final_state']
            convergence_info.append(recall_result['iterations'])

            # 计算准确率
            accuracy = self.calculate_cued_accuracy(original_pattern, recalled_pattern, cue_indices)
            accuracies.append(accuracy)

            # 存储详细结果
            detailed_result = {
                'pattern_idx': pattern_idx,
                'original_pattern': original_pattern.copy(),
                'cue_indices': cue_indices.copy(),
                'cue_pattern': [original_pattern[i] if i in cue_indices else None for i in range(6)],
                'result_pattern': recalled_pattern.copy(),
                'accuracy': accuracy,
                'convergence_steps': recall_result['iterations'],
                'converged': recall_result['converged']
            }
            detailed_results.append(detailed_result)

            convergence_status = "收敛" if recall_result['converged'] else "未收敛"
            print(f"  模式 {pattern_idx+1}: 提示位 {list(cue_indices)} -> 准确率 {accuracy:.1%} "
                  f"({convergence_status}, {recall_result['iterations']}步, {inference_time*1000:.1f}ms)")

        avg_accuracy = np.mean(accuracies)
        avg_inference_time = np.mean(inference_times)
        avg_convergence_steps = np.mean(convergence_info)

        return {
            'name': 'HashExtendedHopfield 模式B',
            'training_time': training_time,
            'avg_accuracy': avg_accuracy,
            'avg_inference_time': avg_inference_time,
            'avg_convergence_steps': avg_convergence_steps,
            'accuracies': accuracies,
            'detailed_results': detailed_results
        }

    def run_comparison(self):
        """运行完整对比"""
        hopfield_results = self.test_hopfield_network()
        print()
        boltzmann_results = self.test_boltzmann_machine()
        print()
        hash_a_results = self.test_hash_extended_hopfield_mode_a()
        print()
        hash_b_results = self.test_hash_extended_hopfield_mode_b()

        print("\n" + "=" * 100)
        print("📊 性能对比报告")
        print("=" * 100)

        # 训练对比
        print("🎯 训练性能:")
        print(f"  Hopfield Network:        {hopfield_results['training_time']:.3f}s (解析解)")
        print(f"  Boltzmann Machine:       {boltzmann_results['training_time']:.3f}s "
              f"({'收敛' if boltzmann_results['training_converged'] else '未收敛'}, {boltzmann_results['training_epochs']}轮)")
        print(f"  HashExtended 模式A:      {hash_a_results['training_time']:.3f}s (解析解)")
        print(f"  HashExtended 模式B:      {hash_b_results['training_time']:.3f}s (解析解)")

        # 推理对比
        print("\n⚡ 推理性能:")
        print(f"  Hopfield Network:        {hopfield_results['avg_inference_time']*1000:.1f}ms "
              f"(平均, {hopfield_results['avg_convergence_steps']:.1f}步)")
        print(f"  Boltzmann Machine:       {boltzmann_results['avg_inference_time']*1000:.1f}ms "
              f"(平均, {boltzmann_results['avg_convergence_steps']:.1f}步)")
        print(f"  HashExtended 模式A:      {hash_a_results['avg_inference_time']*1000:.1f}ms "
              f"(平均, {hash_a_results['avg_convergence_steps']:.1f}步)")
        print(f"  HashExtended 模式B:      {hash_b_results['avg_inference_time']*1000:.1f}ms "
              f"(平均, {hash_b_results['avg_convergence_steps']:.1f}步)")

        # 准确率对比
        print("\n🎯 准确率对比:")
        print(f"  Hopfield Network:        {hopfield_results['avg_accuracy']:.1%} (平均)")
        print(f"  Boltzmann Machine:       {boltzmann_results['avg_accuracy']:.1%} (平均)")
        print(f"  HashExtended 模式A:      {hash_a_results['avg_accuracy']:.1%} (平均)")
        print(f"  HashExtended 模式B:      {hash_b_results['avg_accuracy']:.1%} (平均)")

        # 详细准确率分析
        print("\n📋 详细准确率 (按模式):")
        print("  模式    Hopfield   Boltzmann   Hash-A     Hash-B")
        print("  " + "-" * 50)
        for i in range(len(self.training_patterns)):
            hopfield_acc = hopfield_results['accuracies'][i]
            boltzmann_acc = boltzmann_results['accuracies'][i]
            hash_a_acc = hash_a_results['accuracies'][i]
            hash_b_acc = hash_b_results['accuracies'][i]
            print(f"    {i+1}      {hopfield_acc:.1%}       {boltzmann_acc:.1%}     {hash_a_acc:.1%}     {hash_b_acc:.1%}")

        # 汇总行
        avg_hopfield = hopfield_results['avg_accuracy']
        avg_boltzmann = boltzmann_results['avg_accuracy']
        avg_hash_a = hash_a_results['avg_accuracy']
        avg_hash_b = hash_b_results['avg_accuracy']
        print("  " + "-" * 50)
        print(f"  平均      {avg_hopfield:.1%}       {avg_boltzmann:.1%}     {avg_hash_a:.1%}     {avg_hash_b:.1%}")

        # 详细结果表格
        self._show_detailed_results_table([hopfield_results, boltzmann_results, hash_a_results, hash_b_results])

        # 总结
        print("\n🏆 总结:")
        all_accuracies = [avg_hopfield, avg_boltzmann, avg_hash_a, avg_hash_b]
        all_names = ["Hopfield Network", "Boltzmann Machine", "HashExtended 模式A", "HashExtended 模式B"]

        best_idx = np.argmax(all_accuracies)
        winner = all_names[best_idx]
        best_accuracy = all_accuracies[best_idx]

        print(f"  准确率优胜者: {winner} ({best_accuracy:.1%})")
        print(f"  训练效率:    Hopfield系列 (解析解 vs Boltzmann迭代优化)")

        # 推理效率比较
        all_inference_times = [hopfield_results['avg_inference_time'], boltzmann_results['avg_inference_time'],
                              hash_a_results['avg_inference_time'], hash_b_results['avg_inference_time']]
        fastest_idx = np.argmin(all_inference_times)
        fastest_name = all_names[fastest_idx]
        print(f"  推理效率:    {fastest_name}")

        return hopfield_results, boltzmann_results, hash_a_results, hash_b_results

    def _show_detailed_results_table(self, all_results):
        """显示详细结果表格"""
        print("\n" + "=" * 120)
        print("📋 详细推理结果表格")
        print("=" * 120)

        # 表头
        print("模式  原始数据    提示掩码    Hopfield    Boltzmann   Hash-A      Hash-B")
        print("-" * 120)

        for i in range(len(self.training_patterns)):
            original = ''.join(map(str, self.training_patterns[i]))

            # 获取提示掩码
            cue_indices = self.get_cue_setup(i)
            cue_mask = ''.join(['*' if j in cue_indices else '_' for j in range(6)])

            # 获取各模型结果
            hopfield_result = ''.join(map(str, all_results[0]['detailed_results'][i]['result_pattern'])) if 'detailed_results' in all_results[0] else "N/A"
            boltzmann_result = ''.join(map(str, all_results[1]['detailed_results'][i]['result_pattern'])) if 'detailed_results' in all_results[1] else "N/A"
            hash_a_result = ''.join(map(str, all_results[2]['detailed_results'][i]['result_pattern']))
            hash_b_result = ''.join(map(str, all_results[3]['detailed_results'][i]['result_pattern']))

            print(f" {i+1:2d}   {original}      {cue_mask}      {hopfield_result}      {boltzmann_result}     {hash_a_result}     {hash_b_result}")

        print("-" * 120)
        print("说明: '*' 表示提示位置，'_' 表示需要推理的位置")


if __name__ == "__main__":
    comparison = NetworkComparison()
    hopfield_results, boltzmann_results, hash_a_results, hash_b_results = comparison.run_comparison()
