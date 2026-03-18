"""
对比 Hopfield Network (CAM) 和 Boltzmann Machine (Generative Model)
各展所长：HF 做联想回忆，BM 做生成采样
"""

import numpy as np
import time
import sys
import os

from src.hopfield_tools import StandardHopfieldNetwork, StandardBoltzmannMachine

sys.path.append(os.path.join(os.path.dirname(__file__), 'SlideComponents', 'S02_RAMvsCAM'))
from shared import (
    get_patterns_for_compare, get_cue_indices, calculate_cued_accuracy,
    evaluate_generated_samples
)


class NetworkComparison:
    def __init__(self):
        self.training_patterns = get_patterns_for_compare()
        print("=" * 80)
        print("Hopfield Network (CAM) vs Boltzmann Machine (Generative)")
        print("=" * 80)
        print(f"训练数据: {len(self.training_patterns)} 个模式，每个 6 位")
        print("-" * 80)

    # ===== Hopfield: 联想回忆 (CAM) =====

    def test_hopfield_recall(self):
        """测试 Hopfield Network 的联想回忆能力"""
        print("\n[Hopfield] 联想回忆测试（Content Addressable Memory）")
        print("  任务: 给定 3 位提示，回忆完整的 6 位模式")

        hopfield = StandardHopfieldNetwork(n_neurons=6)

        start_time = time.time()
        hopfield.train(self.training_patterns)
        training_time = time.time() - start_time

        accuracies = []
        for pattern_idx, original_pattern in enumerate(self.training_patterns):
            cue_indices = get_cue_indices(pattern_idx)

            cue_pattern = [None] * 6
            for idx in cue_indices:
                cue_pattern[idx] = original_pattern[idx]

            recall_result = hopfield.recall(cue_pattern, max_iterations=50, verbose=False)
            recalled_pattern = recall_result['final_state']

            accuracy = calculate_cued_accuracy(original_pattern, recalled_pattern, cue_indices)
            accuracies.append(accuracy)

            cue_str = ''.join('*' if j in cue_indices else '_' for j in range(6))
            original_str = ''.join(map(str, original_pattern))
            recalled_str = ''.join(map(str, recalled_pattern))
            status = "OK" if accuracy == 1.0 else "MISS"
            print(f"  模式 {pattern_idx+1}: {original_str}  提示 {cue_str}  → {recalled_str}  "
                  f"准确率 {accuracy:.0%} {status}")

        avg_accuracy = np.mean(accuracies)
        print(f"  训练时间: {training_time*1000:.1f}ms (解析解)")
        print(f"  平均准确率: {avg_accuracy:.1%}")
        return {"avg_accuracy": avg_accuracy, "accuracies": accuracies}

    # ===== Boltzmann Machine: 生成采样 =====

    def test_boltzmann_generation(self, n_samples=20):
        """测试 Boltzmann Machine 的生成能力"""
        print(f"\n[BM] 生成能力测试（Generative Model）")
        print(f"  任务: 从学到的分布中自由生成 {n_samples} 个样本")

        class MockContext:
            def play(self, *args, **kwargs): pass
            def add(self, *args): pass
            def remove(self, *args): pass

        bm = StandardBoltzmannMachine(MockContext(), n_visible=6, n_invisible=12)
        bm.learning_rate = 0.1
        bm.temperature = 1.0

        start_time = time.time()
        training_result = bm.train_with_convergence(
            training_data=self.training_patterns,
            max_epochs=500, batch_size=4, cd_steps=10,
            patience=50, min_improvement=1e-5, verbose=False
        )
        training_time = time.time() - start_time

        convergence = "收敛" if training_result["converged"] else "未收敛"
        print(f"  训练: {convergence} (第{training_result['final_epoch']}轮, "
              f"损失: {training_result['best_loss']:.4f}, {training_time:.1f}s)")

        start_time = time.time()
        samples = bm.generate_samples(n_samples=n_samples, gibbs_steps=500, burn_in=200)
        gen_time = time.time() - start_time

        eval_r = evaluate_generated_samples(samples, self.training_patterns)

        print(f"  生成时间: {gen_time:.1f}s ({n_samples} 个样本)")
        print(f"  生成样本:")
        tp_arr = np.array(self.training_patterns)
        for i, s in enumerate(samples):
            sample_str = ''.join(str(int(v)) for v in s)
            if i in eval_r["match_indices"]:
                tag = "MATCH"
            else:
                min_d = min(int(np.sum(np.abs(s - p))) for p in tp_arr)
                tag = f"新颖 (Hamming={min_d})"
            print(f"    样本 {i+1:2d}: {sample_str}  [{tag}]")

        print(f"\n  匹配训练数据: {eval_r['match_count']}/{eval_r['total']}")
        print(f"  新颖样本:     {eval_r['novel_count']}/{eval_r['total']}")
        print(f"  到最近训练模式的平均 Hamming 距离: {eval_r['avg_min_hamming']:.2f}")
        print(f"  比特密度: 训练={eval_r['training_density']:.2f}, 生成={eval_r['generated_density']:.2f}")
        return eval_r

    # ===== 总结 =====

    def run_comparison(self):
        hopfield_results = self.test_hopfield_recall()
        bm_results = self.test_boltzmann_generation()

        print("\n" + "=" * 80)
        print("总结")
        print("=" * 80)
        print(f"  Hopfield (CAM):  联想回忆准确率 {hopfield_results['avg_accuracy']:.1%}")
        print(f"  BM (Generative): 匹配率 {bm_results['match_count']}/{bm_results['total']}, "
              f"平均 Hamming {bm_results['avg_min_hamming']:.1f}")
        print()
        print("  两者解决的问题不同:")
        print("    HF = 给定部分信息，回忆已存储的完整模式（内容寻址）")
        print("    BM = 学习数据的概率分布，能生成新的、合理的样本（生成模型）")
        print("=" * 80)

        return hopfield_results, bm_results


if __name__ == "__main__":
    comparison = NetworkComparison()
    comparison.run_comparison()
