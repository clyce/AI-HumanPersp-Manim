"""
SC08: 从记忆到理解 —— 结构性信用分配、BM 与 RBM

叙事结构（重设计版）：
1. 承接 SC07 容量极限 → 加隐藏层的自然想法 → 结构性信用分配问题
2. BM 的方案：概率激活 + 对比散度（用统计代替梯度）→ 全连接训练不收敛
3. RBM：去掉层内连接 → CD 高效收敛
4. 四个理解层次实验（分布学习、成对推断、多模态分布★、高阶结构）
5. 总结对比表 + 过渡下一章
"""

import numpy as np
from manim import *
from src.SlideFrames import *
from src.configs import *
from src.hopfield_tools import (
    StandardBoltzmannMachine, BoltzmannMachineVisualizer,
    HopfieldNetworkTools, StandardHopfieldNetwork,
    RestrictedBoltzmannMachine,
)
from SlideComponents.S02_RAMvsCAM.shared import (
    get_patterns_for_compare, evaluate_generated_samples,
    get_multimodal_patterns, get_xor_patterns,
    compute_conditional_stats, count_distinct_outputs,
)

DOT_SPACING = 0.15
CARD_RADIUS = 0.08


class SubConsSlideComponent(TitleSlideComponent):
    def __init__(self, context):
        super().__init__(context, "从记忆到理解")

    def render_content(self):
        self._part1_hf_hidden_layer()
        self._part2_introduce_bm()
        self._part3_introduce_rbm()
        self._part4_four_levels()
        self._part5_summary_table()

    # ======================= helpers =======================

    def _card(self, value):
        """Create a single pattern dot (blue=1, red=0), matching SC05-SC07 style."""
        color = BLUE if value == 1 else RED
        return Dot(radius=CARD_RADIUS, color=color, fill_opacity=0.8)

    def _pattern_row(self, pattern, base_pos, spacing=DOT_SPACING,
                     border_color=None):
        """
        Build a VGroup of dots for *pattern* centred at *base_pos*.
        Returns (VGroup_of_dots, list_of_individual_dots).
        If *border_color* is given, a surrounding rounded rectangle is added.
        """
        dots = []
        n = len(pattern)
        for i, v in enumerate(pattern):
            d = self._card(v)
            x = base_pos[0] + (i - (n - 1) / 2) * spacing
            d.move_to([x, base_pos[1], 0])
            dots.append(d)
        grp = VGroup(*dots)
        if border_color is not None:
            border = SurroundingRectangle(
                grp, color=border_color, buff=0.06,
                corner_radius=0.05, stroke_width=2,
            )
            grp.add(border)
        return grp, dots

    def _cue_row(self, cue, base_pos, spacing=DOT_SPACING):
        """
        Build a cue display: known bits as coloured dots, unknown as grey '?'.
        *cue* uses None for unknown positions.
        """
        objs = []
        n = len(cue)
        for i, v in enumerate(cue):
            x = base_pos[0] + (i - (n - 1) / 2) * spacing
            if v is not None:
                d = self._card(v)
                d.move_to([x, base_pos[1], 0])
                objs.append(d)
            else:
                q = Text("?", font_size=14, color=GRAY)
                q.move_to([x, base_pos[1], 0])
                objs.append(q)
        return VGroup(*objs)

    def _section_cleanup(self, *groups):
        """Fade-out everything passed in and wait a beat."""
        all_objs = []
        for g in groups:
            if isinstance(g, (list, tuple)):
                all_objs.extend(g)
            else:
                all_objs.append(g)
        if all_objs:
            self.context.play(*[FadeOut(o) for o in all_objs], run_time=0.4)

    # ---------- 2-D network diagram helpers ----------

    def _build_layer_ring(self, n, center, radius, color, node_radius=0.1):
        """Return (positions_list, dots_list) for a ring of *n* nodes."""
        positions, dots = [], []
        for i in range(n):
            angle = i * TAU / n - PI / 2
            pos = center + np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0,
            ])
            positions.append(pos)
            dot = Dot(radius=node_radius, color=color, fill_opacity=0.9)
            dot.move_to(pos)
            dots.append(dot)
        return positions, dots

    def _connect_all(self, positions, color=GRAY, width=1, opacity=0.4):
        """Fully connect a set of positions. Returns list[Line]."""
        lines = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                l = Line(positions[i], positions[j],
                         color=color, stroke_width=width,
                         stroke_opacity=opacity)
                lines.append(l)
        return lines

    def _connect_bipartite(self, pos_a, pos_b,
                           color=GRAY, width=0.6, opacity=0.25):
        """Connect every node in set A to every node in set B."""
        lines = []
        for pa in pos_a:
            for pb in pos_b:
                l = Line(pa, pb, color=color, stroke_width=width,
                         stroke_opacity=opacity)
                lines.append(l)
        return lines

    # ==================================================================
    # Part 1 — "给 Hopfield 加个隐藏层？"
    # ==================================================================

    def _part1_hf_hidden_layer(self):
        space = self.get_available_space()

        recap = Text(
            "SC07 告诉我们：Hopfield 的记忆容量 ≈ 0.14N",
            font=font_main_text, font_size=20, color="#FF6B6B",
        )
        recap.move_to([0, space["center_y"] + 1.8, 0])
        self.context.play(Write(recap), run_time=0.6)

        idea = Text(
            "自然的想法：加一个隐藏层来扩充容量？",
            font=font_heading, font_size=24, color="#FFD700",
        )
        idea.move_to([0, space["center_y"] + 1.0, 0])
        self.context.play(Write(idea), run_time=0.7)
        self.context.next_slide()

        # --- 1b. visible ring ---
        vis_center = np.array([0, space["center_y"] - 0.5, 0])
        vis_pos, vis_dots = self._build_layer_ring(
            6, vis_center, 1.2, BLUE_C)
        vis_conns = self._connect_all(vis_pos, GRAY, 1, 0.4)
        vis_label = Text("可见层", font=font_main_text,
                         font_size=14, color=BLUE_C)
        vis_label.next_to(vis_center, DOWN, buff=1.5)

        self.context.play(
            *[FadeIn(d) for d in vis_dots],
            *[Create(c) for c in vis_conns],
            FadeIn(vis_label),
            run_time=0.6,
        )
        self.context.next_slide()

        # --- add hidden layer below ---
        hid_center = vis_center + DOWN * 2.2
        hid_pos, hid_dots = self._build_layer_ring(
            6, hid_center, 0.9, GREEN_C, node_radius=0.08)
        inter_conns = self._connect_bipartite(vis_pos, hid_pos)
        hid_label = Text("隐藏层", font=font_main_text,
                         font_size=14, color=GREEN_C)
        hid_label.next_to(hid_center, DOWN, buff=1.2)
        q_mark = Text("?", font_size=36, color="#FFD700", weight=BOLD)
        q_mark.move_to(hid_center)

        self.context.play(
            *[FadeIn(d) for d in hid_dots],
            *[Create(c) for c in inter_conns],
            FadeIn(hid_label), FadeIn(q_mark),
            run_time=0.8,
        )
        self.context.next_slide()

        # --- credit-assignment explanation (shrink network to left) ---
        net_group = VGroup(
            *vis_dots, *vis_conns, vis_label,
            *hid_dots, *inter_conns, hid_label, q_mark,
        )
        self.context.play(
            net_group.animate.scale(0.5).shift(LEFT * 4.5),
            run_time=0.5,
        )

        txt_x, txt_top = 1.0, space["top"] - 0.3
        prob_title = Text("结构性信用分配问题", font=font_heading,
                          font_size=24, color="#FF6B6B")
        prob_title.move_to([txt_x, txt_top, 0])
        self.context.play(Write(prob_title), run_time=0.5)

        points = [
            ("Hebbian 规则是纯局部的", "只看这条连接两端的两个神经元"),
            ("隐藏层没有目标状态", "训练时可见层被钳制到数据——隐藏层呢？"),
            ("全局误差无法传递", "回忆出错 → 无法判断哪个隐藏神经元该负责"),
            ("反向传播也走不通", "HF 状态是离散 +1/-1，梯度在此无定义"),
        ]
        txt_objs = []
        for idx, (heading, detail) in enumerate(points):
            y = txt_top - 0.8 - idx * 0.9
            h = Text(f"✗ {heading}", font=font_main_text,
                     font_size=17, color="#FF6B6B")
            h.move_to([txt_x, y, 0])
            d = Text(detail, font=font_main_text,
                     font_size=14, color="#AAAAAA")
            d.move_to([txt_x, y - 0.3, 0])
            txt_objs.extend([h, d])
            self.context.play(Write(h), run_time=0.35)
            self.context.play(FadeIn(d), run_time=0.25)

        self.context.next_slide()

        conclusion = Text(
            "问题不在于隐藏层——而在于我们没有办法训练它",
            font=font_heading, font_size=20, color="#FFD700",
        )
        conclusion.move_to([txt_x, space["bottom"] + 1.2, 0])
        self.context.play(Write(conclusion), run_time=0.7)

        need = Text(
            "我们需要一种全新的学习方式——不依赖梯度，也能训练隐藏层",
            font=font_main_text, font_size=17, color="#90EE90",
        )
        need.move_to([txt_x, space["bottom"] + 0.5, 0])
        self.context.play(Write(need), run_time=0.7)
        self.context.next_slide()

        self._section_cleanup(
            net_group, prob_title, *txt_objs,
            recap, idea, conclusion, need,
        )

    # ==================================================================
    # Part 2 — 引入 BM："用统计代替梯度"  (+network diagram)
    # ==================================================================

    def _part2_introduce_bm(self):
        space = self.get_available_space()
        net_x = -3.8

        title2 = Text(
            "Boltzmann Machine：用统计代替梯度",
            font=font_heading, font_size=24, color="#87CEEB",
        )
        title2.move_to([0, space["top"] - 0.2, 0])
        self.context.play(Write(title2), run_time=0.6)

        # --- network diagram (left side) ---
        vis_c = np.array([net_x, space["center_y"] + 0.5, 0])
        hid_c = np.array([net_x, space["center_y"] - 1.3, 0])
        vis_pos, vis_dots = self._build_layer_ring(6, vis_c, 1.0, BLUE_C, 0.08)
        hid_pos, hid_dots = self._build_layer_ring(6, hid_c, 0.8, GREEN_C, 0.07)
        inter = self._connect_bipartite(vis_pos, hid_pos, GRAY, 0.5, 0.2)
        # intra-layer connections (the "problem" with full BM)
        intra_vis = self._connect_all(vis_pos, RED_C, 1.2, 0.5)
        intra_hid = self._connect_all(hid_pos, RED_C, 1.0, 0.5)

        vis_lbl = Text("可见层", font=font_main_text, font_size=12, color=BLUE_C)
        vis_lbl.next_to(vis_c, LEFT, buff=1.3)
        hid_lbl = Text("隐藏层", font=font_main_text, font_size=12, color=GREEN_C)
        hid_lbl.next_to(hid_c, LEFT, buff=1.1)

        self.context.play(
            *[FadeIn(d) for d in vis_dots + hid_dots],
            *[Create(c) for c in inter],
            FadeIn(vis_lbl), FadeIn(hid_lbl),
            run_time=0.5,
        )
        self.context.play(
            *[Create(c) for c in intra_vis + intra_hid],
            run_time=0.6,
        )

        intra_note = Text("层内也全连接", font=font_main_text,
                          font_size=11, color=RED_C)
        intra_note.next_to(hid_c, DOWN, buff=1.1)
        self.context.play(FadeIn(intra_note), run_time=0.3)
        self.context.next_slide()

        # --- CD explanation (right side) ---
        txt_x = 1.8
        cd_items = [
            ("1. 概率性激活", "神经元以 sigmoid 概率激活", "#F0F8FF"),
            ("2. 正相（看数据）", "钳制可见层 → 观察隐藏层反应", "#90EE90"),
            ("3. 负相（做白日梦）", "网络自由运行 → 观察自发状态", "#FFD700"),
            ("4. ΔW = 正相关联 − 负相关联", "完全不需要梯度", "#87CEEB"),
        ]
        cd_objs = []
        for idx, (head, sub, col) in enumerate(cd_items):
            y = space["top"] - 0.9 - idx * 0.8
            h = Text(head, font=font_main_text, font_size=15, color=col)
            h.move_to([txt_x, y, 0])
            s = Text(sub, font=font_main_text, font_size=12, color="#AAAAAA")
            s.move_to([txt_x, y - 0.25, 0])
            cd_objs.extend([h, s])
            self.context.play(Write(h), run_time=0.35)
            self.context.play(FadeIn(s), run_time=0.2)

        self.context.next_slide()

        # --- but full connectivity ⇒ slow ---
        problem = VGroup(
            Text("但全连接 → 负相必须逐个采样 → 极慢 → 不收敛",
                 font=font_main_text, font_size=15, color="#FF6B6B"),
            Text("实验：Full BM loss 停在 0.13–0.29",
                 font=font_main_text, font_size=14, color="#AAAAAA"),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        problem.move_to([txt_x, space["bottom"] + 1.0, 0])
        self.context.play(FadeIn(problem), run_time=0.5)
        self.context.next_slide()

        net_grp = VGroup(
            *vis_dots, *hid_dots, *inter,
            *intra_vis, *intra_hid,
            vis_lbl, hid_lbl, intra_note,
        )
        self._section_cleanup(title2, net_grp, *cd_objs, problem)

        # store intra connections for Part 3 reuse
        self._bm_vis_pos = vis_pos
        self._bm_hid_pos = hid_pos

    # ==================================================================
    # Part 3 — 引入 RBM (transition animation + loss comparison)
    # ==================================================================

    def _part3_introduce_rbm(self):
        space = self.get_available_space()
        net_x = -3.5

        title3 = Text(
            "受限 Boltzmann Machine (RBM)",
            font=font_heading, font_size=24, color="#90EE90",
        )
        title3.move_to([0, space["top"] - 0.2, 0])
        self.context.play(Write(title3), run_time=0.5)

        # --- rebuild BM network (left) with intra-layer highlighted ---
        vis_c = np.array([net_x, space["center_y"] + 0.5, 0])
        hid_c = np.array([net_x, space["center_y"] - 1.3, 0])
        vis_pos, vis_dots = self._build_layer_ring(6, vis_c, 1.0, BLUE_C, 0.08)
        hid_pos, hid_dots = self._build_layer_ring(6, hid_c, 0.8, GREEN_C, 0.07)
        inter = self._connect_bipartite(vis_pos, hid_pos, GRAY, 0.5, 0.3)
        intra_vis = self._connect_all(vis_pos, RED_C, 1.5, 0.6)
        intra_hid = self._connect_all(hid_pos, RED_C, 1.2, 0.6)

        bm_label = Text("Full BM", font=font_main_text,
                        font_size=14, color="#FF6B6B")
        bm_label.next_to(hid_c, DOWN, buff=1.1)

        self.context.play(
            *[FadeIn(d) for d in vis_dots + hid_dots],
            *[Create(c) for c in inter + intra_vis + intra_hid],
            FadeIn(bm_label),
            run_time=0.5,
        )

        key_idea = Text(
            "一个简单约束：同层不连接",
            font=font_heading, font_size=20, color="#FFD700",
        )
        key_idea.move_to([1.5, space["top"] - 0.9, 0])
        self.context.play(Write(key_idea), run_time=0.5)
        self.context.next_slide()

        # --- animate removing intra-layer connections ---
        self.context.play(
            *[FadeOut(c) for c in intra_vis + intra_hid],
            run_time=0.8,
        )
        rbm_label = Text("RBM", font=font_main_text,
                         font_size=14, color="#90EE90")
        rbm_label.move_to(bm_label.get_center())
        self.context.play(
            FadeOut(bm_label), FadeIn(rbm_label),
            run_time=0.4,
        )

        # benefits
        benefits = VGroup(
            Text("→ 给定可见层，隐藏层条件独立 → 并行采样",
                 font=font_main_text, font_size=14, color="#87CEEB"),
            Text("→ 给定隐藏层，可见层条件独立 → 并行采样",
                 font=font_main_text, font_size=14, color="#87CEEB"),
            Text("→ CD 训练终于可行",
                 font=font_main_text, font_size=15, color="#90EE90"),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        benefits.move_to([1.8, space["center_y"] + 0.0, 0])
        for b in benefits:
            self.context.play(Write(b), run_time=0.35)
        self.context.next_slide()

        # --- train RBM & show loss comparison ---
        self.training_patterns = get_patterns_for_compare()

        training_msg = Text("训练 RBM 中…",
                            font=font_main_text, font_size=14, color="#FFD700")
        training_msg.move_to([1.8, space["center_y"] - 1.5, 0])
        self.context.play(Write(training_msg), run_time=0.3)

        self.rbm = RestrictedBoltzmannMachine(
            n_visible=6, n_hidden=36, seed=42)
        result = self.rbm.train(
            self.training_patterns, epochs=2000, cd_steps=10)

        self.hf = StandardHopfieldNetwork(n_neurons=6)
        self.hf.train(self.training_patterns)

        self.context.play(FadeOut(training_msg), run_time=0.2)

        # visual loss comparison bar chart
        bm_loss = 0.21
        rbm_loss = result['best_loss']
        chart = BarChart(
            values=[bm_loss, rbm_loss],
            bar_names=["Full BM", "RBM"],
            y_range=[0, 0.3, 0.05],
            x_length=3.5, y_length=2.0,
            bar_colors=["#FF6B6B", "#90EE90"],
            bar_width=0.6,
            bar_fill_opacity=0.8,
            x_axis_config={
                "font_size": 18,
                "label_constructor": lambda s: Text(
                    str(s), font_size=16),
            },
            y_axis_config={
                "font_size": 14,
                "label_constructor": lambda s: Text(
                    str(s), font_size=12),
            },
        )
        chart.move_to([2.0, space["center_y"] - 1.8, 0])
        chart_title = Text("训练 Loss 对比",
                           font=font_main_text, font_size=14, color="#F0F8FF")
        chart_title.next_to(chart, UP, buff=0.15)
        self.context.play(FadeIn(chart), FadeIn(chart_title), run_time=0.6)

        bar_labels = chart.get_bar_labels(
            font_size=14,
            label_constructor=lambda s: Text(str(s), font_size=14),
        )
        self.context.play(FadeIn(bar_labels), run_time=0.3)
        self.context.next_slide()

        net_grp = VGroup(
            *vis_dots, *hid_dots, *inter, rbm_label,
        )
        self._section_cleanup(
            title3, net_grp, key_idea, benefits,
            chart, chart_title, bar_labels,
        )

    # ==================================================================
    # Part 4 — 四个理解层次实验
    # ==================================================================

    def _part4_four_levels(self):
        space = self.get_available_space()

        title4 = Text(
            "RBM 学到了什么？四个层次的测试",
            font=font_heading, font_size=24, color="#FFD700",
        )
        title4.move_to([0, space["top"] - 0.2, 0])
        self.context.play(Write(title4), run_time=0.6)
        self.context.next_slide()

        self._level1_distribution(space)
        self._level2_pairwise(space)
        self._level3_multimodal(space)
        self._level4_xor(space)

        self.context.play(FadeOut(title4), run_time=0.3)

    # ---------- L1: Distribution Learning ----------

    def _level1_distribution(self, space):
        l1_title = Text(
            "Level 1：分布学习 —— RBM 记住了数据长什么样吗？",
            font=font_main_text, font_size=18, color="#87CEEB",
        )
        l1_title.move_to([0, space["top"] - 0.8, 0])
        self.context.play(Write(l1_title), run_time=0.5)

        # --- show training data as dot rows (left) ---
        train_label = Text("训练数据", font=font_main_text,
                           font_size=14, color="#FFD700")
        train_label.move_to([-4.5, space["top"] - 1.4, 0])
        self.context.play(FadeIn(train_label), run_time=0.2)

        train_grps = []
        tp = self.training_patterns
        for idx, pat in enumerate(tp):
            y = space["top"] - 1.8 - idx * 0.35
            grp, _ = self._pattern_row(pat, [-4.5, y, 0])
            train_grps.append(grp)

        self.context.play(
            *[FadeIn(g) for g in train_grps], run_time=0.5)
        self.context.next_slide()

        # --- generate samples and display (right) ---
        gen_label = Text("RBM 自由生成", font=font_main_text,
                         font_size=14, color="#90EE90")
        gen_label.move_to([0.5, space["top"] - 1.4, 0])
        self.context.play(FadeIn(gen_label), run_time=0.2)

        samples = self.rbm.generate_samples(
            n_samples=16, gibbs_steps=500, burn_in=200)
        eval_result = evaluate_generated_samples(
            samples, self.training_patterns)
        match_set = set(eval_result['match_indices'])

        gen_grps = []
        n_show = min(len(samples), 16)
        for idx in range(n_show):
            y = space["top"] - 1.8 - idx * 0.35
            border = "#90EE90" if idx in match_set else "#FF6B6B"
            grp, _ = self._pattern_row(
                samples[idx].tolist(), [0.5, y, 0],
                border_color=border)
            gen_grps.append(grp)

        self.context.play(
            *[FadeIn(g) for g in gen_grps], run_time=0.6)

        # legend
        legend = VGroup(
            Text("■ 绿框 = 精确匹配训练数据",
                 font=font_main_text, font_size=11, color="#90EE90"),
            Text("■ 红框 = 不匹配",
                 font=font_main_text, font_size=11, color="#FF6B6B"),
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        legend.move_to([4.5, space["top"] - 1.6, 0])
        self.context.play(FadeIn(legend), run_time=0.3)

        # summary stats
        match_pct = eval_result["match_count"] / eval_result["total"] * 100
        stats = VGroup(
            Text(f"匹配率: {eval_result['match_count']}/{eval_result['total']}"
                 f" ({match_pct:.0f}%)",
                 font=font_main_text, font_size=14,
                 color="#90EE90" if match_pct > 80 else "#FFD700"),
            Text(f"平均 Hamming 距离: {eval_result['avg_min_hamming']:.2f}",
                 font=font_main_text, font_size=14, color="#F0F8FF"),
        ).arrange(DOWN, buff=0.15)
        stats.move_to([4.5, space["center_y"] - 0.5, 0])
        self.context.play(FadeIn(stats), run_time=0.4)

        verdict = Text(
            "✓ RBM 精确学会了训练数据的分布",
            font=font_main_text, font_size=16, color="#90EE90",
        )
        verdict.move_to([0, space["bottom"] + 0.6, 0])
        self.context.play(Write(verdict), run_time=0.5)
        self.context.next_slide()

        self._section_cleanup(
            l1_title, train_label, *train_grps,
            gen_label, *gen_grps, legend, stats, verdict,
        )

    # ---------- L2: Pairwise Inference ----------

    def _level2_pairwise(self, space):
        l2_title = Text(
            "Level 2：成对推断 —— 变量间关系？",
            font=font_main_text, font_size=18, color="#87CEEB",
        )
        l2_title.move_to([0, space["top"] - 0.8, 0])
        self.context.play(Write(l2_title), run_time=0.5)

        # setup description
        setup_txt = Text(
            "训练数据中 bit0 与 bit5 有强相关 → 固定 bit0=1，推断 bit5",
            font=font_main_text, font_size=15, color="#F0F8FF",
        )
        setup_txt.move_to([0, space["top"] - 1.4, 0])
        self.context.play(Write(setup_txt), run_time=0.5)

        # show cue
        cue = [1, None, None, None, None, None]
        cue_label = Text("线索:", font=font_main_text,
                         font_size=14, color="#FFD700")
        cue_label.move_to([-2.5, space["top"] - 2.0, 0])
        cue_display = self._cue_row(cue, [-1.2, space["top"] - 2.0, 0])
        self.context.play(FadeIn(cue_label), FadeIn(cue_display),
                          run_time=0.4)
        self.context.next_slide()

        # --- HF recall ---
        hf_label = Text("Hopfield 回忆:",
                        font=font_main_text, font_size=14, color="#87CEEB")
        hf_label.move_to([-4.0, space["center_y"] + 0.5, 0])

        hf_results_list = []
        for _ in range(20):
            r = self.hf.recall(cue)
            hf_results_list.append(r['final_state'])
        hf_p = np.mean([r[5] for r in hf_results_list])

        hf_row, _ = self._pattern_row(
            hf_results_list[0], [-1.8, space["center_y"] + 0.5, 0])

        self.context.play(FadeIn(hf_label), FadeIn(hf_row), run_time=0.4)

        # --- RBM conditional ---
        rbm_label = Text("RBM 条件采样:",
                         font=font_main_text, font_size=14, color="#90EE90")
        rbm_label.move_to([-4.0, space["center_y"] - 0.2, 0])

        rbm_samples = self.rbm.conditional_sample(
            {0: 1}, [0], n_samples=100, gibbs_steps=300)
        rbm_p = compute_conditional_stats(rbm_samples, 0, 1, 5)

        modal_pattern = rbm_samples[0].tolist()
        rbm_row, _ = self._pattern_row(
            modal_pattern, [-1.8, space["center_y"] - 0.2, 0])
        self.context.play(FadeIn(rbm_label), FadeIn(rbm_row), run_time=0.4)

        # --- probability comparison (right side) ---
        tp = np.array(self.training_patterns)
        mask = tp[:, 0] == 1
        target_p = float(np.mean(tp[mask, 5]))

        prob_title = Text("P(bit5=1 | bit0=1)",
                          font=font_main_text, font_size=14, color="#FFD700")
        prob_title.move_to([3.5, space["center_y"] + 1.2, 0])

        prob_chart = BarChart(
            values=[target_p, hf_p, rbm_p],
            bar_names=["目标", "HF", "RBM"],
            y_range=[0, 1.1, 0.2],
            x_length=3.5, y_length=2.0,
            bar_colors=["#FFD700", "#87CEEB", "#90EE90"],
            bar_width=0.5,
            bar_fill_opacity=0.8,
            x_axis_config={
                "font_size": 16,
                "label_constructor": lambda s: Text(
                    str(s), font_size=14),
            },
            y_axis_config={
                "font_size": 12,
                "label_constructor": lambda s: Text(
                    str(s), font_size=11),
            },
        )
        prob_chart.move_to([3.5, space["center_y"] - 0.2, 0])
        prob_labels = prob_chart.get_bar_labels(
            font_size=13,
            label_constructor=lambda s: Text(str(s), font_size=13),
        )
        self.context.play(
            FadeIn(prob_title), FadeIn(prob_chart),
            FadeIn(prob_labels), run_time=0.5)

        hf_note = Text(
            "Hebbian 权重天然编码成对相关 → HF 也通过了！",
            font=font_main_text, font_size=15, color="#FFD700",
        )
        hf_note.move_to([0, space["center_y"] - 1.8, 0])
        self.context.play(Write(hf_note), run_time=0.5)

        next_q = Text(
            "但——成对推断是 Hopfield 的全部能力吗？",
            font=font_main_text, font_size=16, color="#FF6B9D",
        )
        next_q.move_to([0, space["bottom"] + 0.6, 0])
        self.context.play(Write(next_q), run_time=0.5)
        self.context.next_slide()

        self._section_cleanup(
            l2_title, setup_txt, cue_label, cue_display,
            hf_label, hf_row, rbm_label, rbm_row,
            prob_title, prob_chart, prob_labels,
            hf_note, next_q,
        )

    # ---------- L3: Multi-modal conditional distribution (CORE) ----------

    def _level3_multimodal(self, space):
        l3_title = Text(
            "Level 3：多模态条件分布 ★ 核心区别 ★",
            font=font_main_text, font_size=20, color="#FF6B9D",
        )
        l3_title.move_to([0, space["top"] - 0.8, 0])
        self.context.play(Write(l3_title), run_time=0.5)

        # --- show multimodal training data ---
        setup = Text(
            "bit0=1 时有两组正确答案（各占 50%）",
            font=font_main_text, font_size=16, color="#FFD700",
        )
        setup.move_to([0, space["top"] - 1.35, 0])
        self.context.play(Write(setup), run_time=0.4)

        mm_patterns = get_multimodal_patterns()

        grp_a_label = Text("Group A (ends 011):",
                           font=font_main_text, font_size=12, color="#87CEEB")
        grp_a_label.move_to([-5.0, space["top"] - 1.9, 0])
        grp_b_label = Text("Group B (ends 100):",
                           font=font_main_text, font_size=12, color="#FFB347")
        grp_b_label.move_to([-5.0, space["top"] - 2.6, 0])

        data_grps = []
        # Group A: first 2 patterns (bit0=1, ends 011)
        for i in range(2):
            y = space["top"] - 1.9 - i * 0.3
            g, _ = self._pattern_row(
                mm_patterns[i], [-2.8, y, 0],
                border_color="#87CEEB")
            data_grps.append(g)
        # Group B: patterns 2-3 (bit0=1, ends 100)
        for i in range(2, 4):
            y = space["top"] - 2.6 - (i - 2) * 0.3
            g, _ = self._pattern_row(
                mm_patterns[i], [-2.8, y, 0],
                border_color="#FFB347")
            data_grps.append(g)

        self.context.play(
            FadeIn(grp_a_label), FadeIn(grp_b_label),
            *[FadeIn(g) for g in data_grps],
            run_time=0.5,
        )
        self.context.next_slide()

        # --- train models ---
        rbm_mm = RestrictedBoltzmannMachine(
            n_visible=6, n_hidden=36, seed=42)
        rbm_mm.train(mm_patterns, epochs=2000, cd_steps=10)

        hf_mm = StandardHopfieldNetwork(n_neurons=6)
        hf_mm.train(mm_patterns)

        # --- cue ---
        cue_txt = Text("线索: bit0 = 1",
                       font=font_main_text, font_size=14, color="#FFD700")
        cue_txt.move_to([2.0, space["top"] - 1.9, 0])
        cue_display = self._cue_row(
            [1, None, None, None, None, None],
            [4.5, space["top"] - 1.9, 0])
        self.context.play(FadeIn(cue_txt), FadeIn(cue_display), run_time=0.3)

        # === HF side (left-center) ===
        hf_col_x = -2.0
        hf_header = Text("Hopfield", font=font_heading,
                         font_size=16, color="#87CEEB")
        hf_header.move_to([hf_col_x, space["center_y"] + 0.8, 0])
        self.context.play(FadeIn(hf_header), run_time=0.2)

        hf_outputs = set()
        for _ in range(20):
            r = hf_mm.recall([1, None, None, None, None, None])
            hf_outputs.add(tuple(r['final_state']))

        hf_rows = []
        unique_hf = list(hf_outputs)
        for idx, pat in enumerate(unique_hf[:4]):
            y = space["center_y"] + 0.3 - idx * 0.35
            g, _ = self._pattern_row(list(pat), [hf_col_x, y, 0])
            hf_rows.append(g)
        self.context.play(*[FadeIn(r) for r in hf_rows], run_time=0.4)

        hf_verdict = Text(
            f"始终同一个输出 ({len(hf_outputs)} 种)",
            font=font_main_text, font_size=13, color="#FF6B6B",
        )
        hf_verdict.move_to(
            [hf_col_x, space["center_y"] + 0.3 - len(unique_hf[:4]) * 0.35 - 0.2, 0])
        self.context.play(FadeIn(hf_verdict), run_time=0.3)

        # === RBM side (right-center) ===
        rbm_col_x = 3.0
        rbm_header = Text("RBM", font=font_heading,
                          font_size=16, color="#90EE90")
        rbm_header.move_to([rbm_col_x, space["center_y"] + 0.8, 0])
        self.context.play(FadeIn(rbm_header), run_time=0.2)

        rbm_samples = rbm_mm.conditional_sample(
            {0: 1}, [0], n_samples=100, gibbs_steps=500)
        rbm_stats = count_distinct_outputs(rbm_samples)
        rbm_p5 = compute_conditional_stats(rbm_samples, 0, 1, 5)

        # frequency bar chart for RBM outputs
        dist = rbm_stats['distribution']
        top_patterns = list(dist.items())[:6]
        bar_vals = [v for _, v in top_patterns]
        bar_names_list = []
        for pat, _ in top_patterns:
            tail = "".join(str(b) for b in pat[3:])
            bar_names_list.append(tail)

        if bar_vals:
            freq_chart = BarChart(
                values=bar_vals,
                bar_names=bar_names_list,
                y_range=[0, max(bar_vals) + 5, max(max(bar_vals) // 4, 1)],
                x_length=4.0, y_length=1.8,
                bar_colors=["#90EE90", "#87CEEB", "#FFD700",
                            "#FF6B9D", "#FFB347", "#DDA0DD"],
                bar_width=0.5,
                bar_fill_opacity=0.8,
                x_axis_config={
                    "font_size": 14,
                    "label_constructor": lambda s: Text(
                        str(s), font_size=12),
                },
                y_axis_config={
                    "font_size": 12,
                    "label_constructor": lambda s: Text(
                        str(s), font_size=10),
                },
            )
            freq_chart.move_to([rbm_col_x, space["center_y"] - 0.2, 0])
            freq_title = Text("采样频率 (bits 3-5)",
                              font=font_main_text, font_size=12, color="#F0F8FF")
            freq_title.next_to(freq_chart, UP, buff=0.1)
            freq_labels = freq_chart.get_bar_labels(
                font_size=11,
                label_constructor=lambda s: Text(str(s), font_size=11),
            )
            self.context.play(
                FadeIn(freq_chart), FadeIn(freq_title),
                FadeIn(freq_labels), run_time=0.6)

        self.context.next_slide()

        # --- key insights ---
        insight1 = Text(
            "HF 确定性 → 永远一个答案 → 丢失 50% 信息",
            font=font_main_text, font_size=14, color="#FF6B6B",
        )
        insight1.move_to([0, space["center_y"] - 1.7, 0])

        insight2 = Text(
            "RBM 从分布采样 → 覆盖多种可能 → 频率逼近真实概率",
            font=font_main_text, font_size=14, color="#90EE90",
        )
        insight2.move_to([0, space["center_y"] - 2.1, 0])

        self.context.play(Write(insight1), run_time=0.5)
        self.context.play(Write(insight2), run_time=0.5)

        core_msg = Text(
            "记忆 = 查找最近的答案 | 理解 = 知道哪些答案是合理的",
            font=font_heading, font_size=18, color="#FFD700",
        )
        core_msg.move_to([0, space["bottom"] + 0.6, 0])
        self.context.play(Write(core_msg), run_time=0.6)
        self.context.next_slide()

        cleanup_list = [
            l3_title, setup, grp_a_label, grp_b_label,
            *data_grps, cue_txt, cue_display,
            hf_header, *hf_rows, hf_verdict,
            rbm_header, insight1, insight2, core_msg,
        ]
        if bar_vals:
            cleanup_list.extend([freq_chart, freq_title, freq_labels])
        self._section_cleanup(*cleanup_list)

    # ---------- L4: XOR ----------

    def _level4_xor(self, space):
        l4_title = Text(
            "Level 4：高阶结构 (XOR) —— RBM 也不是万能的",
            font=font_main_text, font_size=18, color="#87CEEB",
        )
        l4_title.move_to([0, space["top"] - 0.8, 0])
        self.context.play(Write(l4_title), run_time=0.5)

        # --- XOR truth table ---
        xor_explain = Text(
            "XOR 规则: bit5 = bit0 ⊕ bit1",
            font=font_main_text, font_size=16, color="#F0F8FF",
        )
        xor_explain.move_to([0, space["top"] - 1.4, 0])
        self.context.play(Write(xor_explain), run_time=0.4)

        # visual truth table
        table_header = Text(
            "bit0  bit1  →  bit5(预期)",
            font=font_main_text, font_size=14, color="#FFD700",
        )
        table_header.move_to([-4.0, space["top"] - 2.0, 0])
        self.context.play(FadeIn(table_header), run_time=0.2)

        test_cases = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
        truth_rows = []
        for idx, (b0, b1, exp) in enumerate(test_cases):
            y = space["top"] - 2.5 - idx * 0.4
            row_grp = VGroup()
            d0 = self._card(b0)
            d0.move_to([-5.0, y, 0])
            d1 = self._card(b1)
            d1.move_to([-4.5, y, 0])
            arrow = Text("→", font_size=14, color=GRAY)
            arrow.move_to([-4.0, y, 0])
            d_exp = self._card(exp)
            d_exp.move_to([-3.5, y, 0])
            row_grp.add(d0, d1, arrow, d_exp)
            truth_rows.append(row_grp)

        self.context.play(
            *[FadeIn(r) for r in truth_rows], run_time=0.4)
        self.context.next_slide()

        # --- train and test ---
        xor_patterns = get_xor_patterns()
        rbm_xor = RestrictedBoltzmannMachine(
            n_visible=6, n_hidden=36, seed=42)
        rbm_xor.train(xor_patterns, epochs=2000, cd_steps=10)

        hf_xor = StandardHopfieldNetwork(n_neurons=6)
        hf_xor.train(xor_patterns)

        # results display (right side)
        result_header = Text(
            "        HF     RBM",
            font=font_main_text, font_size=14, color="#FFD700",
        )
        result_header.move_to([2.0, space["top"] - 2.0, 0])
        self.context.play(FadeIn(result_header), run_time=0.2)

        hf_correct = 0
        rbm_correct = 0
        result_rows = []
        for idx, (b0, b1, expected) in enumerate(test_cases):
            y = space["top"] - 2.5 - idx * 0.4

            # HF test
            cue = [b0, b1, None, None, None, None]
            hf_result = hf_xor.recall(cue)
            hf_pred = hf_result['final_state'][5]
            hf_ok = hf_pred == expected
            if hf_ok:
                hf_correct += 1

            # RBM test
            rbm_samp = rbm_xor.conditional_sample(
                {0: b0, 1: b1}, [0, 1],
                n_samples=50, gibbs_steps=300)
            rbm_pred = int(np.mean(rbm_samp[:, 5]) > 0.5)
            rbm_ok = rbm_pred == expected
            if rbm_ok:
                rbm_correct += 1

            # visual row
            case_label = Text(f"{b0}⊕{b1}={expected}",
                              font=font_main_text, font_size=12, color="#AAAAAA")
            case_label.move_to([0.5, y, 0])

            hf_mark = Text("✓" if hf_ok else "✗",
                           font_size=18,
                           color="#90EE90" if hf_ok else "#FF6B6B")
            hf_mark.move_to([2.0, y, 0])

            rbm_mark = Text("✓" if rbm_ok else "✗",
                            font_size=18,
                            color="#90EE90" if rbm_ok else "#FF6B6B")
            rbm_mark.move_to([3.2, y, 0])

            row = VGroup(case_label, hf_mark, rbm_mark)
            result_rows.append(row)

        self.context.play(
            *[FadeIn(r) for r in result_rows], run_time=0.5)

        # summary
        summary = VGroup(
            Text(f"HF:  {hf_correct}/4 正确",
                 font=font_main_text, font_size=15,
                 color="#FF6B6B" if hf_correct < 3 else "#90EE90"),
            Text(f"RBM: {rbm_correct}/4 正确",
                 font=font_main_text, font_size=15,
                 color="#FF6B6B" if rbm_correct < 3 else "#90EE90"),
            Text("两者都无法可靠学会 XOR",
                 font=font_main_text, font_size=15, color="#FFD700"),
        ).arrange(DOWN, buff=0.2)
        summary.move_to([0, space["center_y"] - 1.5, 0])
        self.context.play(FadeIn(summary), run_time=0.5)

        why = VGroup(
            Text("XOR 是非线性关系 — 单层隐藏层不够表示",
                 font=font_main_text, font_size=14, color="#AAAAAA"),
            Text("要学会 XOR → 需要更深的网络 → 下一章",
                 font=font_main_text, font_size=15, color="#87CEEB"),
        ).arrange(DOWN, buff=0.2)
        why.move_to([0, space["bottom"] + 0.8, 0])
        self.context.play(FadeIn(why), run_time=0.4)
        self.context.next_slide()

        self._section_cleanup(
            l4_title, xor_explain,
            table_header, *truth_rows,
            result_header, *result_rows,
            summary, why,
        )

    # ==================================================================
    # Part 5 — 总结对比
    # ==================================================================

    def _part5_summary_table(self):
        space = self.get_available_space()

        insight_title = Text(
            "从「记忆」到「理解」", font=font_heading, font_size=28,
            color="#FFD700", weight=BOLD,
        )
        insight_title.move_to([0, space["top"] - 0.2, 0])
        self.context.play(Write(insight_title), run_time=0.7)

        header = Text(
            "           Hopfield   HF+Hidden    Full BM     RBM",
            font=font_main_text, font_size=13, color="#FFD700",
        )

        rows_data = [
            ("训练方式", "Hebbian    无法训练     CD(不收敛)  CD(高效)"),
            ("核心瓶颈", "容量≈0.14N 信用分配     MCMC太慢    单层→低阶"),
            ("L1 分布  ", "✗ 退化     —            △ 噪声     ✓"),
            ("L2 成对  ", "✓          —            ✗ 收敛差    ✓"),
            ("L3 多模态", "✗          —            —           ✓ ★"),
            ("L4 XOR  ", "✗          —            —           ✗"),
        ]

        table_group = VGroup(header)
        for label, values in rows_data:
            row = Text(f"{label}  {values}",
                       font=font_main_text, font_size=12, color="#F0F8FF")
            table_group.add(row)

        table_group.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        table_group.move_to([0, space["center_y"] + 0.3, 0])

        self.context.play(FadeIn(header), run_time=0.4)
        for row_mob in table_group[1:]:
            self.context.play(FadeIn(row_mob), run_time=0.25)
        self.context.next_slide()

        closing = VGroup(
            Text(
                "想要更大的记忆 → 训练不了隐藏层 → 用统计代替梯度",
                font=font_main_text, font_size=15, color="#87CEEB",
            ),
            Text(
                "→ 全连接太慢 → 限制连接 → 获得全新能力：用概率思考",
                font=font_main_text, font_size=15, color="#87CEEB",
            ),
        ).arrange(DOWN, buff=0.15)
        closing.move_to([0, space["bottom"] + 1.6, 0])
        self.context.play(FadeIn(closing), run_time=0.6)

        transition = Text(
            "下一章：概率思考如何通过更深的层次，学会真正复杂的结构？",
            font=font_heading, font_size=18, color="#90EE90",
        )
        transition.move_to([0, space["bottom"] + 0.6, 0])
        self.context.play(Write(transition), run_time=0.7)
        self.context.next_slide()
