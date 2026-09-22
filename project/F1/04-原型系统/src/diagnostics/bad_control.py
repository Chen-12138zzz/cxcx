# -*- coding: utf-8 -*-
"""
F1 · 坏控制 / 中介变量检查

【为什么这个模块在养殖场景里最关键】
因果推断最常见的致命错误不是"少调整了变量"，而是**多调整了不该调整的变量**。
多调整不会报错、不会让模型指标变差、甚至会让系数看起来更"显著"，
但**它估的已经不是你想估的那个量了**。

三类必须区分的变量（本项目养殖场景，对应核心算法设计说明第 2.2 节）：

    混杂（confounder）：同时影响 T 与 Y，且**不是** T 的结果 → ★ 必须调整
       例：塘面积、水深、塘型、气温
    中介（mediator）  ：受 T 影响，又影响 Y           → ★ 绝不能调整（若要估总效应）
       例：**水质退化指数**（投喂 ↑ → 有机物 ↑ → 溶氧 ↓ → 生长 ↓）
    碰撞（collider）  ：同时受 T 与 Y（或其未测原因）影响 → ★ 绝不能调整
       例：**FCR（饲料转化率）** = 饲料量 / 增重，分子含 T、分母含 Y

调整中介 → 估的是**直接效应**，把"通过水质起作用"的那部分效应屏蔽掉了。
            报告若把直接效应当总效应 → 系统性低估（本项目合成数据实测见下）。
调整碰撞 → 打开 T 与 Y 之间的后门，制造**本来不存在的相关**，
            方向不可预测，可能比不调整更糟。

【诊断方式（不依赖结构方程，只用可检验的操作性判据）】
对每个候选调整变量 Z，做三件事：

  ① **时序/逻辑三重检查**（写入台账，人工确认，不可自动化）
     Q1 Z 在 T 之前就被决定吗？（是 → 可能混杂）
     Q2 Z 是 T 的结果吗？（是 → 中介，检查失败）
     Q3 Z 是否由 T 与 Y 共同决定，或由 Y 的原因决定？（是 → 碰撞，检查失败）

  ② **数据驱动的警示信号**（自动，用于提示人工复核，不作为判定依据）
     - Z 与 T 的关联强度：Z ~ T + X 中 T 的偏 R²；若很高 → 可疑（可能是中介）
     - Z 与 Y 的关联强度：同上
     - Z 被加入调整集后效应的位移量：Δθ = θ(含Z) − θ(不含Z)
       · Δθ 显著为负且趋近"直接效应" → 强烈提示 Z 是中介
       · Δθ 方向不定、数值巨大        → 提示 Z 是碰撞（放大噪声）

  ③ **两组对照估计**（本项目实际做法）
     估三次：不含 Z（= 主结果）／含 Z（= 被污染结果）／
     以及不用 Z 但用 Z 的"前因"（若可得，用于验证 Z 确实受 T 影响）
     三者差异写进验证报告，**让读者自己看到"多调整一列"的代价**。

【关键纪律（红线 K-6）】
本项目**禁止**在报告中把"含中介的调整模型"结果当作总效应报告。
若确实要报告直接效应，标题与正文必须明确写出"直接效应（条件于水质）"，
并同时给出总效应，不得只报其一。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# 已知的本项目变量角色（写在代码里，防止被静默改动）
KNOWN_ROLES: dict[str, str] = {
    "pond_area": "confounder",
    "pond_depth": "confounder",
    "pond_type": "confounder",
    "season_temp": "confounder",
    "water_degrad": "mediator",     # ★ 投喂影响水质，水质影响生长
    "fcr": "collider",              # ★ 分子含投喂量、分母含增重
    "shrimp_appetite": "unmeasured_confounder",   # ★ 不可观测，仅用于诊断评估
}

# 各角色对调整集的处置规则（代码化，不靠人记）
ROLE_POLICY: dict[str, str] = {
    "confounder": "must_adjust",
    "mediator": "must_not_adjust",           # 若要估总效应
    "collider": "must_not_adjust",
    "unmeasured_confounder": "cannot_adjust",  # 观测不到，只能做敏感性分析
}


@dataclass
class BadControlDiagnosis:
    """坏控制检查结果。"""
    candidate: str
    role: str                     # 台账中登记的角色
    policy: str                   # 应如何处置
    in_adjustment_set: bool       # 当前是否被放进调整集
    compliant: bool               # 是否符合 policy
    n_in_adjustment_set: int
    partial_r2_with_t: float      # Z ~ T + X 中 T 的偏 R²
    partial_r2_with_y: float      # Z ~ Y + X 中 Y 的偏 R²
    theta_without: float | None   # 不含 Z 的效应估计
    theta_with: float | None      # 含 Z 的效应估计
    delta_theta: float | None     # 位移量（含 − 不含）
    warning: str = ""
    notes: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "变量": self.candidate,
            "登记角色": self.role,
            "处置规则": self.policy,
            "是否在调整集": "是" if self.in_adjustment_set else "否",
            "合规": "✓" if self.compliant else "✗ 违规",
            "偏R²(T)": (f"{self.partial_r2_with_t:.4f}"
                        if np.isfinite(self.partial_r2_with_t) else "—"),
            "偏R²(Y)": (f"{self.partial_r2_with_y:.4f}"
                        if np.isfinite(self.partial_r2_with_y) else "—"),
            "θ(不含Z)": ("—" if self.theta_without is None
                         else f"{self.theta_without:+.4f}"),
            "θ(含Z)": ("—" if self.theta_with is None else f"{self.theta_with:+.4f}"),
            "Δθ": ("—" if self.delta_theta is None else f"{self.delta_theta:+.4f}"),
            "警示": self.warning,
        }


def _partial_r2(z: np.ndarray, target: np.ndarray, X: np.ndarray) -> float:
    """在控制 X 后，Z 对 target 的偏 R²（用线性回归近似，仅作警示信号）。

    做法：target ~ X 与 target ~ X + Z 两个回归的 R² 之差。
    ⚠️ 这是**线性**近似：真实关系非线性时该值可能低估关联强度。
       它只用于提示人工复核，**不作为判定 Z 角色的依据**。
    """
    if X.size == 0:
        A0 = np.ones((len(target), 1))
    else:
        A0 = np.column_stack([np.ones(len(target)), X])
    A1 = np.column_stack([A0, z])

    def r2(A):
        beta, *_ = np.linalg.lstsq(A, target, rcond=None)
        res = target - A @ beta
        ss_res = float(res @ res)
        ss_tot = float(np.sum((target - target.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return float(r2(A1) - r2(A0))


def check_bad_control(df: pd.DataFrame,
                      t_col: str = "feed_rate",
                      y_col: str = "growth",
                      covariates: list[str] | None = None,
                      candidates: list[str] | None = None,
                      roles: dict[str, str] | None = None,
                      run_contrast: bool = True,
                      seed: int = 20260920) -> list[BadControlDiagnosis]:
    """对候选调整变量逐一做坏控制检查。

    参数
    ----
    covariates : **基线**调整集（应只含混杂）
    candidates : 要检查的变量（默认检查 KNOWN_ROLES 中所有非混杂变量 + 基线里出现的）
    roles      : 角色台账（默认 KNOWN_ROLES）；未登记的变量标为 "unknown"
    run_contrast : 是否实际跑"含/不含 Z"两次估计（较慢，但给出最有说服力的 Δθ）
    """
    if covariates is None:
        from synth import available_covariates
        covariates = available_covariates()
    if roles is None:
        roles = KNOWN_ROLES
    if candidates is None:
        candidates = [c for c in df.columns
                      if c in roles and roles[c] != "confounder"
                      and not c.startswith("_")]

    X = df[covariates].to_numpy(dtype=float)
    T = df[t_col].to_numpy(dtype=float)
    Y = df[y_col].to_numpy(dtype=float)

    out: list[BadControlDiagnosis] = []

    for z_name in candidates:
        z = df[z_name].to_numpy(dtype=float)
        role = roles.get(z_name, "unknown")
        policy = ROLE_POLICY.get(role, "undetermined")
        in_set = z_name in covariates
        compliant = (
            (policy == "must_adjust" and in_set)
            or (policy == "must_not_adjust" and not in_set)
            or (policy == "cannot_adjust" and not in_set)
        )

        pr_t = _partial_r2(z, T, X)
        pr_y = _partial_r2(z, Y, X)

        theta_wo = theta_w = delta = None
        warn = ""
        notes: list[str] = []

        if run_contrast:
            try:
                from estimators import estimate_ate_continuous
                r0 = estimate_ate_continuous(df, t_col, y_col, covariates,
                                             seed=seed, engine="manual")
                theta_wo = r0.point
                r1 = estimate_ate_continuous(df, t_col, y_col, covariates + [z_name],
                                             seed=seed, engine="manual")
                theta_w = r1.point
                delta = theta_w - theta_wo
            except Exception as e:
                notes.append(f"对照估计失败：{type(e).__name__}: {e}")

        # ---- 警示逻辑：代码化，避免事后自由裁量 ----
        if policy == "must_not_adjust" and in_set:
            warn = f"★ 违规：{role} 不应进入调整集"
        elif policy == "must_not_adjust" and delta is not None:
            if role == "mediator" and abs(delta) > 0.02:
                warn = (f"量级警示：加入后效应位移 {delta:+.4f}，"
                        f"与『中介会屏蔽部分效应』的预期一致；"
                        f"须确认报告的到底是总效应还是直接效应（红线 K-6）")
            elif role == "collider" and abs(delta) > 0.02:
                warn = (f"量级警示：加入后效应位移 {delta:+.4f}，"
                        f"碰撞会打开后门通路，位移方向与大小不可预期")
        if policy == "must_adjust" and not in_set:
            warn = "★ 违规：混杂未进入调整集（遗漏混杂 = 偏倚）"
        if policy == "undetermined":
            warn = "角色未登记，须人工确认时序（Q1/Q2/Q3）后再决定是否调整"

        # 数据信号提示（不覆盖上面的规则判定）
        if pr_t > 0.1:
            notes.append(f"Z 与 T 的偏 R²={pr_t:.4f} 较高，提示 Z 可能受 T 影响"
                         f"（互为因果或中介），须核对时序")
        if pr_y > 0.1:
            notes.append(f"Z 与 Y 的偏 R²={pr_y:.4f} 较高，提示 Z 可能是 Y 的近端决定因素"
                         f"（若同时受 T 影响则为中介/碰撞）")

        out.append(BadControlDiagnosis(
            candidate=z_name, role=role, policy=policy,
            in_adjustment_set=in_set, compliant=compliant,
            n_in_adjustment_set=len(covariates),
            partial_r2_with_t=pr_t, partial_r2_with_y=pr_y,
            theta_without=theta_wo, theta_with=theta_w, delta_theta=delta,
            warning=warn, notes=notes,
        ))

    return out


def audit_adjustment_set(covariates: list[str],
                         roles: dict[str, str] | None = None) -> dict:
    """静态审计一个调整集：逐项对台账，返回违规清单。

    这是**不需要数据**就能跑的检查，应在每次实验前先跑一遍，
    防止实验脚本里静默多加了 water_degrad 或 fcr。
    """
    if roles is None:
        roles = KNOWN_ROLES
    violations, unknown, ok = [], [], []
    for c in covariates:
        role = roles.get(c, "unknown")
        policy = ROLE_POLICY.get(role, "undetermined")
        if policy == "must_adjust":
            ok.append(c)
        elif policy in ("must_not_adjust", "cannot_adjust"):
            violations.append({"变量": c, "角色": role, "原因": f"{role} 不应进入调整集"})
        else:
            unknown.append(c)
    return {
        "adjustment_set": list(covariates),
        "n": len(covariates),
        "ok": ok,
        "violations": violations,
        "unknown_role": unknown,
        "pass": len(violations) == 0,
        "note": ("静态审计只核对角色台账；台账本身的正确性需要人工按 Q1/Q2/Q3 确认，"
                 "审计工具不能替代这一确认"),
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import synth

    print("=" * 96)
    print("坏控制检查冒烟测试")
    print("=" * 96)

    # ---- ① 静态审计：三种调整集 ----
    print("\n【静态审计】")
    cases = {
        "正确（仅混杂）": ["pond_area", "pond_depth", "pond_type", "season_temp"],
        "错误（多加了中介）": ["pond_area", "pond_depth", "pond_type",
                              "season_temp", "water_degrad"],
        "错误（多加了碰撞）": ["pond_area", "pond_depth", "pond_type",
                              "season_temp", "fcr"],
        "错误（两者都加）": ["pond_area", "pond_depth", "pond_type",
                            "season_temp", "water_degrad", "fcr"],
    }
    for name, cs in cases.items():
        a = audit_adjustment_set(cs)
        flag = "通过" if a["pass"] else f"**不通过（{len(a['violations'])} 项违规）**"
        print(f"  {name:20s} {flag}")
        for v in a["violations"]:
            print(f"      · {v['变量']}（{v['角色']}）：{v['原因']}")

    # ---- ② 动态对照：加入中介/碰撞后效应位移多少 ----
    print("\n【动态对照】含/不含 Z 的效应位移（u_strength=0.5，n=4000）")
    df = synth.generate(n=4000, u_strength=0.5, nonlinear=True, seed=3)
    base = ["pond_area", "pond_depth", "pond_type", "season_temp"]
    diags = check_bad_control(df, covariates=base, candidates=["water_degrad", "fcr"])
    print(f"  真值（斜率口径，仅参考）：", end="")
    print(f"{float(df['_true_ate'].iloc[0]) / (df.attrs['treat_high_mean'] - df.attrs['treat_low_mean']):+.4f}")
    for d in diags:
        print(f"\n  · {d.candidate}（{d.role}）")
        print(f"      θ(不含Z) = {d.theta_without:+.4f}")
        print(f"      θ(含Z)   = {d.theta_with:+.4f}")
        print(f"      Δθ       = {d.delta_theta:+.4f}")
        print(f"      偏R²(T) = {d.partial_r2_with_t:.4f}   偏R²(Y) = {d.partial_r2_with_y:.4f}")
        if d.warning:
            print(f"      {d.warning}")
        for n in d.notes:
            print(f"      · {n}")
