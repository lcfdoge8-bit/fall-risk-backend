"""
Data Generation Script — Elderly Fall Risk Prediction
=====================================================
Group 3: Predictive AI for Elderly Fall Risk Assessment（养老院场景）
第一步：基于 Data Generation Blueprint 生成 2,000 份合成患者档案

字段设计（10 输入 + 2 目标，对应蓝图）：
  核心参数:  age, night_bed_exits, night_activity_duration_min, past_falls,
             mobility_score, high_risk_medication
  补充参数:  cognitive_impairment, polypharmacy_count, orthostatic_hypotension,
             tug_seconds
  目标变量:  fall_risk_score (0-1), fall_risk_level (LOW/MEDIUM/HIGH)

文献依据：[R2]新加坡指南 [R3]Ganz [R5]Woolcott [R6]BMC TUG [R7]USPSTF
          [R8]J.Urology [R9]Vaughan [R10]Tinetti [R11]NHS [R12]离床监测

用法：python generate_fall_risk_data.py --n 2000 --seed 42 --output fall_risk_patients_2000.csv
"""

import argparse
import numpy as np
import pandas as pd
#For line 125
import random
from faker import Faker


def generate_fall_risk_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fake = Faker("zh_CN")          # 中文姓名/地址，增强拟真
    Faker.seed(seed)
    np.random.seed(seed)

    # ================= 补充参数先行（用于驱动核心参数相关性） =================
    # age: 右偏分布，集中 75-90（养老院人群）[R2][R3]
    age_probe = np.clip(rng.normal(80, 7, n).astype(int), 60, 100)

    # cognitive_impairment: 0/1/2（无/轻/中重度），随 age 递增 [R3: LR 4.2/17]
    cognitive = rng.choice([0, 1, 2], n, p=[0.50, 0.30, 0.20])
    cognitive = np.where(age_probe >= 85,
                         np.clip(cognitive + rng.choice([0, 1], n, p=[0.5, 0.5]),
                                 0, 2),
                         cognitive)

    # polypharmacy_count: 0-15 种，均值≈5（养老院常服 5-8 种）[R5]
    poly_count = np.clip(rng.normal(5, 3, n).astype(int), 0, 15)

    # high_risk_medication: 与多重用药正相关 [R5: 抗抑郁 OR1.68 等]
    high_risk_med = (poly_count > 4) | (rng.random(n) < 0.15)

    # orthostatic_hypotension: 随高风险用药（降压/利尿剂）上调 [R11]
    ortho = rng.random(n) < (0.25 + 0.15 * high_risk_med)

    age = age_probe

    # ================= 核心参数采样（边际分布 + 相关性） =================
    # night_bed_exits: 混合泊松；认知障碍者夜间徘徊概率上升 [R8][R9][R12]
    mix = rng.random(n)
    exits = np.where(
        (mix < 0.60) & (cognitive == 0), rng.poisson(0.8, n),
        np.where((mix < 0.90) | (cognitive >= 1), rng.poisson(2.5, n),
                 rng.poisson(5.0, n)))
    exits = np.clip(exits, 0, 8)

    # night_activity_duration_min: 离床次数 × 单次时长 + 噪声 [R12]
    duration = (exits * rng.uniform(5, 20, n) + rng.uniform(0, 15, n)).astype(int)
    duration = np.clip(duration, 0, 120)

    # past_falls: 零膨胀（65% 为 0；≥2 次为复发性跌倒）[R3][R7][R11]
    u2 = rng.random(n)
    falls = np.where(u2 < 0.65, 0,
                     np.where(u2 < 0.85, 1, rng.poisson(2.0, n)))
    falls = np.clip(falls, 0, 5)

    # tug_seconds: 随 age 递增（60 岁≈11s → 100 岁≈20s）[R6: ≥13.5s 高风险]
    tug_mean = 11 + (age - 60) * 0.22
    tug = np.clip(rng.normal(tug_mean, 5, n), 8, 40)

    # mobility_score: 由 TUG 映射（1-10 分，越低越差）[R6][R7]
    mobility = np.where(tug < 13.5, rng.integers(8, 11, n),
                        np.where(tug <= 20, rng.integers(5, 8, n),
                                 rng.integers(1, 5, n)))
    mobility = np.clip(mobility, 1, 10)

    # ================= 风险点计数 → 目标变量 =================
    # Tinetti 叠加规律 [R10]：0 因素 8% → ≥4 因素 78%
    points = (age >= 80).astype(int)
    points += np.where(falls >= 1, 1, 0) + np.where(falls >= 2, 1, 0)
    points += np.where(exits >= 2, 1, 0) + np.where(exits >= 5, 1, 0)
    points += np.where(duration >= 30, 1, 0) + np.where(duration >= 60, 1, 0)
    points += np.where(mobility <= 5, 1, 0) + np.where(mobility <= 3, 1, 0)
    points += high_risk_med.astype(int)
    points += np.where(cognitive >= 1, 1, 0) + np.where(cognitive == 2, 1, 0)
    points += (poly_count > 4).astype(int)
    points += ortho.astype(int)

    # 校准公式：0 分 → 0.12（LOW 用例），9 分 → 0.89（HIGH 用例）
    score = np.clip(0.12 + 0.0856 * points + rng.normal(0, 0.03, n), 0, 1)
    level = np.where(score < 0.33, "LOW",
                     np.where(score < 0.66, "MEDIUM", "HIGH"))

    # ================= 拟真身份字段（Faker） =================
    df = pd.DataFrame({
        "patient_id":      [f"P{20260000 + i:05d}" for i in range(n)],
        "name":            [fake.name() for _ in range(n)],
        "age":             age,
        "night_bed_exits": exits,
        "night_activity_duration_min": duration,
        "past_falls":      falls,
        "mobility_score":  mobility,
        "high_risk_medication": high_risk_med,
        "cognitive_impairment": cognitive,
        "polypharmacy_count":    poly_count,
        "orthostatic_hypotension": ortho,
        "tug_seconds":     np.round(tug, 1),
        "fall_risk_score": np.round(score, 3),
        "fall_risk_level": level,
    })
    return df


def main():
    parser = argparse.ArgumentParser(description="生成老年人跌倒风险合成数据")
    parser.add_argument("--n", type=int, default=2000, help="样本量")
    parser.add_argument("--seed", type=int, default=random.randint(60, 100), help="随机种子") # [ changed default=42 to default=random.randint(60, 100) ] Because of the web dashboard need to generate random mock patient profiles.
    parser.add_argument("--output", type=str,
                        default="fall_risk_patients_2000.csv", help="输出 CSV 路径")
    args = parser.parse_args()

    df = generate_fall_risk_data(args.n, args.seed)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"✅ 已生成 {len(df)} 条患者档案 → {args.output}")
    print(f"级别分布: {df['fall_risk_level'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"risk_score: 均值={df['fall_risk_score'].mean():.3f}, "
          f"范围=[{df['fall_risk_score'].min():.2f}, {df['fall_risk_score'].max():.2f}]")


if __name__ == "__main__":
    main()
