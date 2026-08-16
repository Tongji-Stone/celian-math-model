"""绘制指定电池的 SOH 曲线。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_chinese_font() -> str:
    """选择系统中可用的中文字体，并返回字体名称。"""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    selected = next((name for name in candidates if name in installed), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return selected


def plot_battery_soh(
    cycles: pd.DataFrame,
    battery_id: int,
    output_path: Path,
) -> None:
    """绘制指定电池的平滑 SOH 曲线。"""
    required = {"battery_id", "cycle", "SOH_smooth"}
    missing = required.difference(cycles.columns)
    if missing:
        raise ValueError(f"循环数据缺少字段: {sorted(missing)}")

    battery = cycles.loc[cycles["battery_id"] == battery_id].sort_values("cycle")
    if battery.empty:
        raise ValueError(f"未找到 battery_id={battery_id} 的数据")

    configure_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        battery["cycle"],
        battery["SOH_smooth"],
        color="#4C78A8",
        linewidth=2.2,
        label="平滑 SOH（SOH_smooth）",
    )
    ax.set_title(f"{battery_id} 号电池 SOH 曲线")
    ax.set_xlabel("循环次数")
    ax.set_ylabel("电池健康状态（SOH）")
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all_battery_soh(cycles: pd.DataFrame, output_path: Path) -> None:
    """用蓝到红渐变绘制全部电池的平滑 SOH 曲线。"""
    required = {"battery_id", "cycle", "SOH_smooth"}
    missing = required.difference(cycles.columns)
    if missing:
        raise ValueError(f"循环数据缺少字段: {sorted(missing)}")

    battery_ids = sorted(int(value) for value in cycles["battery_id"].unique())
    if not battery_ids:
        raise ValueError("循环数据中没有电池记录")

    configure_chinese_font()
    fig, ax = plt.subplots(figsize=(14, 8))
    colormap = matplotlib.colormaps.get_cmap("coolwarm")
    colors = colormap(np.linspace(0, 1, len(battery_ids)))
    for color, battery_id in zip(colors, battery_ids):
        battery = cycles.loc[cycles["battery_id"] == battery_id].sort_values("cycle")
        ax.plot(
            battery["cycle"],
            battery["SOH_smooth"],
            color=color,
            linewidth=1.05,
            alpha=0.78,
        )

    ax.set_title("全部电池平滑 SOH 曲线")
    ax.set_xlabel("循环次数")
    ax.set_ylabel("电池健康状态（SOH_smooth）")
    ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.6)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=colors[0], linewidth=2, label=f"低编号：电池 {battery_ids[0]}"),
            plt.Line2D([0], [0], color=colors[-1], linewidth=2, label=f"高编号：电池 {battery_ids[-1]}"),
        ],
        loc="upper right",
        title="颜色由蓝到红",
    )
    scalar_mappable = plt.cm.ScalarMappable(
        cmap=colormap,
        norm=plt.Normalize(vmin=battery_ids[0], vmax=battery_ids[-1]),
    )
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.015)
    colorbar.set_label("battery_id")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制指定电池的 SOH 曲线")
    parser.add_argument("--battery-id", type=int, default=1, help="电池编号，默认为 1")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "output" / "A" / "cycle_train_cleaned.csv",
        help="循环数据 CSV 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "A" / "SOH_1_example.png",
        help="输出图像路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cycles = pd.read_csv(args.input)
    plot_battery_soh(cycles, args.battery_id, args.output)
    print(f"已生成: {args.output}")


if __name__ == "__main__":
    main()
