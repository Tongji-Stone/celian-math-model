# 参赛论文 LaTeX 工程

本目录为 A 题论文，**不含参赛者姓名、学校与队号**。正文四问已写入；图件在 `paper/figures/`。

## 编译

需要 XeLaTeX + CTeX + BibTeX。在 `paper/` 下执行：

```bat
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

若 `main.pdf` 被占用，可用 `-jobname=paper_preview` 写出预览 PDF。

## 赛制对应

| 要求 | 实现 |
|------|------|
| 第 1 页摘要专用页（含标题、关键词，无英文） | `main.tex` 摘要页 |
| 页码页脚居中，从 1 连续编号 | `fancyhdr` |
| 第 2 页起正文，无目录，正文拟控 30 页 | 未生成 `\tableofcontents` |
| 附录：支撑材料列表、完整源程序、AI 声明 | `sections/app_*.tex` |
| 参考文献规范引用 | `refs.bib` + 正文 `\cite` |

## 正文口径（与 `A/口径.md` 一致）

- **问题一**：策略 9\(\rightarrow\)7、IQR 清洗、典型长寿 \(C_1>C_2\)（约 10.04 min）、典型短寿 \(3.7\mathrm{C}(31\%)-5.9\mathrm{C}\)。
- **问题二**：34 块 / 6 策略；两窗口暴露 + 受限 PySR；`SOH_200` Kruskal \(H=12.3346\)；`Fade_200` \(H=14.79\)。
- **问题三**：近段线性为主（\(k=150\) RMSE \(2.71\times 10^{-4}\)）；融合为结构对照；\(\hat N\) 不进入问题四。
- **问题四**：正式推荐已实验 \(5.3\mathrm{C}(54\%)-4.0\mathrm{C}\)；邻域候选 \(5.3\mathrm{C}(50\%)-4.0\mathrm{C}\)。

## 提交命名（队号确定后）

- 论文 PDF：`XXX_参赛论文.pdf`（由 `main.tex` 编译，勿压缩）
- 支撑材料：`XXX_支撑材料.zip`（见附录文件列表，≤20MB；不含 MATR `raw/` 与融合 `weights/`）
