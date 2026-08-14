# 参赛论文 LaTeX 工程（半成品）

本目录为 A 题论文半成品，**不含参赛者姓名、学校与队号**。

## 编译

需要 XeLaTeX + CTeX + BibTeX。在 `paper/` 下执行：

```bat
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

## 赛制对应

| 要求 | 实现 |
|------|------|
| 第 1 页摘要专用页（含标题、关键词，无英文） | `main.tex` 摘要页 |
| 页码页脚居中，从 1 连续编号 | `fancyhdr` |
| 第 2 页起正文，无目录，正文拟控 30 页 | 未生成 `\tableofcontents` |
| 附录：支撑材料列表、完整源程序、AI 声明 | `sections/app_*.tex` |
| 参考文献规范引用 | `refs.bib` + 正文 `\cite` |

## 完成度

- **问题一**：已按第三次迭代写入——策略 9\(\rightarrow\)7、IQR 清洗、SOH 总图/分布/箱线、典型长/短寿命及 \(C_1>C_2\) 解释。
- **问题二**：已完成策略差异检验、参数关联/回归诊断与混合效应轨迹分析；结果在 `output/A/question2/`。
- **问题三--四**：仅框架、公式与待填结果框。
- 源程序通过 `\lstinputlisting` 引用 `../src/A/`。
- 最新编译 PDF：优先打开 `paper_preview.pdf`（若 `main.pdf` 正被占用可能未刷新）。

## 提交命名（队号确定后）

- 论文 PDF：`XXX_参赛论文.pdf`（由 `main.tex` 编译，勿压缩）
- 支撑材料：`XXX_支撑材料.zip`（见附录文件列表，≤20MB）
