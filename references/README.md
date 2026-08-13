# 参考文献库（A题：锂离子电池快充策略对寿命衰减的影响建模与优化）

本文件夹汇总赛题相关的**论文、公开数据集与书籍**，并给出每条文献的简要说明与适用问题。

> 说明：受版权限制，此处**不存放论文 PDF 全文**，仅提供规范引用信息、链接与使用指引。请通过学校图书馆、DOI 或开放获取链接自行下载。

---

## 目录结构

```text
references/
├── README.md              # 本说明（分类导读 + 每条文献简介）
├── references.bib         # BibTeX，可直接用于 LaTeX / Zotero
└── reading-priority.md    # 建议阅读顺序（按时间紧张程度）
```

---

## 快速索引：按问题找文献

| 问题 | 优先阅读 |
|------|----------|
| 总纲 / 数据集来源 | Severson 2019；Attia 2020 |
| 问题1 探索分析与机制解释 | Liu 2019；Dubarry 2009/2012；Ahmed 2017 |
| 问题2 策略参数影响 | Ruiz 2023；Wang 2024；Geslin 2023；Ma 2024 |
| 问题3 寿命预测 | Severson 2019；Attia 2021；Geslin 2023 |
| 问题4 充电时间–寿命优化 | Attia 2020；Plett BMS Vol.III；JES 2024 fast-charging |
| 统计 / 优化方法 | Kutner；Deb；Frazier 2018；Pinheiro & Bates |

---

## 一、必读核心论文（与 MIT–Stanford 数据直接相关）

### 1. Severson et al., *Nature Energy*, 2019

- **题名**：Data-driven prediction of battery cycle life before capacity degradation  
- **DOI**：[10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8)  
- **开放 PDF**：[MIT Braatz 组镜像](https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf)  
- **数据集**：[https://data.matr.io/1/](https://data.matr.io/1/)  
- **简要说明**：本题数据源头论文。124 块 A123 LFP/石墨 18650 在不同快充协议下循环，寿命约 150–2300 圈；用早期循环特征（尤其放电电压曲线相关特征）在容量尚未明显衰减时预测寿命。  
- **对本题的帮助**：问题1–4总纲；理解两阶段快充 `C1-Q1%-C2`、EOL=80% SOH、早期预测思路；写数据来源与相关工作时必引。

### 2. Attia et al., *Nature*, 2020

- **题名**：Closed-loop optimization of fast-charging protocols for batteries with machine learning  
- **DOI**：[10.1038/s41586-020-1994-5](https://doi.org/10.1038/s41586-020-1994-5)  
- **简要说明**：在早期寿命预测模型之上，用贝叶斯优化在大量候选快充协议中高效搜索长寿命协议；展示“预测 + 优化”闭环。  
- **对本题的帮助**：**问题4** 的直接范本；也可支撑问题3“早期数据预测寿命”的合理性。

### 3. Attia et al., *Journal of The Electrochemical Society*, 2021

- **题名**：Statistical Learning for Accurate and Interpretable Battery Lifetime Prediction  
- **DOI**：[10.1149/1945-7111/ac2704](https://doi.org/10.1149/1945-7111/ac2704)  
- **简要说明**：强调可解释统计学习与特征工程（如 \(\Delta Q(V)\)）；讨论为何简单线性模型在该数据上仍很强。  
- **对本题的帮助**：**问题3** 特征提取、模型选择与“可解释性”写作。

### 4. Geslin et al., *Joule*, 2023

- **题名**：相关评论/分析：充电条件特征在寿命预测中的作用（Braatz 组相关工作）  
- **PDF**：[Geslin_Joule_2023.pdf](http://web.mit.edu/braatzgroup/Geslin_Joule_2023.pdf)  
- **简要说明**：指出在 Severson 类数据中，编码充电条件的特征会抬升跨协议预测精度，但也影响模型可迁移性解释。  
- **对本题的帮助**：问题2/3 中是否把 `C1,Q1,C2` 纳入预测特征的消融讨论；解释“策略信息 vs 早期衰减特征”。

---

## 二、问题1 / 机制解释类论文

### 5. Liu, Zhu & Cui, *Nature Energy*, 2019

- **题名**：Challenges and opportunities towards fast-charging battery materials  
- **DOI**：[10.1038/s41560-019-0405-3](https://doi.org/10.1038/s41560-019-0405-3)  
- **开放 PDF**：[Stanford Cui 组](https://web.stanford.edu/group/cui_group/papers/Yayuan_Cui_NATENG_2019.pdf)  
- **简要说明**：快充材料与失效机理综述：传质、电荷转移、热管理；强调高倍率下锂沉积与安全风险。  
- **对本题的帮助**：问题1“长/短寿命策略差异”的物理机制表述；高 SOC 区间高倍率更伤电池的文献支撑。

### 6. Dubarry & Liaw, *Journal of Power Sources*, 2009

- **题名**：Identify capacity fading mechanism in a commercial LiFePO4 cell  
- **DOI**：[10.1016/j.jpowsour.2009.05.036](https://doi.org/10.1016/j.jpowsour.2009.05.036)  
- **简要说明**：商业 LFP 电芯容量衰减机理识别；与 NMC 等体系老化路径不同。  
- **对本题的帮助**：本题为 LFP，机制讨论应引用 LFP 文献，避免套用三元电池结论。

### 7. Dubarry, Truchot & Liaw, *Journal of Power Sources*, 2012

- **题名**：Synthesize battery degradation modes via a diagnostic and prognostic model  
- **DOI**：[10.1016/j.jpowsour.2012.07.016](https://doi.org/10.1016/j.jpowsour.2012.07.016)  
- **相关工具**：[‘Alawa 介绍页](https://www.hnei.hawaii.edu/alawa/)  
- **简要说明**：提出用数字孪生思路合成/诊断退化模式（如 LLI、LAM 等），支撑 SOH 诊断与预后。  
- **对本题的帮助**：问题1/2 把“容量掉了”翻译成“可能的退化模式”；机制章节框架。

### 8. Ahmed et al., *Journal of Power Sources*, 2017

- **题名**：Enabling fast charging – A battery technology gap assessment  
- **DOI**：[10.1016/j.jpowsour.2017.06.055](https://doi.org/10.1016/j.jpowsour.2017.06.055)  
- **简要说明**：美国 DOE 视角的快充技术缺口评估：倍率、温控、电极与系统层约束。  
- **对本题的帮助**：问题1/2 背景与“为何要研究快充–寿命权衡”。

---

## 三、问题2：参数影响、老化建模与多段快充

### 9. Ruiz et al., 退化机理综述（TU Delft）, 2023

- **题名**：Physics-Based and Data-Driven Modeling of Degradation Mechanisms for Lithium-Ion Batteries: A Review  
- **PDF**：[TU Delft Pure](https://pure.tudelft.nl/ws/portalfiles/portal/237769237/Physics-Based_and_Data-Driven_Modeling_of_Degradation_Mechanisms_for_Lithium-Ion_BatteriesA_Review.pdf)  
- **简要说明**：物理模型与数据驱动模型对照；SEI、锂沉积、knee-point 等。  
- **对本题的帮助**：问题2 机制讨论与模型分类综述引用。

### 10. Wang et al., *J. Electrochem. Soc.*, 2024

- **题名**：High Accuracy and Applicability Battery Aging Models for Electric Vehicle Applications  
- **DOI**：[10.1149/1945-7111/ada73e](https://doi.org/10.1149/1945-7111/ada73e)  
- **简要说明**：半经验老化模型（Arrhenius、幂律 \(t^z\) 等）及其适用性讨论。  
- **对本题的帮助**：问题2 建立 `C1/Q1/C2`–寿命回归/半经验式时的公式参考。

### 11. Li et al., *Entropy*, 2026

- **题名**：Review of State-of-the-Art Degradation Models for Lithium-Ion Batteries  
- **DOI**：[10.3390/e28060669](https://www.mdpi.com/1099-4300/28/6/669)  
- **简要说明**：较新的退化模型综述，覆盖 SEI、锂沉积、颗粒开裂与热耦合。  
- **对本题的帮助**：相关工作与局限性讨论的较新引用。

### 12. Ma et al., *Batteries*, 2024

- **题名**：An Aging-Optimized State-of-Charge-Controlled Multi-Stage Constant Current (MCC) Fast Charging Algorithm...  
- **DOI**：[10.3390/batteries10080267](https://www.mdpi.com/2313-0105/10/8/267)  
- **简要说明**：按 SOC 分段的多段恒流快充，在缩短时间同时尽量避免锂沉积。  
- **对本题的帮助**：与赛题**两阶段快充**形式高度一致；问题2/4 策略设计对照。

### 13. *J. Electrochem. Soc.*, 2024（快充 + 退化 + 单体差异）

- **题名**：Fast Charging of Lithium-Ion Batteries While Accounting for Degradation and Cell-to-Cell Variability  
- **DOI**：[10.1149/1945-7111/ad76dd](https://iopscience.iop.org/article/10.1149/1945-7111/ad76dd)  
- **简要说明**：在快充协议设计中显式考虑退化与 cell-to-cell 不确定性。  
- **对本题的帮助**：问题2“同策略寿命分散”、问题4“约束 C-rate/电压”的写法。

---

## 四、问题3：寿命 / SOH / RUL 预测补充

> 核心仍以 Severson 2019、Attia 2021、Geslin 2023 为主。以下用于扩展方法对比。

### 14. 各类 Battery RUL / SOH 综述（检索入口）

- **建议检索词**：`battery remaining useful life prediction review`、`SOH estimation lithium-ion review`  
- **简要说明**：用于对比 LSTM、GPR、XGBoost、粒子滤波等常见 RUL 路线。  
- **对本题的帮助**：问题3“模型选择与精度评价”的相关工作段落；不必全部精读，选 1–2 篇综述即可。

---

## 五、问题4：优化与贝叶斯优化

### 15. Frazier, 2018

- **题名**：A Tutorial on Bayesian Optimization  
- **常见出处**：arXiv:1807.02811  
- **链接**：[https://arxiv.org/abs/1807.02811](https://arxiv.org/abs/1807.02811)  
- **简要说明**：贝叶斯优化入门教程：采集函数、探索–利用权衡。  
- **对本题的帮助**：若问题4采用贝叶斯优化（对齐 Attia 2020），可作方法引用。

### 16. Deb, *Multi-Objective Optimization Using Evolutionary Algorithms*

- **简要说明**：NSGA-II 等多目标进化算法经典教材。  
- **对本题的帮助**：问题4 Pareto 前沿（充电时间 vs 寿命）求解方法。

---

## 六、公开数据集

### 17. MIT–Stanford–TRI Battery Dataset（本题主数据）

- **入口**：[https://data.matr.io/1/](https://data.matr.io/1/)  
- **对应论文**：Severson et al., 2019  
- **简要说明**：完整循环曲线、电压–容量等原始信息；本题给出的 Excel 是其整理子集。  
- **对本题的帮助**：需要 \(\Delta Q(V)\) 等高级特征时可回源数据；必须在论文中声明数据来源。

### 18. CALCE Battery Data

- **入口**：[https://calce.umd.edu/data](https://calce.umd.edu/data)  
- **简要说明**：马里兰大学 CALCE 多种化学体系与工况的公开老化数据。  
- **对本题的帮助**：方法泛化讨论、对比验证（非必须）。

### 19. Oxford Battery Intelligence Lab Datasets

- **入口**：[https://battery-intelligence-lab.github.io/data-and-code/](https://battery-intelligence-lab.github.io/data-and-code/)  
- **简要说明**：Oxford Battery Degradation Dataset、路径依赖老化等。  
- **对本题的帮助**：SOC/路径依赖相关扩展讨论。

### 20. NASA PCoE Battery Dataset

- **入口**：NASA Prognostics Data Repository（搜索 `NASA battery dataset PCoE`）  
- **简要说明**：经典 RUL/SOH 基准数据。  
- **对本题的帮助**：问题3方法 benchmark 对照。

### 21. dos Reis et al., *Energy and AI* / 数据综述, 2021

- **题名**：Lithium-ion battery data and where to find it  
- **DOI**：[10.1016/j.egyai.2021.100081](https://doi.org/10.1016/j.egyai.2021.100081)  
- **简要说明**：系统汇总公开锂电数据集与下载线索。  
- **对本题的帮助**：附录“相关公开数据”索引。

---

## 七、书籍（建模基础）

### 22. Gregory L. Plett — *Battery Management Systems*

- **Vol. I: Battery Modeling**（等效电路与建模基础）  
- **Vol. II: Equivalent-Circuit Methods**（SOC/SOH 估计）  
- **Vol. III: Physics-Based Methods**（物理模型、快充与寿命相关控制）  
- **简要说明**：BMS 领域标准教材。  
- **对本题的帮助**：SOH 定义、估计与**问题4快充优化**概念；Vol. III 最贴问题4。

### 23. Newman & Balsara — *Electrochemical Systems* (4th ed.)

- **简要说明**：电化学系统经典教材；多孔电极与电池 continuum 建模基础（DFN/P2D 一脉）。  
- **对本题的帮助**：机制深入与物理模型背景（不必整本精读）。

### 24. Kutner et al. — *Applied Linear Statistical Models*

- **简要说明**：回归、ANOVA、实验设计与推断。  
- **对本题的帮助**：**问题2** 策略间差异显著性、多元回归。

### 25. Pinheiro & Bates — *Mixed-Effects Models in S and S-PLUS*

- **简要说明**：混合效应模型经典参考。  
- **对本题的帮助**：问题2 中“策略随机效应 + 参数固定效应”的写法。

### 26. Brockwell & Davis — *Introduction to Time Series and Forecasting*

- **简要说明**：时间序列基础。  
- **对本题的帮助**：问题3 SOH 时序外推的统计背景（若采用经典时序模型）。

---

## 八、使用约定（给队友）

1. **写论文时**：优先引用本目录“一、必读核心”四篇 + Liu 2019。  
2. **数据声明**：必须引用 Severson 2019，并注明 [data.matr.io/1](https://data.matr.io/1/)。  
3. **导入文献管理器**：直接导入 `references.bib`。  
4. **不要**把付费墙 PDF 未经授权上传到本仓库。  
5. 若新增文献：请同步更新本 README 与 `references.bib`，并注明“对应问题”。

---

## 九、推荐引用格式示例（GB/T 7714 风格示意）

```text
[1] SEVERSON K A, ATTIA P M, JIN N, et al. Data-driven prediction of battery cycle life before capacity degradation[J]. Nature Energy, 2019, 4(5): 383-391.
[2] ATTIA P M, GROVER A, JIN N, et al. Closed-loop optimization of fast-charging protocols for batteries with machine learning[J]. Nature, 2020, 578: 397-402.
[3] LIU Y, ZHU Y, CUI Y. Challenges and opportunities towards fast-charging battery materials[J]. Nature Energy, 2019, 4: 540-550.
```

更完整的条目见同目录 `references.bib`。
