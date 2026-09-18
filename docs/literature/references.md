# 参考资料总表

> 本文件保存项目引用来源的完整清单与逐条注释；`docs/research_plan.md` §21 只保留核心来源与指针。
> **引用纪律**：① 只使用一手或权威来源（官方页面、DOI、PubMed、期刊页、作者官方仓库）；
> ② 未核读正文的来源必须标注「需人工确认」；③ **不得**仅凭关键词检索宣称「首次」或「绝对创新」。

## A. 临床与任务背景

1. [PI-RADS v2.1, American College of Radiology](https://www.acr.org/-/media/ACR/Files/RADS/PI-RADS/PIRADS-V2-1.pdf)
   —— PZ 以 DWI 为主导序列、TZ 以 T2W 为主导序列；本项目只作为**医学动机**，不硬编码为分割规则。
2. [PI-CAI: Artificial intelligence and radiologists in prostate cancer detection on MRI](https://doi.org/10.1016/S1470-2045(24)00220-1)
   —— Lancet Oncology 2024；PI-CAI 主论文。

## B. PI-CAI 官方数据与标注（一手事实）

3. PI-CAI 官方数据页：<https://pi-cai.grand-challenge.org/DATA/> ——
   配准状态（公开训练集**原则上不共配准**，仅 54/9107 手动刚性配准；隐藏调优/测试全部配准）、
   HBV = 轴位高 b 值 DWI（**b ≥ 1000 s/mm²**）、**ADC 绝对强度跨中心不可直接解读**、
   影像许可 **CC BY-NC 4.0**、公开集**含 ProstateX 的 328 例**、参考标准（阳性 = 组织学 ISUP ≥ 2；
   阴性 = ISUP ≤ 1 或 MRI PI-RADS ≤ 2）。
   → 本项目 §4.2/§4.2.1 与 §4.5 的官方依据。
4. [PI-CAI expert-derived annotations（含 Pooch25）](https://github.com/DIAGNijmegen/picai_labels) ——
   标注目录结构、`human_expert/original` 与 `resampled` 的差异、Pooch25（对 205 例此前仅 AI 标注的阳性补充
   人类专家标注，**二分类 0/1**）与 Bosma22a/b AI 标注；README 明确提示
   全腺体 AI 分割错误示例 `11050_1001070`；**不得与 ProstateX 叠加**。
5. [PI-CAI 影像数据（Zenodo）](https://doi.org/10.5281/zenodo.6624726)
6. [Official PI-CAI baseline](https://github.com/DIAGNijmegen/picai_baseline) ——
   官方 nnU-Net（v1）基线与 Focal+CE 损失定义（本项目 N0 的 v2 port 依据）。
7. [Official PI-CAI evaluation library（picai_eval）](https://github.com/DIAGNijmegen/picai_eval) ——
   detection map、AP、FROC 与 `min_overlap`（IoU ≥ 0.1）口径来源。

## C. 方法学基线（骨干与融合）

8. [nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation](https://doi.org/10.1038/s41592-020-01008-z)
9. [nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation](https://arxiv.org/abs/2404.09556)
10. [Official nnU-Net Residual Encoder presets](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md) ——
    `blocks_per_stage=(1,3,4,6,6,6,6)` 的官方来源；本项目**不重新规划 plan**，故不与官方 ResEnc-M 等价。
11. [Prostate cancer segmentation from MRI by a multistream fusion encoder](https://doi.org/10.1002/mp.16374)
12. [NaMa: Neighbor-Aware Multi-Modal Adaptive Learning for Prostate Tumor Segmentation](https://doi.org/10.1609/aaai.v38i5.28215) ——
    影像驱动自适应融合的最近邻基线（M2 的对照来源）。
13. [NesMFle: A Neighbor-Sensitive Multi-Modal Flexible Learning Framework](https://doi.org/10.1109/TBME.2025.3562766)
14. [Lesion-guided selective multi-modal integration for prostate cancer segmentation and PI-RADS grading](https://doi.org/10.1002/mp.70019) ——
    LeSMI / Dynamic Modality Weighting；「学习三个序列权重」本身不构成创新。

## D. 分区先验与解剖引导（新颖性邻接工作）

15. [AtPCa-Net: anatomical-aware prostate cancer detection network on multi-parametric MRI](https://doi.org/10.1038/s41598-024-56405-7)
16. [Z-SSMNet: Zonal-aware Self-supervised Mesh Network](https://doi.org/10.1016/j.compmedimag.2025.102510)
17. [Enhancing prostate cancer segmentation in bpMRI: Integrating zonal awareness into attention-guided U-Net](https://doi.org/10.1177/20552076251314546)
18. [Advancements in prostate cancer segmentation: Integrating prostate zonal information](https://pmc.ncbi.nlm.nih.gov/articles/PMC13071292/)
19. [HeviAI23 zonal segmentation predictions](https://doi.org/10.5281/zenodo.7615350) —— PZ/TZ 来源敏感性审计对象。
20. Backbone-Conditional Behavior of Modality Gating in Multi-Modal Prostate MRI Segmentation（MIGF）：
    <https://arxiv.org/abs/2604.10702> —— **模态级**门控（PI-CAI + Prostate158）；
    摘要层面未见 PZ/TZ 条件化；其报告「卷积 backbone 上门控塌缩为近静态模态先验（跨样本权重 SD 0.0033）」
    的对本项目 M2/H2 有直接参考价值。**方法正文未核读，需人工确认。**
21. [Attention U-Net](https://arxiv.org/abs/1804.03999)、[UNet++](https://pmc.ncbi.nlm.nih.gov/articles/PMC7329239/) ——
    说明 encoder–decoder / skip 改造已有充分先例，本项目不作为创新。

## E. 外部候选数据集

22. [Prostate158: expert-annotated 3T MRI dataset](https://doi.org/10.1016/j.compbiomed.2022.105817) ——
    G0-E 首选外部候选；标签语义与 DWI/HBV 映射待审计。

## F. 规范与报告指南（文档边界与统计纪律的依据）

23. SPIRIT 2025 statement: updated guideline for protocols of randomised trials——
    BMJ 2025;389:e081477，DOI [10.1136/bmj-2024-081477](https://doi.org/10.1136/bmj-2024-081477)；
    同步发表于 Nature Medicine 2025;31:1784–1792，DOI [10.1038/s41591-025-03668-w](https://doi.org/10.1038/s41591-025-03668-w)。
    可读取正文要点：协议是 living document；**每一版方案都应包含透明审计留痕（日期 + 变更描述）**；
    重要修订须上报并在完成报告中描述；SAP 应在方案中被引用且可获取（条目 5）；SPIRIT = 计划、CONSORT = 结果，
    两者比对可识别选择性报告与未披露修订。
    **适用性限制**：SPIRIT 面向随机对照试验；本项目只借用「协议/结果分离、版本留痕、SAP 预先冻结」三条原则。
24. CLAIM 2024 Update——Radiology: Artificial Intelligence，DOI
    [10.1148/ryai.240300](https://doi.org/10.1148/ryai.240300)（官方页面 <https://pubs.rsna.org/page/ai/claim>）。
    AI 医学影像研究的**报告**清单，与协议互补。**本次核查未取回清单正文（页面 403 / PDF 不可解析），
    条目编号与措辞需人工确认。**
25. Pooch et al., 2025（Pooch25 标注来源，预印本）：
    <https://www.medrxiv.org/content/early/2025/05/13/2025.05.13.25327456> ——
    使用 Pooch25 标注时必须引用；**是否已正式发表及其标注流程细节需人工确认。**
26. 本项目本次核查的完整记录：`docs/literature/novelty_and_standards_check_20260917.md`
    （含检索方式、限制、未做事项与系统检索协议建议）。
