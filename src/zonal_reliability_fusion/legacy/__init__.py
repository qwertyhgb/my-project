"""归档的旧研究线（modality gate / feature fusion / zonal reference）。

这一层的内容**不再是项目主线**，但必须保留：既有 checkpoint 的输出目录由 Trainer 类名决定，
改名即让历史产物失联；已完成的实验是论文的 preliminary / negative evidence，必须可复现、
可核查。

内容
----
- :mod:`fusion_networks`——输入级 modality gate、浅层序列特异特征融合、同区参照残差融合；
- :mod:`prior_transforms`——旧 Dataset606 的 PZ/TZ 增强边界（行为契约被新主线继承）；
- :mod:`fusion_trainers`——上述网络对应的全部 Trainer 类。

使用规则
--------
- **禁止**在本包内新增方法或修改超参数；
- **禁止**让新主线的默认训练流程依赖这些模块（兼容转发层除外）；
- 需要引用旧实验时，请链接 ``docs/archive/`` 下的归档文档，不要在主 README 里介绍它们。
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
