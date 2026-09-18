# Third-party reference code

本目录保存论文复现、接口参考和基线实现。第三方仓库原则上保持只读，自研修改放在 `src/zonal_reliability_fusion/integrations/`。

| 目录 | 来源 | 用途 | 固定版本 |
|---|---|---|---|
| `nnUNet/` | https://github.com/MIC-DKFZ/nnUNet | 3D 分割基线、预处理和训练框架 | tag `v2.6.2`, commit `74ceb6803d10dcee29b2cc481678d3a3d069f281` |

后续加入 Z-SSMNet、TriD 或其他参考实现时，应同时记录仓库 URL、commit/tag、许可证和实际借鉴的模块。
