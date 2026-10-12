# Third-party reference code

本目录保存训练框架与论文参考源码。第三方源码保持只读，项目扩展放在
`src/zonal_reliability_fusion/` 的对应模块中；新增参考源码不自动改变主模型或运行环境。

| 目录 | 来源 | 用途 | 固定版本 / 本地核查版本 | 许可证 |
|---|---|---|---|---|
| `nnUNet/` | [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) | 项目实际使用的预处理、训练与推理框架 | tag `v2.6.2`, commit `74ceb6803d10dcee29b2cc481678d3a3d069f281` | 本地LICENSE：Apache-2.0 |
| `Prostate-Lesion-Segmentation/` | [NUBagciLab/Prostate-Lesion-Segmentation](https://github.com/NUBagciLab/Prostate-Lesion-Segmentation) | *Align then Refine: Text-Guided 3D Prostate Lesion Segmentation* 的作者代码，当前供结构与复现条件阅读参考 | 2026-10-12核查HEAD：`d6b549286999407ca7458146c7d8da7717bd0e00`；提交日期2026-04-23，本地工作区干净 | 本地LICENSE：Apache-2.0 |

## Align then Refine：源码位置与使用边界

论文：[arXiv:2604.18713](https://arxiv.org/abs/2604.18713)。作者README说明该工作被IEEE EMBC 2026接收。
本地源码位置为`third_party/Prostate-Lesion-Segmentation/`，版本以表中commit为本次快照。
源码已由用户加入；目前未安装该fork、未通过其入口启动训练，也未接入项目Trainer注册表。
本次确认了README、许可证、Git来源及相关入口，尚未完成全部代码与论文的逐项一致性审计。

| 阅读内容 | 作者仓库内的位置 |
|---|---|
| 模型与运行说明 | `README.md` |
| 多编码器基础Trainer | `nnunetv2/training/nnUNetTrainer/nnUNetTrainerMultiEncoderUNet.py` |
| 文本条件、alignment/heatmap辅助训练 | `nnunetv2/training/nnUNetTrainer/nnUNetTrainerMultiEncoderUNetText.py` |
| Attention/refinement条件 | `nnunetv2/training/nnUNetTrainer/nnUNetTrainerMultiEncoderAttnUNetText.py` |
| 作者训练、推理及评价脚本 | `tools/train/`、`tools/infer/`、`tools/eval/` |

本项目仍使用固定`third_party/nnUNet`及项目入口；当前C/D的shallow stems、
pre-fusion coarse lesionness与预测解剖条件融合详见[Method](../docs/Method.md)。
作者仓库的多编码器、文本监督与refinement可用于理解相关工作的机制和比较条件，
不能据源码加入就声称本项目已经使用或复现这些模块，也不将作者论文指标当作同协议基线结果。

该fork的Python包名同样是`nnunetv2`，pyproject声明版本也为2.6.2，版本号本身不能区分源码。
因此保持`lm`的现有依赖与`scripts/env_nnunet.sh`指定的官方源码路径，
不将这个fork以`pip install -e`安装到`lm`，不把其目录加入项目运行时PYTHONPATH。
作者README中的环境创建、安装和训练命令是上游复现说明，不是本项目执行指令。

主仓库`.gitignore`继续忽略第三方源码，只跟踪本说明；新源码是独立本地checkout，
不是主仓库已配置的Git submodule。其他机器需要单独取得作者仓库并checkout表中commit，
仅克隆本项目不会自动得到该参考源码。未修改ignore规则或上传整份第三方源码。

后续新增或更新参考源码时，同步记录URL、commit/tag、许可证、阅读/复现状态和实际借鉴的模块；
代码接入须另有明确授权与项目侧变更记录，不能把参考快照更新默认为主线架构变化。
