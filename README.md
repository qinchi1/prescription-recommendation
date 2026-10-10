# 中药集合推荐与重排序

研究代码：以固定词表中的症状多热向量为输入，生成中药候选集合，并通过双教师蒸馏训练候选集合重排序模型。

```text
390 维症状 → 图候选模型 → 811 种中药打分 → Top-50 → 集合重排序
                                                 ↑ 训练时蒸馏
                                      遮蔽中药教师 + 证候/治法教师
```

输出是中药 ID 和排序分数。模型不生成剂量，当前交付也不提供临床疗效或临床处方审核证据。

## 实现

| 模块 | 代码 | 内容 |
|---|---|---|
| 数据与特征 | `herbrec/data/` | 冻结词表和分割、90 维知识特征、23 维属性、BGE 症状编码、候选特征 |
| 候选模型 | `herbrec/models/knowledge_graph.py` | GATConv、TransformerConv、跨注意力，预测 811 维中药分数 |
| 教师 | `herbrec/teachers/` | 症状条件下的遮蔽中药预测，以及证候、治法和中药多任务教师 |
| 重排序 | `herbrec/models/reranker.py` | 集合 Transformer，在标准化候选分数上添加有界残差 |
| 损失与训练 | `herbrec/losses/`、`herbrec/training/` | 监督排序损失、双教师蒸馏、覆盖感知门控和消融接口 |
| 推理与评价 | `herbrec/inference.py`、`herbrec/evaluation.py` | 标签与教师目标不进入学生推理，输出排序和集合推荐指标 |

默认参数集中在 `herbrec/configuration.py`，命令入口为 `main.py`。

## 环境

已做 CPU 小批量检查的环境组合：Python 3.11、NumPy 1.26.4、pandas 2.2.3、PyTorch 2.6.0、torch-geometric 2.6.1、transformers 4.57.6。该组合不是完整环境锁定文件；GPU 兼容性没有在本代码包中全面验证。

```powershell
python install_dependencies.py
python main.py --help
```

依赖安装脚本使用部分宽版本范围。安装成功或模块导入成功，不等同于完成真实数据训练。

## 必需运行资产

代码包不包含病例数据、中药知识文件、预训练 BGE 模型、教师缓存和模型权重，也不提供这些资产的自动下载。数据的公开来源、版本、访问方式与许可信息仍需作者在发布运行资产时补充；不能仅凭本代码包独立复现论文结果。

`--source-root` 指向运行资产根目录，`--output-root` 指向新的产物目录。默认资产布局由 `herbrec/paths.py` 定义：

| 相对资产目录 | 主要内容 |
|---|---|
| `data/prescriptions/` | `X_symptom_multihot.npy`（N×390）、`Y_herb_multihot.npy`（N×811）、症状/中药词表 JSON、`train_idx.npy`、`val_idx.npy`、`test_idx.npy` |
| `data/knowledge/`、`data/knowledge_features/` | 中药知识表、证候到属性映射、90/23 维特征 |
| `bge-base-zh-v1.5/`、`data/symptom_embeddings/` | 本地 BGE 模型、症状嵌入及词表顺序校验信息 |
| `data/candidate_folds/` | 每折 fit/calibration/holdout 清单、对应模型与图，以及 merged OOF 数组 |
| `data/reranking/`、`artifacts/reranking_inputs/` | 静态统计资产、训练张量、验证或测试的无标签输入 |
| `artifacts/masked_herb_teacher/` | 遮蔽教师各折清单、权重、频率分组和 OOF 教师缓存 |
| `data/clinical_teacher/`、`artifacts/clinical_teacher/` | 临床来源数组、冻结来源分折、映射清单、教师权重、输出与支持掩码 |
| `checkpoints/`、`data/knowledge_graph/` | 候选模型、学生模型权重和训练图 |

词表顺序、样本 ID、候选顺序和各缓存必须一致。当前代码读取已有标准化数组与冻结分割，尚不包含从任意原始病例重新构建整套资产的流程。

## 在已有研究工作区运行

已有 `GraphAI-for-TCM-main` 资产时，使用 `--existing-project` 兼容原目录布局。以下命令在本项目根目录执行；修改 `$source` 为实际资产目录，`$output` 使用新的目录。

```powershell
$source = "../GraphAI-for-TCM-main"
$output = "../prescription_recommendation_runs/check"

python main.py validate --source-root $source --output-root $output --existing-project --device cpu --limit 4 --train-steps
```

该命令依赖已有参考权重和预测文件，检查模型加载、真实数据计算、梯度及各训练器的小批量更新。

已有候选 OOF 数组和两类教师缓存时，可检查候选预测、学生训练、推理与验证评价的衔接：

```powershell
$output = "../prescription_recommendation_runs/pipeline_check"
python main.py run --source-root $source --output-root $output --existing-project --device cpu --limit 4 --epochs 1 --max-batches 1
```

这是有限样本的运行检查，生成的权重与指标不能作为正式实验结果。去掉三个限制参数会使用配置中的完整学生训练预算，但仍依赖已有冻结资产。

单独执行学生预测时，显式指定已选定的权重：

```powershell
python main.py predict --source-root $source --output-root $output --existing-project --device cpu --split validation --checkpoint "$output/checkpoints/reranker/full/best_model.pt"
```

`run --retrain` 在预先准备的冻结分折与部分 OOF 资产上重训各模型，不是从空目录或任意原始数据开始的独立复现命令。

## 产物与评价

运行产物写入 `--output-root`：候选预测位于 `candidates/`，学生输入位于 `reranking_inputs/`，权重位于 `checkpoints/`，排序结果位于 `predictions/`，评价位于 `evaluation/`。

预测 NPZ 包含 `sample_ids`、`candidate_ids`、`scores`、`residuals` 和 `ranked_ids`。ID 对应资产目录中的冻结中药词表。分数用于排序，不应解读为临床用药概率。

默认 `ranking_metrics` 的 P/R/F1@5/10/20 汇总所有样本命中数，属于 micro 指标；CandidateRecall@50 同样按全部真实中药命中数汇总。候选模型和临床教师选模使用 `mean_case_f1`，属于样本平均 F1。两种 F1 不能以同名指标直接比较提升。

## 验证范围与当前限制

整理代码时，已在原有运行资产下验证 4 个真实样本的模型计算、各训练器的一次批量更新，以及候选预测到学生训练、推理、验证评价的连接。原有 Student 权重预测一致，候选权重预测的最大差异约为 3.8×10⁻⁶。未重新执行完整训练、多种子主实验或独立测试；小批量结果不证明性能提升或论文发表准备度。

当前版本仍有以下需要修复或补充的事项：

- OOF 导出存在分折隔离检查，但没有完整核验 checkpoint 实际训练/校准样本与图来源；`strict_oof=true` 不能代替模型来源证据。
- 静态中药频率和症状共现特征只排除第 0 折后拟合，不能声称所有训练输入都逐样本折外生成。这与验证特征仅使用训练统计是不同的问题。
- 分阶段命令的默认权重路径、训练和验证的预测字段尚未完全统一；部分入口可能回退到源目录权重。
- 样本 ID 的负数边界和评价分区校验需要加强，正式评价前必须核对样本所属分区。
- CLI 的 `--seed` 尚未控制全部训练器；无教师消融仍会读取教师资产。

正式发布可复现结果前，需要修复这些代码问题，明确资产获取方式，并提供相同分割与统一指标下的基线、消融和统计结果。README 不构成这些问题已经解决的声明。

当前包尚未附带开源许可证；第三方数据、模型和代码的再分发条件需要分别确认。
