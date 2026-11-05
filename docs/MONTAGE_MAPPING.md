# Montage标准模板映射（0.8.0）

本功能装配原始通道名称、标准电极名、候选/配置通道类型与模板几何。坐标为**MNE模板坐标，不是产妇实测电极位置**。不会改名原始信号、改变通道顺序、重新参考、插值坏导联或改变四域特征。已有信号分析可在缺少位置时继续；位置缺失不等于该通道没有信号。

## 来源与坐标系

- 用户选择标准10–20模板。冻结数据由MNE-Python 1.13.2的公开API生成；请求名`standard_1020`，该版本实际采用其新名称`colin27_1020`，含94个标准/扩展点位。
- 调用`make_standard_montage(..., head_size='auto')`；用`compute_native_head_t`和`DigMontage.apply_trans`将原始MRI坐标转换为`head`坐标。另与`Info.set_montage`得到的坐标逐点验证一致。
- 导出单位**米**，x向右、y向前、z向上。不能当作MNI坐标直接解释。
- 保存MNE版本、变换矩阵、三基准点、模板SHA-256与生成脚本。MNE许可见`backend/data/LICENSE_MNE.txt`，导出来源说明也包含完整许可。

## 名称与类型规则

| 原始名 | 标准名 | 处理 |
|---|---|---|
| Fp1/FP1/EEG Fp1-REF等格式变体 | Fp1 | 只规范名称格式，不推断参考电极 |
| T3 / T4 | T7 / T8 | 与当前模板中的旧名点位逐点核对重合 |
| T5 / T6 | P7 / P8 | 与当前模板中的旧名点位逐点核对重合 |
| T1 / T2 | 留空 | 无同名模板点位；设备定义未确认，不套用其他别名 |
| PG1 / PG2 / X1–X6 | 留空 | 无足够采集依据，不猜测坐标 |
| A1 / A2 | A1 / A2 | 列出模板耳部坐标；不确定其采集角色，不自动设为参考或纳入原头皮特征 |
| EKG/ECG/EOG/EMG/Status等 | 不赋头皮坐标 | 按RecordInfo类型或明确标注的候选标签规则分类 |
| F3-A1等双极/非标准名称 | 留空 | 不能把两个电极的导联信号强行当成一个头皮点 |

优先使用RecordInfo配置类型；没有配置时，类型是标签规则识别的候选，非采集用途确证。多个通道指向同一物理模板坐标时标记冲突，禁止静默装配。**不调用MNE的全局自动别名匹配**；仅使用本表明确且由模板坐标验证过的四条旧名规则。

## 网站位置

1. 顶部展开“Montage映射”，可查看当前数据出现过的通道目录及CSV/模板位置示意图。目录只含通道名称与类型，不含患者身份或各次记录。
2. 多份入口选择RecordInfo或自动导入文件夹后，JSON预览中出现当前配置的映射表，可搜索原始/标准名，筛选待核对项，下载当前映射CSV。
3. 单份、多份、配对结果各有实际通道映射，标明本次是否纳入汇总。多份表按文件编号保留每段原始通道；失败段仍列出，并标注未纳入。
4. 单份Excel新增“Montage模板映射”工作表。图表为三个坐标投影的点位核对图，未做功率插值或脑源定位。颜色标明本次保留、未纳入或尚未质控。

## MNE装配接口

`backend/eeg_montage.py`提供`make_mne_montage(labels, recordinfo=None, allow_partial=False)`，返回`DigMontage`及映射信息。`ch_pos`键保留原始通道名，匹配到的坐标来自标准名。

默认对未确认EEG坐标或重复位置报错；明确设`allow_partial=True`时只装配确认的子集，返回表仍保留缺失项。不调用重参考或坏道插值，也不修改Raw。

```python
from eeg_montage import make_mne_montage

# 已确认类型和名称、且有模板位置的示例；保留T3原名，坐标来自T7。
montage, mapping = make_mne_montage(["Fp1", "Fp2", "F3", "F4", "T3"])
# 若用于MNE Raw，先核对其通道类型；可在副本的已确认通道子集上设置。
# raw_geometry = raw.copy().pick(montage.ch_names)
# raw_geometry.set_montage(montage, match_case=True, match_alias=False,
#                         on_missing="raise")
```

## 重建与验证

安装可选`requirements-mne.txt`后，`tools/build_montage_template.py`可重建模板；网站运行读取冻结模板，不强制安装MNE。

`tools/assemble_montage_mapping.py`可读取单个RecordInfo、单次记录目录或记录上层目录，汇总通道名称/类型并导出映射CSV与模板图。不读取BDF波形，不导出源路径、姓名、患者字段或摄像头设置；出现次数指配置文件次数，不能当成受试者数。

测试包含坐标来源/单位/方向、旧名等点位、未知/双极名称缺失、非EEG排除、重复坐标冲突、隐私字段排除和导出。实际MNE验证保留原始通道名及波形，检查`set_montage`后坐标一致、参考状态与采样值未改变。

参考：[MNE标准montage](https://mne.tools/stable/generated/mne.channels.make_standard_montage.html)、[MNE原始坐标到head变换](https://mne.tools/stable/generated/mne.channels.compute_native_head_t.html)、[MNE DigMontage](https://mne.tools/stable/generated/mne.channels.DigMontage.html)。
