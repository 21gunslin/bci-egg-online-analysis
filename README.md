# 本机脑电研究工作台

在本机浏览器分析 BDF / FIF；Python 服务仅监听 `127.0.0.1:8765`。命令行兼容 EDF / GDF。

## 想看懂代码，先从这里开始

打开[代码阅读流程 JPG](docs/code_reading_flow.jpg)，再读[从零读懂代码](docs/CODE_READING_GUIDE.md)。主线是：HTML按钮 → 单份JavaScript入口 → HTTP服务 → 分析总流程 → 读取 → 预处理 → 特征 → 图表与网页。

`eeg_feature_engine.py`保留主流程和兼容导入；参数、读取、预处理、公式和报告分别位于`eeg_models.py`、`eeg_readers.py`、`eeg_preprocessing.py`、`eeg_features.py`、`eeg_reporting.py`。常用MNE参数集中在`eeg_mne_ops.py`。前端也按单份、配对、多份、事件、解释和原始检视拆分，使用原生JavaScript。

标准读取、IIR滤波、Welch PSD、逐窗功率和FIF优先使用MNE公开接口；NumPy/SciPy保留明确的论文公式。原始BDF保真兼容层单独保留，不隐式上采样或改写Status。Black和Prettier只用于开发排版，网站运行不需要安装npm依赖。

## 功能

- 辅助人工巡检：全记录候选概览、一键定位、持续性问题合并、每页4通道与自动翻页；明确登记复核后记录准确浏览覆盖，候选及进度独立于BAD/FIF。

- 只读自适应检视：新BDF可只载入查看，旧结果可直接恢复；纵轴逐通道/同单位统一/手动标尺切换，横轴缩放、拖动和整段概览，支持单通道放大及仅导出当前视图，不改写BDF/FIF。

- 原始BDF质控与FIF：独立、不滤波/重采样/重参考/删道删段；原始波形检视、全通道全频Welch谱、手工坏道/坏段、枕区睁闭眼α峰与质控CSV，双精度FIF携带模板坐标及标记并逐块回读核对。详见[使用及保留口径](docs/RAW_BDF_QC.md)。

- Montage模板映射：RecordInfo预览和分析结果展示原始/标准电极名、通道类型与head坐标，提供CSV和模板点位图；不确定通道不强配，详见[装配说明](docs/MONTAGE_MAPPING.md)。

- 文件夹导入：选择某次姓名编号记录文件夹，或选择上层总目录并按编号/姓名/时间查找一次记录；自动获取数字子文件夹里的data、匹配evt与两份JSON，仍可手动增删与替换。

- 可选“论文特征方案”：每时相287列，配对前/后/差值861列；时频先汇总频段功率再计算标准差/CV，提供字典、缺失原因和固定列宽表。详见[计算口径](docs/THESIS_FEATURE_PROFILE.md)。

- 单份 EEG：0.5–45 Hz 四阶 Butterworth 双向零相位滤波，片段质控，时域、频域、时频和 PLV 网络特征。
- 镇痛前后对比：共同保留通道与相同有效时长，统一图表尺度，特征差值与 CSV 下载。
- spike 独立入口：单通道 1 Hz 设备计数；可添加同一记录段原始 EEG，复核候选通道、对应头皮区域及后续慢波。
- 多份入口可选导入datainfo.json生成计划及份数，并用RecordInfo.json按名称核对通道类型；保留时间差提示与安全白名单。
- 多份 data 时间顺序队列：2–30份、同一evt、间隙保留、独立连续段计算和加权特征汇总。
- 原三个入口均可选上传 evt.bdf，查看原始事件标签、按文件头对齐的时间轴及片段内外位置。
- 图表可点击放大，并支持新窗口查看和下载。

## 安装和启动

建议 Python 3.10–3.12。先安装依赖：

```powershell
python -m pip install -r requirements.txt
```

Windows 双击 `scripts/启动脑电特征网页.bat`，再访问 <http://127.0.0.1:8765/>。
可双击 `scripts/停止脑电特征网页.bat` 停止后台服务。
其他系统可运行：

```sh
python backend/eeg_web_server.py
```

完整使用方法见 [使用说明](docs/USER_GUIDE.md)。

## 数据与方法说明

此仓库只包含程序与使用说明，不包含产妇原始记录、特征表、随访资料、报告、日志或分析缓存。运行时输出保存在本机 `results/eeg_web_output/`，已列入 Git 忽略规则。

`spike.bdf` 支持的是标签为 spike 的单通道 1 Hz 计数结构。该文件本身不能定位出现通道、头皮区域或慢波；需要配套原始 EEG。自动波形检测是未经专家标签验证的启发式候选筛查。对应头皮区域不是脑源定位，结果不能用于诊断癫痫或产后抑郁。

配对图时间为质控有效片段的拼接时间，不是原始连续时间，也不能用于宫缩事件同步。镇痛前后标签须人工核对，差值不证明因果作用。

## 当前验证

v0.11.0通过84项Python和30项JavaScript回归测试；新增MNE与既定SciPy滤波/PSD/STFT密度的数值对照，并保留重构前后的合成BDF/FIF特征对照。另检查本机API、图表和草稿操作。功能验证不代表临床预测有效性。

## 目录与维护

| 目录 | 用途 |
|---|---|
| `backend/` | 文件读取、预处理、特征计算、配对/spike 分析与 HTTP API |
| `web/` | HTML、CSS 和 JavaScript 页面 |
| `scripts/` | Windows 后台服务启动、健康检查及停止脚本 |
| `docs/` | 使用方法、架构与方法边界 |
| `tests/` | 不依赖真实患者资料的合成数据验证 |

验证命令：

```sh
python -m unittest discover -s tests -v
python -m compileall -q backend
node --check web/eeg_app.js
node --test tests/test_folder_import.cjs
```

Node.js 仅用于 JavaScript 语法检查，网页运行不需要 Node.js。

贡献与分支流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，版本变更见 [CHANGELOG.md](CHANGELOG.md)。


## MNE 对照审查（0.2.0）

可选安装 `python -m pip install -r requirements-mne.txt` 后执行：

```sh
python tools/mne_parity_review.py
```

可用 `--real-data-root` 指定本机 EEG 数据目录；输出默认保存在已忽略的 `results/mne_review/`。工具不输出患者姓名、文件路径或文件头身份信息。完整审查范围、已修正问题及剩余限制见 [MNE_REVIEW.md](docs/MNE_REVIEW.md)。`tools/` 保存独立对照验证工具。

对照通过不等于临床有效性验证。未知参考、设备单位误标、BAD 注释与拼接边界仍需结合采集资料处理。


## BDF/FIF互斥导入与同目录保存

单份、多份及镇痛前后入口可单选BDF或FIF，禁止混用。原始质控可从本机选择BDF，将主EEG组保存为同目录`fdata.fif`，已有导出先备份。FIF单份分析另有原始坏道、坏段、眼状态/EOG复核图。详见[FIF导入导出说明](docs/FIF_IMPORT_EXPORT.md)。

## 批量转换、特征与随机森林

新增独立批处理入口`scripts/run_batch_fif_rf.ps1`：逐份保真转换BDF到同目录FIF，提取论文特征，再汇总成每位产妇一行，采用固定参数随机森林和按产妇独立的重复5折交叉验证计算OOF AUC。转换、质控和建模排除分别留痕，不覆盖未知的已有FIF。

入门请看[从BDF到AUC的代码阅读指南](docs/BATCH_RANDOM_FOREST_GUIDE.md)，先读`tools/train_random_forest.py`的`main()`。可复现依赖单列于`requirements-batch.txt`；患者特征、预测、模型及报告只保存在本机忽略目录。内部评价不等于临床有效性。

另附[批处理代码流程JPG](docs/batch_random_forest_flow.jpg)。新增7项合成数据测试覆盖保真转换、通道名称对齐、病例排除和训练/测试泄漏；`tools/validate_batch_results.py`复核实际结果中的标签对应、分折与指标。
