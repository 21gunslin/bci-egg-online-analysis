# 从零读懂脑电网站代码

先打开[代码阅读流程 JPG](code_reading_flow.jpg)，再按下面顺序阅读。不要一开始钻进二进制解码或网络公式，也不用从第一个文件第一行一直读到最后一行。

## 先建立三个概念

**前端**是浏览器里看到的按钮、输入框和图片。它负责收集选择、显示结果，用原生 JavaScript 编写，不需要学习 React 等框架。

**后端**是本机运行的 Python 程序。浏览器把文件和设置送给它，它负责读取、计算、保存图表。

**库**是别人已经写好、经过维护的工具。这里用 MNE 处理标准脑电读取、滤波、PSD、注释和 FIF；用 NumPy 处理数值表，用 SciPy 补充已有的信号与统计公式，用 Pandas 整理表格，用 Matplotlib 画图，用 OpenPyXL 写 Excel。没有额外引入图论库或其他小众脑电分析框架。

## 第一次只追一条主线：单份分析

### 1. 在 HTML 里找到按钮

打开 `web/eeg_feature_app.html`，搜索 `singleForm`。你会看到文件选择、格式单选、参数和“开始单份分析”按钮。

HTML 描述页面上**有什么**；CSS 决定**长什么样**；JavaScript 决定**按下以后做什么**。这一阶段只要认清 `id`：它是元素的名字，例如 `singleFile` 是文件选择框。

### 2. 找到按钮绑定的函数

打开 `web/eeg_single_analysis.js`，读 `handleSingleSubmit(event)`。它按顺序完成：

1. 阻止表单直接刷新网页。
2. 读出分析参数与勾选的特征类别。
3. 按需上传事件文件。
4. 调用 `analyzeSingleFile(...)`。
5. 用 `render(...)` 把结果放到页面。

`const result = await ...` 的意思是“等这个动作完成，再把返回值命名为 result”。`async/await` 用来等待本机服务计算，不代表在这里计算脑电。

`analyzeSingleFile` 定义在 `web/eeg_ui_common.js`。它把文件发给 `/api/analyze`。`post(...)` 发送 JSON 小参数；上传脑电则直接发送文件内容。

### 3. 沿着 URL 找后端接收位置

打开 `backend/eeg_web_server.py`，搜索 `_single_analysis`。这个函数像接线员：检查格式、读取设置、建立结果目录，然后调用 `analyze_eeg_file`。

它处理“文件怎么进来、结果怎么送回去”，不负责定义均值、PSD 或 PLV 的数学公式。`eeg_web_extra.py` 负责多份、配对、上传等扩展接口；`eeg_raw_qc_api.py` 单独处理原始检视与导出。

### 4. 先读主流程，暂时跳过具体公式

打开 `backend/eeg_feature_engine.py`，读 `analyze_eeg_file`。这是学习后端最重要的入口：

```python
data, labels, units, header, annotations = read_eeg_file(source)
preprocessed = preprocess_eeg(data, header, settings)
channel_table, summary_table, metadata = extract_features(preprocessed, settings)
# 后面把已有结果整理成图、CSV、Excel和文字报告。
```

这里为了讲解省略了实际函数里的部分参数。请理解各变量的意义，而不是复制这四行替代正式程序。

| 名称 | 可以怎么理解 |
|---|---|
| `source` | 输入文件的本机路径 |
| `data` | 数值表；每行一个通道，每列一个采样时刻 |
| `labels` | 每一行对应的电极/通道名称 |
| `units` | 每行的单位；不能把数字计数当成微伏 |
| `header` | 文件信息卡：时长、采样率、通道、注释等 |
| `settings` | 参数表：目标采样率、分段长度、陷波等 |
| `preprocessed` | 处理后的数据及其原始时间索引、质控记录 |
| `channel_table` | 各通道的特征表 |
| `summary_table` | 脑区或全脑汇总特征表 |
| `metadata` | 本次使用的公式、参数、缺失原因等说明 |

主文件保留了旧函数名的导入兼容。看到 `from eeg_preprocessing import preprocess_eeg`，就去对应模块找函数定义，不要以为它消失了。

### 5. 再看读取与单位

打开 `backend/eeg_readers.py`，读 `read_eeg_file`。BDF/EDF/GDF 的标准信号使用 MNE 公开读取器；FIF 转到 `eeg_fif.py`，保留原有坏道和注释语义。

MNE 的电压按 SI 伏特返回。乘 `1e6` 只是把 V 换成 µV，不是厂家校准，也不是额外除以 1000。无依据的设备倍率不能藏在读取代码里。

`eeg_models.py` 定义 `HeaderInfo` 和 `AnalysisSettings`，可以把它们当作“有固定栏目的一张信息表”。先认识字段名称即可，类型注解不是新的计算步骤。

### 6. 看预处理到底改了什么

打开 `backend/eeg_preprocessing.py`，读 `preprocess_eeg`。顺序是单位换算、分析采样率处理、带通/陷波、通道质控、时间片段质控。

`backend/eeg_mne_ops.py` 集中列出常用 MNE 参数。带通明确指定四阶 Butterworth、IIR、双向零相位和边缘填充；陷波保留 Q=30 的既定设计。参数显式写出，是为了避免重构时误用库的默认值。

FIF 的 BAD 区间在 `eeg_fif.py` 中分开处理：不跨 BAD 边界滤波；保留片段的原始索引，供事件和连续段分析使用。

**看懂数组的一个例子：**`data.shape == (5, 4000)` 表示 5 个通道、每个通道 4000 个点。若 `fs == 200`，每秒有 200 个点，总长就是 20 秒；2 秒片段有 400 点。`axis=1` 通常表示沿“时间”这一列方向计算。

### 7. 最后再读公式

标准方案在 `eeg_features.py`；论文固定 287 列方案在 `eeg_thesis_features.py`。两套口径有意分别保留，不把有不同统计定义的指标混在一起。

- 时域：先看 `_windowed_features` 或 `time_features`，从均值、标准差、RMS开始。
- 频域：看 `welch_psd` 和频段积分。PSD 是“每 Hz 的功率密度”，频段功率需要乘频率间隔再累加。
- 时频：论文方案用 MNE `average=None` 保留每个 Hann 窗口的 PSD，等价于同参数 STFT 的模平方密度。先加成频段功率时间序列，再求时间标准差和 CV。
- 脑网络：PLV 和阈值图指标按论文公式用 NumPy/SciPy 编写。先读输入矩阵的形状、范围和输出字段，再看矩阵乘法与最短路径。

`NaN` 表示不可计算或缺失，不是零。`True/False` 是“是/否”的标记；质控掩码中的 `True` 表示保留。`data[indices]` 表示按序号挑选行，不是自动改变采样率。

### 8. 跟着输出回到网页

`backend/eeg_reporting.py` 把已算出的结果做成图表和工作簿。HTTP 服务把结果打包成 JSON，其中有指标和图片链接；前端 `eeg_ui_common.js` 的 `render` 读取这些链接并显示图片。

画图不等于重新计算。调整波形纵轴、横轴或图片缩放，不应改动原始样本或 FIF。

## 主线看懂后，再读其他入口

| 功能 | 前端起点 | 后端起点 |
|---|---|---|
| 镇痛前后配对 | `eeg_pair_analysis.js` | `eeg_web_extra.py` → `eeg_pair_spike.py` |
| 多份文件和文件夹 | `eeg_session_ui.js`、`eeg_folder_import.js` | `eeg_session.py` |
| 事件标记 | `eeg_events_ui.js` | `eeg_events.py` |
| 原始检视、手工标记 | `eeg_raw_qc.js` | `eeg_raw_qc_api.py` |
| 黄色候选定位 | `eeg_artifact_patrol.js` | `eeg_artifact_patrol.py` |
| 绿色填入草稿 | `eeg_artifact_patrol.js` → `eeg_patrol_math.js` | 只改页面草稿，不发送 FIF 导出请求 |
| 原始频域查看 | `eeg_view_spectrum.js` | `eeg_view_spectrum.py` |
| 同目录 FIF 保存 | `eeg_raw_qc.js` | `eeg_raw_qc.py` → `eeg_local_fif.py` |

**保真兼容层放到最后再读。**`eeg_raw_bdf.py` 是有限的必要例外：原始查看与无损导出必须保留不同原采样率、原数字值及完整 Status。MNE Raw 要求同一采样率，混合读取会对低速信号上采样，所以这里保留单独的原生访问层；不能为了缩短代码让原始数据悄悄改变。该层不取代普通特征入口的 MNE 读取器。

## 你可以做的三个小练习

1. 在 HTML 找到“开始单份分析”，一直追到 Python 的 `analyze_eeg_file`，把途经的文件名写下来。
2. 用合成数据把显示窗长从 4 秒改成 2 秒，观察请求参数变化。它与分析片段长度是两个不同的设置。
3. 按黄色按钮，再按绿色按钮，找到 `buildDraft` 与 `applyDraft`。确认它们只是改草稿；只有正式导出入口才会生成 FIF。

初学时每个函数先回答三个问题：**拿进来什么？改了什么？交出去什么？**能回答这三个问题，就已经理解了主线。复杂公式可以稍后逐项核对。

## 维护代码

Python 使用 Black，前端使用 Prettier。它们只整理排版，不属于网站运行依赖。配置在 `pyproject.toml`、`.prettierrc.json`，测试在 `tests/`。

```powershell
python -m pip install -r requirements-dev.txt
python -m black backend tests tools
npm ci
npm run format:web
python -m unittest discover -s tests -p "test_*.py"
npm run test:web
```

核对 MNE 接口可查官方说明：[BDF 读取](https://mne.tools/stable/generated/mne.io.read_raw_bdf.html)、[IIR 滤波](https://mne.tools/stable/generated/mne.filter.filter_data.html)、[Welch PSD](https://mne.tools/stable/generated/mne.time_frequency.psd_array_welch.html)。项目固定 MNE 1.13.2 系列，并通过数值回归测试核对窗口、滤波和单位口径。
