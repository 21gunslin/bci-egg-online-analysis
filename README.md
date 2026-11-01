# 本机脑电研究工作台

在本机浏览器分析 BDF / EDF / GDF；Python 服务仅监听 `127.0.0.1:8765`。

## 功能

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

已检查 Python / JavaScript 语法、合成窄峰与后续慢波候选、相同记录的零差值比较、本机 API 与结果下载、浏览器图表加载及点击放大。功能验证不代表临床预测有效性。

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
