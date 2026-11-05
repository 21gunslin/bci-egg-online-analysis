# 网页启动与恢复

`127.0.0.1:8765`是本机Python服务地址。电脑重启后，原进程会结束；必须在登录后重新运行服务，浏览器标签页本身不会启动Python。

双击`scripts/安装脑电网页登录自启.bat`可安装当前用户的Windows计划任务`LocalEEGWeb8765`：登录时启动，并每分钟检查巡检进程；后台巡检每15秒检查网站，进程退出后重启。计划任务不可用时使用当前用户启动文件夹快捷方式，无需账户密码。仅监听127.0.0.1。

自动安装在本次代理执行时被自动审批拒绝，返回“blocked by policy”；脚本已准备好，但登录自启尚未安装，需要用户手动运行一次。不能把静态脚本检查或当前服务启动当成重启后验证。

普通使用双击`scripts/启动脑电特征网页.bat`。启动成功后窗口可关闭；重复启动有互斥锁，不重复启动巡检。健康检查验证返回的是本网站，使用前检测Python依赖。服务启动错误保存在`results/eeg_web_output/supervisor.log`和`server_error.log`，前一轮日志保留为`.previous`。

手动停止双击`scripts/停止脑电特征网页.bat`，同时停止巡检并保留停止标记，避免立即自动拉起；再次点击启动会恢复。只结束命令行与本网站路径相符的进程，不结束其他Python程序。

端口仍在监听但健康检查暂时失败时，只记录日志，不强制中止可能仍在进行的长时间分析。已运行进程崩溃后的网页请求或在途分析仍需重新提交，巡检不能恢复内存中的计算任务。

检查状态：`powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/eeg_web_service.ps1 -Status`。

卸载自启：在Windows任务计划程序中删除`LocalEEGWeb8765`；若使用备用启动快捷方式，按Win+R输入`shell:startup`，删除`Local EEG Web.lnk`。停止后不会删除原始数据或已有分析结果。
