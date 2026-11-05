"""MNE启动配置：把这个网站的配置放在项目results里，不改用户桌面MNE设置。

Keep this app's MNE configuration separate from desktop/user MNE settings."""

import os
from pathlib import Path


def load_mne():
    config = Path(__file__).resolve().parents[1] / "results" / "eeg_mne_config"
    if not os.environ.get("_MNE_FAKE_HOME_DIR"):
        config.mkdir(parents=True, exist_ok=True)
        os.environ["_MNE_FAKE_HOME_DIR"] = str(config)
    import mne

    return mne
