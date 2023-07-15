# 使用方法

import audiotools

# 版本号定义

稳定性.新py文件.新方法的加入.日常维护和bug修正

稳定性解释：

    v：正式版本，不出意外的话，稳定可用

    t：过程版本，有大量未解决的问题，不可正常使用

# 规定

1. 输入输出使用tensor
2. audioload使用torchaudio和librosa格式：[通道，数据]。请勿使用scipy的读取

# 各个py文件说明

data：数据处理——修剪音频文件，转换格式之类的

signal：信号处理——基础效果器

analysis：各种分析绘图

# 常见问题

## 关于matplotlib不显示：

wsl可能没有gui，如果想要gui显示，需要下载gui支持，可以尝试：

sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev

这里使用savefig解决

## 将数据放在gpu上运行的方法：

waveform, sample_rate = torchaudio.load('audio.wav')

waveform_gpu = waveform.to(torch.device('cuda'))
