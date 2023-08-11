# 安装

先自行根据自己设备情况安装相对应的pytorch（基于python3.10，pytorch2.0进行的构建）

然后

```
pip install -r requirements.txt
```

或者

```
conda install --yes --file requirements.txt    #这种执行方式，一遇到安装不上就整体停止不会继续下面的包安装
```

# 使用方法

放在工程目录下，然后：

import audiotools as at

# 版本号定义

稳定性.新py文件.新方法的加入.日常维护和bug修正

稳定性解释：

    v：正式版本，不出意外的话，稳定可用

    t：过程版本，有大量未解决的问题，不可正常使用

# 规定

1. 函数输入输出使用tensor
2. audioload使用torchaudio和librosa格式：[通道，数据]。请勿使用scipy的读取

# 各个py文件说明

dp：data processor>修剪音频文件，转换格式等数据操作

dsp：信号处理>基础类型效果器

analysis：分析绘图&特征提取

# 常见问题

## 关于matplotlib不显示：

wsl可能没有gui，如果想要gui显示，需要下载gui支持，可以尝试：

sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev

这里使用savefig解决

## 将数据放在gpu上运行的方法：

waveform, sample_rate = torchaudio.load('audio.wav')

waveform_gpu = waveform.to(torch.device('cuda'))
