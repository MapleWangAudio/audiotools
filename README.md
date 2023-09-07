# 安装

先自行根据自己设备情况安装相对应的pytorch（本项目基于python3.10，pytorch2.0进行的构建，其他版本请自行检查兼容性）

然后

```
pip install -r requirements.txt
```

或者

```
conda install -c pytorch -c defaults --file requirements.txt
```

requirements_add.txt中有一些常用的额外的库，可以选择性安装。当然，不安装add也能正常使用本库

```
conda install --file requirements_add.txt
```

# 使用方法

放在工程目录下，然后

```
from audiotools import analysis as A
from audiotools import process as P
```

# 版本号定义

稳定性.新py文件.新方法的加入.日常维护和bug修正

稳定性解释：

    v：正式版本，不出意外的话，稳定可用

    t：过程版本，有大量未解决的问题，不可正常使用

# 规定

1. 输入输出使用tensor，格式为[ 通道，数据 ]，推荐使用torchaudio读取
2. 所有方法都设计为一次处理一整个音频，而非一个点一个点的处理
3. 所有方法都应该尽量支持多声道处理

# 各个py文件说明

process：各种处理方法

analysis：分析绘图&特征提取

# 常见问题

## 关于matplotlib不显示：

wsl可能没有gui，如果想要gui显示，需要下载gui支持，可以尝试：

sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev

这里使用savefig解决

## 将数据放在gpu上运行的方法：

waveform, sample_rate = torchaudio.load('audio.wav')

waveform_gpu = waveform.to(torch.device('cuda'))
