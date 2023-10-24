# 介绍

音频效果研究中, 常规的语音库并不能很好的进行相关研究, 所以构建了本库, 作为常见音频库的补充, 以便于相关研究.

大多数情况下, 我们以以下的原则进行使用:

    1. 使用torchaudio进行读取

    2. 实时处理时, 优先使用本库方法

    3. 离线处理时, 优先使用torchaudio方法

    4. 对于本库和torchaudio没有的方法, 使用librosa

# 项目结构

process：各种处理方法. 输入输出为一个点

analysis：分析绘图&特征提取. 输入输出为一个点或一个数组, 数组格式为[通道, 数据]

# 版本号定义

稳定性.新py文件.新方法的加入.日常维护和bug修正

稳定性解释：

    v：正式版本, 不出意外的话, 稳定可用

    t：过程版本, 有大量未解决的问题, 不可正常使用

# 安装

**警告：** 

下列命令只能安装cpu版本的pytorch，如果需要gpu版本，请先参考[pytorch官网](https://pytorch.org/get-started/locally/)进行安装，再运行下列命令.

使用:

```
pip install -r requirements.txt
```

或者

```
conda install -c pytorch -c defaults -c conda-forge --file requirements.txt
```

# 使用方法

放在工程目录下, 与main.py同级, 然后

```
from audiotools import *
```

不同级的话, 需要在上列代码之前额外添加如下代码

```
import sys
sys.path.append("/your/audiotools/path")
# 如audiools在/home/user/audiotools
# 则为/home/user即可
# 当然, 也可以使用相对路径
```

为了方便使用，audiotools的__init__.py中已经将常用的库进行了导入，可以直接使用. 包括了如下库：

```
import torch
import torchaudio
import torchaudio.functional as F
import torch.utils.tensorboard as tb
import math
import matplotlib.pyplot as plt
import multiprocessing
import librosa
import numpy as np
import scipy.signal as signal
import scipy.optimize as opt

from tqdm import tqdm
from . import analysis, process
```

如果不喜欢这类写法，可以进入__init__.py，将不需要的库注释掉.

# 常见问题

## 关于matplotlib不显示

wsl可能没有gui, 如果想要gui显示, 需要下载gui支持, 可以尝试：

```
sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
```

## 使用line_profiler进行性能分析

这个可以对函数进行每行的性能分析, 如果需要函数整个运行的性能分析, 可以使用visual studio或者pycharm的性能分析器

```
# 把下行代码加入到需要测试的函数上
@profile
def profile():
    pass
# 进入终端，进行测试 
kernprof -l profile.py
# 读取测试结果
python -m line_profiler profile.py.lprof
```

## 使用torch时将数据放在gpu上运行的方法：

``````
# 检查是否有可用的 GPU 设备
if torch.cuda.is_available():
    # 设置默认的 GPU 设备为 GPU 0
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# 将 tensor 移动到设备 device 上
test = test.to(device)
``````

## 以float64计算的方法:

对音频处理, float32不太够用, 同时float64时numpy的计算会更快, 所以尽量使用float64.

如需在神经网络中使用64bit, 则在主函数的开头加入:

```
torch.set_default_dtype(torch.float64)
```

这将全局指定tensor的精度为float64, 使得所有的tensor都以64位进行计算.

但对于torchaudio的读取, 由于其本身的限制, 即算使用了上述代码全局指定了精度, 仍无法读取为64bit, 只能读成32bit, 这会导致后继所有的基于input初始化的数据都为32bit. 所以我们可以显式的指定读取的格式为64bit:

``` 
input, sr = torchaudio.load(input_path)
input = input.double()
```