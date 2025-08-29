找到你的 CS2 CFG 文件夹，通常路径是：
C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg

在此文件夹内，新建一个文本文档，将下面的代码文件"record_cfg.cfg"粘贴进去 


安装 FFmpeg: 确保你已经下载并正确配置了 FFmpeg。最简单的方法是将其 bin 目录添加到系统环境变量 Path。

放置 Demo 文件: 将你想要转换的 .dem 文件（例如 mydemo.dem）放到 CS2 的 .../game/csgo/ 目录下。

配置脚本: 用记事本或任何文本编辑器打开你创建的 demo_transfer.bat 文件。仔细修改**配置区域**的设置：

CS2_PATH: 确认你的 cs2.exe 路径是否正确。

DEMO_NAME: 输入你的 demo 文件名（不带 .dem 后缀）。

OUTPUT_NAME: 你希望生成的 MP4 文件叫什么名字。

FPS, RESOLUTION_W, RESOLUTION_H: 设置视频帧率和分辨率。

运行脚本: 保存对 demo_transfer.bat 的修改，然后运行。