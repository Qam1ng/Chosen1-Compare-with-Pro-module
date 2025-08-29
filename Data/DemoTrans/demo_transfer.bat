@echo off
setlocal enabledelayedexpansion

:: =================================================================
:: CS2 DEMO 转 MP4 自动化脚本
:: =================================================================

:: ------------------- 配置区域 (根据情况修改) -------------------

:: 1. 设置 CS2 安装路径下的 cs2.exe 的完整路径
set CS2_PATH="C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\bin\win64\cs2.exe"

:: 2. 设置 FFmpeg.exe 的完整路径 (如果已经添加到环境变量Path，则保留 "ffmpeg" 即可)
set FFMPEG_PATH="ffmpeg"

:: 3. 设置存放所有 DEMO 文件的文件夹路径 (!!!)
set DEMO_FOLDER_PATH="C:\Users\YourUser\Desktop\demos"

:: 4. 设置输出 MP4 视频的文件夹路径 (脚本会自动创建此文件夹)
set OUTPUT_FOLDER_PATH="C:\Users\YourUser\Desktop\rendered_videos"

:: 5. 设置视频参数
set FPS=15
:: 录制的帧率 (FPS)，建议15
set RESOLUTION_W=1920
set RESOLUTION_H=1080
:: 录制的分辨率 (建议 1920x1080)

:: 定义CS2游戏主目录
set CS2_GAME_DIR="C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo"

:: ------------------------ 脚本执行区域 (无需修改) ------------------------



echo.
echo [INFO] CS2 DEMO Conversion Program
echo.

:: 检查必要的文件和路径
if not exist %CS2_PATH% (
    echo [ERROR] Can't find cs2.exe! Please check CS2_PATH setting.
    pause
    exit /b
)
if not exist "%DEMO_FOLDER_PATH%" (
    echo [ERROR] Can't find DEMO folder! Please check DEMO_FOLDER_PATH setting.
    pause
    exit /b
)

:: 如果输出目录不存在，则创建
if not exist "%OUTPUT_FOLDER_PATH%" (
    echo [INFO] Output folder does not exist, creating: %OUTPUT_FOLDER_PATH%
    mkdir "%OUTPUT_FOLDER_PATH%"
)

:: --- 主循环 ---
echo [INFO] Scanning folder: %DEMO_FOLDER_PATH%
set /a count=0
for %%f in ("%DEMO_FOLDER_PATH%\*.dem") do set /a count+=1
echo [INFO] Found %count%  .dem files. Begin Processing...
echo.

set /a current=0
for %%f in ("%DEMO_FOLDER_PATH%\*.dem") do (
    set /a current+=1
    set "DEMO_FILENAME=%%~nxf"
    set "DEMO_NAME_ONLY=%%~nf"

    echo.
    echo +------------------------------------------------------------------+
    echo ^| Processing NO. !current! / %count% file: "!DEMO_FILENAME!"
    echo +------------------------------------------------------------------+
    echo.

    :: 1. 复制DEMO到游戏目录
    echo [STEP 1/5] Coping "!DEMO_FILENAME!" to game directory...
    copy "%%f" "%CS2_GAME_DIR%\" > nul
    if errorlevel 1 (
        echo [ERROR] 复制文件失败! 跳过此文件。
        goto :skip
    )

    :: 2. 启动 CS2 并开始录制
    echo [STEP 2/5] 正在启动 CS2 并录制帧...
    start "" /wait %CS2_PATH% -insecure -window -w %RESOLUTION_W% -h %RESOLUTION_H% +sv_cheats 1 +host_framerate %FPS% +startmovie "temp_rec_output" tga +demo_play "!DEMO_NAME_ONLY!; quit"
    echo [INFO] CS2 已关闭，录制完成。

    :: 检查录制文件是否存在
    if not exist "%CS2_GAME_DIR%\temp_rec_output0000.tga" (
        echo [ERROR] Record Fails，NO TGA File! Skip this file.
        goto :cleanup_copy
    )

    :: 3. 使用 FFmpeg 合成视频
    echo [STEP 3/5] Using FFmpeg in integrating videos...
    set "OUTPUT_MP4_PATH=%OUTPUT_FOLDER_PATH%\!DEMO_NAME_ONLY!.mp4"
    %FFMPEG_PATH% -y -framerate %FPS% -i "%CS2_GAME_DIR%\temp_rec_output%%04d.tga" -i "%CS2_GAME_DIR%\temp_rec_output.wav" -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -c:a aac -b:a 192k "!OUTPUT_MP4_PATH!"

    if errorlevel 1 (
        echo [ERROR] FFmpeg Fails!
    ) else (
        echo [SUCCESS] 视频合成成功! 文件已保存为: "!OUTPUT_MP4_PATH!"
    )
    
    :: 4. 清理TGA和WAV临时文件
    echo [STEP 4/5] Cleaning temp files in recording process...
    del "%CS2_GAME_DIR%\temp_rec_output*.tga" > nul
    del "%CS2_GAME_DIR%\temp_rec_output.wav" > nul

    :cleanup_copy
    :: 5. 清理复制到游戏目录的DEMO文件
    echo [STEP 5/5] Cleaning DEMO copies in game directory...
    del "%CS2_GAME_DIR%\!DEMO_FILENAME!" > nul
    
    :skip
)

echo.
echo +------------------------------------------------------------------+
echo ^| All Demos are processed!
echo +------------------------------------------------------------------+
echo.
pause