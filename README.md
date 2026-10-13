# ComfyUI Qwen H3 Prompt

在 ComfyUI 内使用本地 Qwen3.8-27B 驱动 Minimax H3 官方 Skill 或自定义 Skill 生成 H3 提示词与制作方案。南无阿弥陀佛。
节点在 ComfyUI 内启动独立的本地 `llama-server`，推理过程不访问网络。运行完成后结束外部进程，释放GPU显存和系统内存（`force unload model`开启时），不占用ComfyUI内部资源。

节点加载时扫描 `skills/` 和 `custom_skills/` 目录并加载 Skill。

## 官方 Skill 的本地适配边界

`h3-prompt-writing` 官方声明可直接移植。其余 Skill 原本依赖 MiniMax Hub 的画布、选择卡和媒体生成工具。本节点把 Skill 内容作为本地模型指导，Skill 可以输出符合 H3 定义的 prompt。    

与 MiniMax H3 官方 Skills 不同，本节点所使用的 Qwen3.8-27B GGUF 模型不具备分析声音的能力，不能准确提取已有音乐的旋律、BPM、歌词、音色或节拍点。有关声音的提示词只是“创作声音的描述”，不是分析或复刻参考音频。    

输出格式和交互行为由所选 Skill 与用户请求决定。节点只执行一次本地推理，不会额外包装、改写或按 H3 结构修复模型输出。

## 安装节点
使用ComfyUI Manager安装， 或使用以下命令手动安装:
```commandline
cd ComfyUI/custom_nodes/
git clone https://github.com/chflame163/ComfyUI_Qwen_H3_Prompt.git
```

## 模型目录
把 Qwen3.8 视觉模型和GGUF权重文件放到```ComfyUI/models/LLM/```目录或其任意子目录（如```ComfyUI/models/LLM/Qwen3.8/```）均可。节点会递归扫描```ComfyUI/models/LLM```目录，自动检测其中的GGUF文件并填充到节点的```llm_model```和```vision_model```下拉列表：

* 文件名含 ```mmproj``` 的 GGUF 归入```vision_model```（视觉投影器），其余归入```llm_model```（语言模型）。
* 两个下拉框带官方刷新按钮，点击即可立即重新扫描本地目录，把新放入的GGUF文件更新到列表中，无需重启 ComfyUI。
* 通过后端实时扫描目录，目录内容变化（新增/删除模型）后列表会自动同步。

* 视觉权重：
  从 https://huggingface.co/unsloth/Qwen3.8-27B-GGUF 下载 ```mmproj-F16.gguf```
* GGUF权重：   
  对于20G以上显存的设备，推荐从 https://huggingface.co/unsloth/Qwen3.8-27B-GGUF 下载 ```Qwen3.8-27B-Q4_K_M.gguf```。    
  对于16G显存的设备，推荐 https://huggingface.co/soyaakinohara/qwen3.8-27b-abliterated-3.69bpw-12GB-MTP.gguf 。    
  也可以自行下载其他与llama-server兼容的Qwen3.8权重。

### 9B 模型（低显存推荐，8 GB 显存可流畅运行）

Qwen3.8-9B 是把 Qwen3.8-2.4T 的能力蒸馏到 Qwen3.5-9B 架构的模型，带图 H3 工作流建议优先使用它。需要两个文件，都放入 ```ComfyUI/models/LLM/```：

* 语言模型（约 5.4 GB）：从 https://huggingface.co/empero-ai/Qwen3.8-9B-GGUF 下载 ```Qwen3.8-9B-Q4_K_M.gguf```
  - 直接链接：https://huggingface.co/empero-ai/Qwen3.8-9B-GGUF/resolve/main/Qwen3.8-9B-Q4_K_M.gguf
* 视觉投影器 mmproj（约 879 MB）：下载 ```mmproj-BF16.gguf```，建议重命名为 ```mmproj-qwen3.5-9b-BF16.gguf```（与示例工作流一致）
  - ModelScope（国内下载快）：https://modelscope.cn/models/unsloth/Qwen3.5-9B-GGUF/file/view/master/mmproj-BF16.gguf
  - Hugging Face：https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/mmproj-BF16.gguf

## 示例工作流（Example Workflow）

`example workflow/` 目录提供可直接拖入 ComfyUI 画布运行的示例：

- `example workflow/Qwen H3 Prompt Example Workflow.json` —— 加载一张参考图，经 Qwen H3 节点生成 H3 提示词，并用 Preview 节点预览输出。
- `example workflow/Qwen H3 Prompt Example Workflow.png` —— 示例工作流的渲染预览图。

导入工作流时若节点未出现，请确认已安装本节点并刷新页面（Ctrl+F5）。参考图路径需按你本地实际文件调整。

## 安装llama-server runtime

项目只提供一个跨平台安装入口 `install_runtime.py`。它仅使用 Python 标准库，自动识别 Windows、Linux 或 macOS、x64/ARM64 架构及显卡，并为当前机器部署固定的 llama.cpp `b11115` runtime，不需要安装额外的 pip 包。NVIDIA 设备优先使用 `nvidia-smi` 报告的驱动 CUDA 兼容版本，`nvcc` 版本仅作为备用判断。

Windows ComfyUI 官方便携包在节点目录中执行：

```
cd ComfyUI\custom_nodes\ComfyUI_QWen_H3_Prompt
..\..\..\python_embeded\python.exe install_runtime.py
```

Linux 与 macOS 使用Python执行：

```
python install_runtime.py
```

首次启动前必须先执行安装器。安装器会依次完成硬件检测、后端选择、下载、SHA256 校验、安全解压、`llama-server --version` 与 `--list-devices` 验证，最后生成本机专用的 `runtime_config.json`。下载缓存、源码缓存和安装结果都保存在节点的 `runtime/` 目录中。

如果 Windows 报告 `WinError 5` 拒绝访问，先关闭正在运行的 ComfyUI，再使用当前 Windows 用户在节点目录重新执行 `python install_runtime.py --force`。节点启动 llama-server 时会把工作目录设在自定义节点目录，不依赖 runtime 文件夹作为当前目录；这样可以避开旧安装中 runtime 子目录继承的限制性 ACL。

安装器命令：

```
# 只显示检测结果和计划，不下载或安装
python install_runtime.py --dry-run

# 查看当前系统和架构支持的后端
python install_runtime.py --list-backends

# 手动指定后端
python install_runtime.py --backend vulkan

# 强制重新安装
python install_runtime.py --force

# 只使用 runtime/.downloads/b11115 中已经校验的文件
python install_runtime.py --offline
```


### 自动后端选择

| 系统和设备 | 自动顺序 |
| --- | --- |
| Windows x64 NVIDIA | 兼容 CUDA 13.4 时：CUDA 13.4 → CUDA 12.4 → Vulkan → CPU；否则从 CUDA 12.4 开始 |
| Windows x64 AMD | ROCm 10.0 → Vulkan → CPU |
| Windows x64 Intel | SYCL → Vulkan → CPU |
| Windows x64 未识别 GPU | Vulkan → CPU |
| Windows ARM64 NVIDIA | 兼容 CUDA 13.4 时：CUDA 13.4 → CPU；否则 CPU |
| Windows ARM64 Qualcomm Adreno | OpenCL → CPU |
| Linux x64 NVIDIA | 兼容 CUDA 13.4 时：CUDA 13.4 → CUDA 12.8 → Vulkan → CPU；否则从兼容版本开始 |
| Linux x64 AMD | ROCm 10.0 → Vulkan → CPU |
| Linux x64 Intel | SYCL FP16 → Vulkan → CPU |
| Linux x64 未识别 GPU | Vulkan → CPU |
| Linux ARM64 NVIDIA | 兼容 CUDA 13.4 时：CUDA 13.4 → Vulkan → CPU；否则 Vulkan → CPU |
| Linux ARM64 其他设备 | Vulkan → CPU |
| macOS Apple Silicon 或 Intel | Metal → CPU |

Windows 可用后端取决于架构，包括 `cuda12`、`cuda13`、`rocm`、`sycl`、`vulkan`、`opencl` 和 `cpu`。

Linux 官方预编译后端包括 `cuda12`、`cuda13`、`rocm`、`vulkan`、`sycl`、`sycl-fp32`、`openvino` 和 `cpu`，具体选项取决于架构。官方文件是 Ubuntu 构建，其他发行版需要兼容的 glibc/libstdc++。CUDA、ROCm、Vulkan、Intel SYCL/OpenVINO 还需要系统中已经存在相应的显卡驱动和运行库。

macOS 使用对应架构的官方包。Metal 不需要额外 GPU 计算框架，但 Qwen3.8-27B Q4 和视觉投影器仍需要足够的统一内存。

### Linux CUDA 部署

官方 `b11115` 提供 Ubuntu x64 CUDA 12.8/13.4 和 Ubuntu ARM64 CUDA 13.4 压缩包，自动模式会根据本机驱动兼容版本选择。也可显式要求自动选择 CUDA 预编译包：

```
python install_runtime.py --backend cuda
```

当没有兼容的官方包或需要本机构建时，可显式要求安装器从源码构建：

```
python install_runtime.py --backend cuda --build-from-source
```

源码构建不会调用 `sudo` 或自动修改系统。运行前必须已经具备：

- 可正常运行 `nvidia-smi` 的 NVIDIA 驱动；
- 包含 `nvcc` 的 NVIDIA CUDA Toolkit，`nvcc` 位于 `PATH` 中或已经设置 `CUDA_HOME`；
- Git、CMake、C 编译器和 C++ 编译器，例如 Ubuntu/Debian 的 `git cmake build-essential`。

### 运行时自动检测与回退（runtime detection）

节点不再强制依赖独立的 `llama-server`。启动时会自动检测当前机器可用的 llama.cpp 运行时，并按下述优先级选择；启动失败会自动降级到下一个可用运行时，全程在日志中记录选择结果：

1. **独立 llama-server（standalone，`--fit` 动态显存管理）**——默认首选。由 `install_runtime.py` 安装的官方 llama.cpp `b11115` 运行时，在低显存（如 8 GB）下通过把部分层溢出到系统内存来支撑大 context。H3 带图工作流需要 8K+ 上下文，而全量上 GPU 的后端受显存限制无法提供，故 standalone 是稳定首选。
2. **llama-cpp-python（CUDA，进程内推理）**——兜底。如果 `llama_cpp` 已安装到 ComfyUI 的 Python 环境（含 `ggml-cuda.dll`），节点直接用它加载 GGUF 与 mmproj（借用 `comfyui-minimax-h3-prompt-enhancer-T8` 的方案）。CUDA 快，但全量上 GPU 时 context 受显存限制；作为兜底会优先保证足够 context（必要时回退 CPU，慢但可用），而不是压到会触发 context-shift 报错的小值。
3. 两者都不可用时才报错，并给出安装指引。

几点说明：

- 检测结果可通过 `runtime_status()` 查看（`backend/version/gpu_offload/platform` 等），节点每次运行时也会在 `comfyui.log` 打印 `[Qwen H3] Runtime detection ...` 与 `[Qwen H3] Selected llama runtime: ...`。
- 走 llama-cpp-python 后端时，节点会读取 GGUF 头部元数据估算 KV 缓存规模，并保证有效 context 至少 16384（显存不足时允许回退 CPU 也要保住 context，避免 M-RoPE 禁 context-shift 导致的硬错误）。
- 同一后端按调用切换 `think_mode` 时，llama-cpp-python 会重建一次模型实例以保证思考/非思考输出正确（日志可见对应加载耗时）。
- 已检测到独立运行时但无需安装时，`runtime/.downloads` 下的已下载缓存可安全删除以释放磁盘空间。

### 节点 UI 推理参数（带帮助说明）

节点在 UI 上提供 6 个「最优默认」运行时选项，每个都带悬浮帮助说明（建议保持默认值）：

- **runtime_backend（推理后端）**：`auto`（默认，自动检测，优先独立 llama-server，其次 llama-cpp-python）、`vulkan`（强制独立 Vulkan 运行时，`--fit` 显存自适应，低显存推荐）、`cuda`（强制独立 CUDA 运行时，需先 `python install_runtime.py --backend cuda` 安装，未安装时自动回退到 llama-cpp-python 并告警）、`llama-cpp-python`（强制进程内 CUDA 后端）。新选项追加在输入列表末尾，旧工作流自动补默认值；为兼容旧工作流保存的错位布尔值，`True` 会被归一化为 `auto`。
- **context_size（上下文长度）**：默认 `32768`。H3 带图工作流需 8K+；standalone `--fit` 下可安全保持 32768，llama-cpp-python 会按显存自适应，设大值不会被浪费。
- **fit_enabled（显存自适应 --fit）**：默认开启。把部分层溢出到系统内存以支撑大 context，8GB 显存建议保持开启（仅独立运行时生效）。
- **fit_target_mib（--fit 显存目标）**：默认 `1536` MiB，8GB 显存推荐值（仅独立运行时生效）。
- **n_gpu_layers（GPU 卸载层数）**：默认 `auto`（由运行时按显存决定），可选 `-1`（全部层上 GPU，最快但显存占用最高）。
- **flash_attention（Flash Attention）**：默认 `auto`。注意：`off` 会同时关闭量化 KV 缓存（q8_0）以保持兼容，内存占用更高；`on/auto` 才启用量化缓存以支撑 8GB 卡的大 context（仅独立运行时生效）。

后端参数会覆盖 `runtime_config.json` 里的默认值；切换后端/参数时节点自动重建对应的 llama-server 实例。

### 节点底部快捷按钮

节点 UI 底部提供三个一键按钮（需刷新浏览器页面后出现）：

- **🔍 检测当前 runtime**：调用 `GET /qwen_h3/runtime/status`，弹出对话框显示当前已安装/可用的 llama.cpp 运行时信息——独立 runtime 后端（vulkan/cuda）、可执行文件与库的**有效路径**、`--fit`/`n_gpu_layers`/flash 参数、llama-cpp-python 版本与 GPU 卸载状态、`models/LLM` 下已扫描到的 gguf 清单。
- **🖥️ 安装 NV (CUDA) runtime**：调用 `POST /qwen_h3/runtime/install {backend:cuda}`，后台执行 `install_runtime.py --backend cuda`（联网下载 CUDA 独立运行时），按钮下方实时显示进度，完成/失败后弹窗给出结果与日志尾部。
- **🖥️ 安装 Vulkan runtime**：同上，后台执行 `install_runtime.py --backend vulkan`。

安装全程在后台线程执行，不阻塞 ComfyUI；安装状态可通过 `GET /qwen_h3/runtime/install/status` 查询。重复点击同类型安装时若已有任务在跑会返回冲突提示。



## 节点
启动ComfyUI，在 `😺dzNodes/QWen_H3_Prompt` 分类中添加 `Qwen H3 Prompt (Local)`节点。
![image](images/QwenH3Prompt_node.jpg)  

节点选项说明：
- `skill`：`auto`、内置 Skill 或 `custom_skills/` 中配置的第三方 Skill。设为auto时会在全部 Skill 中自动选择。
- `duration`：目标 H3 视频时长，默认 `10.0` 秒，取值范围4~15秒。
- `think_mode`：布尔开关，默认关闭；自动在 Qwen 官方 instruct 和 thinking 推荐参数之间切换。
- `reasoning_effort`：仅在 `think_mode` 开启时生效，默认 `medium`。
- `seed`：种子。
- `reference_images`：动态添加 0-9 张独立 IMAGE。每个插槽只接受一张图片；图片角色由数量和用户提示词共同决定。
- `reference_videos`：动态添加 0-3 路 IMAGE 批次，内部固定按 24 fps 计算时长和采样时间。
- `video_sample_frames_per_sec`：每路参考视频每秒抽取的帧数，默认值为 2。
- `force unload model`：默认开启。开启时每次完成后终止内置 llama-server服务并强制清理内存和显存；关闭时保留模型供下一次运行复用。发生错误时无论选项状态都会强制释放。

### 自定义 Skill

在 `custom_skills/` 下为 Skill 建立独立子目录，并至少放入一个 `SKILL.md`。
- Skill ID 只能使用小写字母、数字、点、下划线和连字符，不能使用 `auto`，也不能与 MiniMax H3 官方 skills 重名。
- 节点只把 skill 作为 Qwen 的提示词指导，不会执行第三方 Skill 声明的脚本、工具、网络调用、Canvas 流程或审批卡片。

### 图片输入与模式路由

节点按照以下规则选择 H3 模式：

| 输入 | 判定方式 |
| --- | --- |
| 没有图片和视频 | 直接使用 T2VA |
| 1 张图片、没有视频 | Qwen 根据用户提示词选择 I2VA、L2VA 或 Ref2VA |
| 2 张图片、没有视频 | Qwen 根据用户提示词选择 FL2VA 或 Ref2VA |
| 3-9 张图片 | 直接使用 Ref2VA |
| 连接任意参考视频 | 直接使用 Ref2VA |

一到两张图片时，节点不会仅根据图片数量猜测首尾帧。要生成关键帧模式提示词，必须在用户提示词中明确说明图片的时间角色：

```text
# 一张首帧图，路由为 I2VA
将 Picture 1 作为目标视频的首帧，从这个画面继续向前发展。

# 一张尾帧图，路由为 L2VA
将 Picture 1 作为目标视频最后一帧，设计一个自然抵达该画面的过程。

# 两张首尾帧图，路由为 FL2VA
Picture 1 是目标视频首帧，Picture 2 是目标视频尾帧，生成从前者连续过渡到后者的过程。
```

如果一到两张图片只是人物、产品、场景、构图或风格参考，也应明确说明其参考作用，节点会选择 Ref2VA：

```text
参考 Picture 1 的人物身份和服装，参考 Picture 2 的场景与灯光，创作一段产品宣传短片；两张图都不是首尾帧。
```

### 参考视频

`reference_videos` 最多连接 3 路，每一路都是按时间顺序排列的 IMAGE batch。每一路至少包含 1 帧，按 24 fps 计算后时长不得超过 15 秒；全部参考视频总时长不能超过 15 秒。

`video_sample_frames_per_sec` 表示从每路视频的每一个一秒时间窗中均匀抽取多少帧，它不是整段视频的总抽帧数，也不会改变最终 H3 视频的帧率。

默认值 `2` 适合大多数动作和运镜。静态产品展示可以使用 `1`；快速动作、复杂转场可提高到 `3-8`。数值越高，每秒保留的动作细节越多，但联系表中单帧面积会变小，同时增加图片编码、内存和推理开销。

连接参考视频后始终使用 Ref2VA。当前接口只接收视频画面帧，不会读取视频文件中的音轨。

### 原创声音与配乐描述

Qwen 会根据用户提示词、目标时长、参考图片以及参考视频的抽样画面，自动创作与画面匹配的声音设计：

- 描述环境声、动作声、空间感和随画面变化的声音事件。
- 描述原创器乐配乐的情绪、乐器、速度、发展、关键画面同步点和收束方式。
- 没有用户明确提供时，不会自行编造对白、旁白、歌词或人声演唱。
- 用户只要求“无背景音乐”时，输出 `non_diegetic_music: N/A`，但仍保留合理的环境声和动作声。
- 用户明确要求“完全静音”“全程静音”或“不要任何声音”时，强制输出 `overall_soundscape: N/A` 和 `non_diegetic_music: N/A`。

这属于根据视觉内容创作新的音频描述，不是分析、复刻或保留参考视频原声。生成的文本最终交给 MiniMax H3 合成原生声音。


## 显存和生命周期
若关闭 `force unload model`，内置服务会继续占用约 20 GB 显存，模型配置不变时下一次可直接复用；再次开启并运行节点、模型配置发生变化或退出 ComfyUI 时会终止驻留服务。

使用 llama-cpp-python 后端时，驻留占用取决于所选模型与封顶后的 context（例如 8 GB 显存的 RTX 2070 Super 加载 Qwen3.8-9B-Q4_K_M + mmproj，模型+KV 约占用 6.2 GB）；`force unload model` 关闭时可复用驻留实例以加快后续运行。


## 上游来源与许可证

- MiniMax H3 Skill 来自 [`MiniMax-AI/MiniMax-H3`](https://github.com/MiniMax-AI/MiniMax-H3) commit `d21241f0a4b3acbb34c97dae47fa417b7065e438`，官方文件保存在 `skills/`。
- 跨平台runtime安装器固定使用 llama.cpp `b11115`，对应完整 commit `d5f66492e661b63e6c74822c2b72f5146053994e`，并对下载文件执行 SHA256 校验。
- Linux CUDA 源码构建使用同一固定 tag 和 commit。llama.cpp 的 MIT 许可证保存在 `third_party/llama.cpp-LICENSE`。
- 本项目仅为学习研究使用，如果作为商业用途，请查阅相关原项目授权协议。
- 南无阿弥陀佛
