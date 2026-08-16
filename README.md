# video2knowledge-doubao

基于豆包（火山引擎）生态的「视频 → 知识」工具：把一段教学/录屏视频自动转换为带时间戳的字幕、融合语音+板书的完整转写，以及可直接用于复习的结构化知识文档（摘要 / 时间线 / 核心要点 / 问答 / 术语表）和问答闪卡。

支持两种输入：

- **本地视频文件**（`--video <path>`）
- **B 站客户端缓存**（`--cid <cid>`，m4s 音视频流自动无损合并封装为 mp4）

## 流水线总览

```
输入 ──┬─ 本地视频 (--video)
      └─ B 站客户端缓存 (--cid) → convert_bilibili_cache.py 无损合并为 mp4
           │
           ├─ (path 1/3) extract_frames.py 帧提取
           │    └─ (可选 --preprocess) preprocess_frames.py 黑板检测/裁剪/增强
           │
           ├─ (path 2/3) ASR：ffmpeg 提取 16kHz WAV → 上传 TOS 取预签名 URL
           │             → 火山 SeedASR 提交/轮询 → subtitles.{srt,vtt,json}
           │
           └─ (path 1/3) OCR：逐帧 → 豆包视觉模型 doubao-seed-2-1-pro
                         → captions.{srt,json}（板书逐字转写 / 画面描述）
                 │
                 ├─ (path 3) merge_visual.py 按时间戳融合 → merged.json
                 │
                 ├─ (可选 --full-transcript) full_transcript.py 豆包融合 语音+板书
                 │                         → full_transcript.{json,srt,md}
                 │
                 └─ build_knowledge.py 豆包生成知识文档 → knowledge.md + cards.csv
```

三条处理路径（`--path`）：

| path | 含义 | 调用的付费能力 | 主要产出 |
|---|---|---|---|
| 1 | 纯画面 OCR：只看板书/画面文字 | OCR（豆包视觉模型） | `captions.*` + `knowledge.*` |
| 2 | 语音转写：只听音频 | ASR（火山 SeedASR） | `subtitles.*` + `knowledge.*` |
| 3 | **融合模式（默认）**：语音 + 画面并行处理后再融合 | ASR + OCR + 知识生成 | `subtitles.*` + `captions.*` + `merged.json` + `knowledge.*` |

> **费用提示**：ASR、OCR、完整转写与知识生成均调用豆包/火山引擎付费接口（按音频时长、图片张数、Token 计费）。`--path 3` 一次会同时触发 ASR + OCR + 知识生成三类调用；帧提取、预处理、缓存转 mp4 等本地操作免费。

## 快速开始

### 1. 安装依赖

要求 `python3 >= 3.10`。

```bash
pip install -r requirements.txt
```

关于 ffmpeg/ffprobe：

- 项目不要求系统安装 ffmpeg。`bin/` 目录下自带 `ffmpeg`（软链到 imageio-ffmpeg 提供的二进制）和 `ffprobe`（bash 包装脚本），`config.py` 会把 `bin/` 自动加入 PATH。
- 若换了机器/重装后 `bin/ffmpeg` 软链失效，重建即可：

```bash
ln -sf "$(python3 -c 'import imageio_ffmpeg,os;print(imageio_ffmpeg.get_ffmpeg_exe())')" bin/ffmpeg
```

### 2. 配置 .env

```bash
cp .env.example .env
```

填入火山引擎 / 豆包凭证（各变量含义见「环境变量配置说明」）。缺关键配置时 ASR/OCR 脚本启动即报错并提示缺哪些变量；知识生成 / 完整转写脚本缺 Key 时自动降级为本地拼接（仍会生成文档，内容为原始字幕回退）。

### 3. 一条命令跑完整流水线（融合模式）

```bash
python3 scripts/run_pipeline.py --video lecture.mp4 --out-dir output --path 3 --preprocess
```

## 使用示例

### 本地视频 → 融合模式（推荐）

```bash
python3 scripts/run_pipeline.py --video lecture.mp4 --out-dir output --path 3 --preprocess
```

### 本地视频 → 融合模式 + 完整转写

```bash
python3 scripts/run_pipeline.py --video lecture.mp4 --out-dir output --path 3 --preprocess --full-transcript
```

### 语音转写（只听音频）

```bash
python3 scripts/run_pipeline.py --video lecture.mp4 --out-dir output --path 2
```

### 纯画面 OCR

```bash
python3 scripts/run_pipeline.py --video lecture.mp4 --out-dir output --path 1 --preprocess
```

### B 站客户端缓存 → 融合模式

B 站桌面/移动端缓存的视频是音视频分离的 m4s 流（缓存头部带占位前缀，ffmpeg 不能直接解析）。先列出本机缓存，再 `--cid` 一键处理（流水线自动合并封装为 mp4 后继续）：

```bash
# 列出所有缓存（含标题、时长、是否完整：ok = 音视频完整可转换）
python3 scripts/convert_bilibili_cache.py --list

# 一键处理指定缓存（cid 即缓存目录名）
python3 scripts/run_pipeline.py --cid 25685856471 --out-dir output --path 3 --preprocess
```

也可以单独把缓存转成 mp4（`-c copy` 无损封装，不重新编码，43 分钟视频约 1 秒内完成）：

```bash
python3 scripts/convert_bilibili_cache.py --cid 25685856471 --out-dir output
python3 scripts/convert_bilibili_cache.py --all --out-dir output
```

### 各 --path 的含义

- **path 1**：只做画面 OCR（板书/录屏文字），不做语音；
- **path 2**：只做语音转写（ASR），`--preprocess` 等画面相关参数不生效；
- **path 3**（默认）：ASR 与 OCR **并行**执行，`merge_visual.py` 按时间戳融合，知识文档同时使用语音与画面信息；
- `--full-transcript` 仅对 path 2/3 生效（path 3 时基于 `merged.json`，path 2 时基于 `subtitles.json`）。

### 单步执行等价于

```bash
python3 scripts/asr_doubao.py --video lecture.mp4 --out-dir output --language zh-CN
python3 scripts/ocr_doubao.py --video lecture.mp4 --out-dir output --prompt-ocr
python3 scripts/merge_visual.py --subtitles output/subtitles.json --visual output/captions.json
python3 scripts/build_knowledge.py --subtitles output/subtitles.json --merged output/merged.json --out-dir output
```

## run_pipeline.py 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--video <path>` | 二选一必填 | 本地视频文件（与 `--cid` 互斥） |
| `--cid <cid>` | 二选一必填 | B 站客户端缓存目录名（cid），自动合并封装为 mp4 后处理 |
| `--cache-dir <path>` | `/Users/duanp/Movies/bilibili` | `--cid` 模式下缓存根目录 |
| `--out-dir <path>` | `output/` | 输出基础目录；实际产物在 `<out-dir>/<视频名>/` |
| `--path 1\|2\|3` | `3` | 处理路径：1=纯画面OCR，2=语音转写，3=融合模式 |
| `--language <code>` | `zh-CN` | ASR 语言代码（`zh-CN`/`en-US`/`ja-JP` 等） |
| `--density-sampling-fps <fps>` | `1.0` | density 模式密集采样率（fps） |
| `--density-floor <%>` | `0.3` | 空板密度下限：板区密度低于该值视为空板，一律不保留 |
| `--density-min-increment <%>` | `0.35` | 保留触发：密度相对上一保留帧净增该值个百分点即保留 |
| `--density-fingerprint-hamming <n>` | `10` | 指纹触发：密度相近但指纹汉明距离 >= 该值即保留 |
| `--density-min-interval <sec>` | `12.0` | 保留帧最小间隔秒数 |
| `--density-erase-drop <%>` | `2.0` | 擦板检测：密度相对上一保留帧骤降该值个百分点视为擦板 |
| `--density-erase-recover <frac>` | `0.7` | 擦板恢复系数：回升至擦前密度该比例即强制保留一帧 |
| `--density-bright-threshold <n>` | `200` | 亮像素阈值（0-255）：板区灰度高于该值计为板书/粉笔像素 |
| `--density-min-frames <n>` | `15` | 低对比度兜底：一次提取帧数低于该值且视频较长时自动降阈值重跑 |
| `--density-max-gap <sec>` | `300.0` | 静止有板期兜底：相邻保留帧间隔超过该秒数时补入采样帧（0=关闭） |
| `--max-frames <n>` | `120` | density 模式最大保留帧数上限（每帧对应一次 OCR 计费，是费用上限） |
| `--preprocess` | 关 | OCR 前预处理帧：检测黑板/白板区域 + 裁剪 + CLAHE 增强 |
| `--no-crop` | 关 | 配合 `--preprocess`，只增强不裁剪 |
| `--caption` | 关 | OCR 使用画面描述模式；默认（不指定）为黑板板书逐字转写 |
| `--full-transcript` | 关 | 融合后额外调用豆包大模型生成完整转写 `full_transcript.{json,srt,md}`（仅 path 2/3） |

> **注意**：流水线默认 OCR 是「逐字转写黑板板书」，加 `--caption` 改为「描述画面内容」。单独运行 `ocr_doubao.py` 时默认恰恰相反（画面描述），需加 `--prompt-ocr` 才是逐字转写板书。

## 脚本清单

| 脚本 | 职责 | 是否调用付费 API |
|---|---|---|
| `run_pipeline.py` | 一键流水线入口：取帧 → 并行 ASR+OCR → 融合 → 知识文档 | 是（视 path） |
| `convert_bilibili_cache.py` | B 站客户端缓存 m4s → mp4（无损 `-c copy`） | 否 |
| `extract_frames.py` | 帧提取：density 文字密度增量采样（板区密度为主 + 内容指纹为辅） | 否 |
| `preprocess_frames.py` | 黑板/白板区域检测、裁剪、CLAHE 增强 + 降噪 + 锐化 | 否 |
| `asr_doubao.py` | 火山 SeedASR 录音文件识别：提取 16kHz WAV → TOS → submit/query 轮询 → `subtitles.*` | 是（ASR） |
| `ocr_doubao.py` | 豆包视觉模型逐帧 OCR / 画面描述 → `captions.*` | 是（OCR） |
| `merge_visual.py` | 按时间戳融合 ASR+OCR → `merged.json` | 否 |
| `full_transcript.py` | 豆包大模型融合语音+板书生成完整转写 → `full_transcript.{json,srt,md}` | 是（Token） |
| `build_full_transcript.py` | 纯本地拼接完整课程文档（语音为主时间线 + 板书内联）→ `full_transcript.md` | 否 |
| `build_knowledge.py` | 豆包生成知识文档（摘要/时间线/要点/问答/术语）→ `knowledge.md` + `cards.csv` | 是（Token） |
| `tos_upload.py` | 上传文件到 TOS 生成预签名 URL（ASR/OCR 用） | 否（仅存储/流量成本） |
| `analyze_frame_density.py` | 分析板区文字密度分布，辅助标定 density 取帧阈值（floor/increment） | 否 |
| `config.py` | 从 `.env` 加载环境变量 + 配置校验 + 把项目 `bin/` 加入 PATH | 否 |

## 环境变量配置说明

复制 `.env.example` 为 `.env` 后填写，脚本自动从项目根目录加载 `.env`。

| 变量 | 说明 | 必填 |
|---|---|---|
| `VOLC_ACCESS_KEY` | 火山引擎 Access Key ID（TOS 上传用） | 是 |
| `VOLC_SECRET_KEY` | 火山引擎 Secret Access Key | 是 |
| `ASR_X_API_KEY` | 豆包语音 ASR 的 API Key（新版控制台） | path 2/3 必填 |
| `ASR_RESOURCE_ID` | ASR 资源 ID。1.0 录音文件识别为 `volc.bigasr.auc`；若控制台开通 2.0 改 `volc.seedasr.auc` | 选填（默认 `volc.bigasr.auc`） |
| `TOS_BUCKET` | TOS 桶名（ASR 上传音频用） | path 2/3 必填 |
| `TOS_ENDPOINT` | TOS endpoint | 选填（默认 `tos-cn-beijing.volces.com`） |
| `TOS_REGION` | TOS region | 选填（默认 `cn-beijing`） |
| `ARK_API_KEY` | 火山方舟 API Key | path 1/3 必填 |
| `ARK_EP_ID` | 视觉模型接入点 `ep-xxx`（doubao-seed-2-1-pro，OCR 用；同时作为知识生成的兜底文本模型） | path 1/3 必填 |
| `ARK_TEXT_EP_ID` | 纯文本模型接入点（知识生成/完整转写用，可设独立更便宜的文本模型） | 选填 |
| `ARK_BASE_URL` | 方舟 API base URL（OCR 用 Responses API） | 选填（默认 `https://ark.cn-beijing.volces.com/api/v3`） |
| `ARK_URL` | 兼容旧配置的完整 chat/completions 地址（知识生成用） | 选填（默认由 `ARK_BASE_URL` 派生） |

## 常见问题

### B 站视频怎么处理？为什么没有 `--url`？

当前版本输入为「本地视频文件」或「B 站**客户端缓存**」。B 站客户端/网页缓存的 m4s 流先用 `convert_bilibili_cache.py` 无损合并为 mp4，再走 `--video` / `--cid` 流水线。未登录状态缓存的通常只有低清晰度（B 站 1080P 需登录会员）。

### 为什么 OCR / 文本模型要禁用 thinking？

`doubao-seed-2-1-pro` 默认开启深度思考，在逐帧 OCR、逐时间窗转写这类批处理场景下，不关闭会导致单帧/单窗耗时 10 分钟以上。项目在 `ocr_doubao.py`、`full_transcript.py`、`build_knowledge.py` 中均显式传入 `"thinking": {"type": "disabled"}`。

### ASR 报错 "requested resource not granted"

`ASR_RESOURCE_ID` 与火山引擎控制台实际开通的版本不一致：

- 开通的是录音文件识别 **1.0** → `volc.bigasr.auc`
- 开通的是 **2.0** → `volc.seedasr.auc`

按控制台开通的版本修改 `.env` 中的 `ASR_RESOURCE_ID` 即可。

### ffmpeg / ffprobe 找不到

- 确认已执行 `pip install -r requirements.txt`（imageio-ffmpeg 会安装 ffmpeg 二进制）；
- `bin/ffmpeg` 软链指向 imageio-ffmpeg 的二进制，换机器/重装后可能失效，重建软链：

```bash
ln -sf "$(python3 -c 'import imageio_ffmpeg,os;print(imageio_ffmpeg.get_ffmpeg_exe())')" bin/ffmpeg
```

- 也可在系统安装 ffmpeg：`config.py` 优先使用 `bin/`，其次使用系统 PATH 中的命令。

### TOS 上传失败

检查 `.env` 中 `TOS_BUCKET` / `TOS_ENDPOINT` / `TOS_REGION` 与 `VOLC_ACCESS_KEY` / `VOLC_SECRET_KEY` 是否正确，确认该 AK/SK 对目标桶有写权限（ASR 需要把音频上传到 TOS 并拿到预签名 URL）。

### OCR 结果为空或太慢

- 默认逐字转写黑板板书；若视频不是板书类（如纯录屏），加 `--caption` 改用画面描述；
- 每帧一次模型调用，慢属正常：用 `--max-frames` 控制费用上限，或调大 `--density-min-increment` / `--density-min-interval` 减少帧数。

## 输出产物

`<out-dir>/<视频名>/` 下的产物（以 path 3 为例）：

```
output/<视频名>/
├── audio_16k.wav           # ASR 中间产物：16kHz 单声道 WAV
├── subtitles.{srt,vtt,json}   # ASR 语音字幕（带时间戳，srt/vtt 可直接导入播放器）
├── captions.{srt,json}     # OCR 画面文字（板书逐字转写 / 画面描述）
├── merged.json             # ASR+OCR 融合结果（segments 含 text+visual，visual_blocks 含全部板书块）
├── full_transcript.{json,srt,md}  # 可选：豆包融合语音+板书的完整通顺转写
├── knowledge.md            # 知识文档：摘要 / 时间线 / 核心要点 / 画面内容 / 问答 / 术语表
├── cards.csv               # 问答闪卡（question,answer,tags,timestamp,source，可导入 Anki）
├── frames/                 # 采样帧 + frames.json（帧 manifest）
├── preprocessed/           # 可选：预处理后帧（黑板裁剪+增强）+ frames.json
└── convert.log             # --cid 模式下的缓存转换日志
```

各文件用途：

- `subtitles.*` — 语音转写结果，带时间戳，可作为视频字幕导入播放器；
- `captions.*` — 板书/画面文字识别结果，时间戳对齐到帧；
- `merged.json` — 融合后的结构化数据（每条语音段绑定对应板书），供知识生成与完整转写使用；
- `full_transcript.*` — 豆包大模型融合语音+板书的完整转写（与 `build_full_transcript.py` 的纯本地拼接版对应）；
- `knowledge.md` — 结构化知识文档，模板见 `assets/default-template.md`；
- `cards.csv` — 基于问答生成的闪卡，可导入 Anki 等复习软件。
