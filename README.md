# video2knowledge-doubao

基于豆包（火山引擎）生态的「视频 → 知识」工具：把一段教学/录屏视频自动转换为带时间戳的字幕、融合语音+板书的完整转写，以及可直接用于复习的结构化知识文档（摘要 / 时间线 / 核心要点 / 问答 / 术语表）和问答闪卡。

支持两种输入：

- **本地视频文件**（`--video <path>`）
- **B 站客户端缓存**（`--cid <cid>`，m4s 音视频流自动无损合并封装为 mp4）

## 流水线总览

```
── run_pipeline.py（单条视频）─────────────────────────────────────────
输入 ──┬─ 本地视频 (--video)
      └─ B 站客户端缓存 (--cid) → convert_bilibili_cache.py 无损合并为 mp4
           │
           ├─ (path 1/3) extract_frames.py density 取帧（文字密度增量采样，唯一模式）
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

── batch_process.py（批量，仅 B 站缓存）───────────────────────────────
缓存根目录 → 逐个转 mp4（converted/）→ density 取帧 → 黑板预处理
            → (可选 --with-asr) ASR → batch_summary.txt
            （默认只做本地免费步骤；已存在产物自动跳过，可中断续跑）
```

三条处理路径（`--path`）：

| path | 含义 | 调用的付费能力 | 主要产出 |
|---|---|---|---|
| 1 | 纯画面 OCR：只看板书/画面文字 | OCR（豆包视觉模型） | `captions.*` + `knowledge.*` |
| 2 | 语音转写：只听音频 | ASR（火山 SeedASR） | `subtitles.*` + `knowledge.*` |
| 3 | **融合模式（默认）**：语音 + 画面并行处理后再融合 | ASR + OCR + 知识生成 | `subtitles.*` + `captions.*` + `merged.json` + `knowledge.*` |

> **费用提示**：ASR、OCR、完整转写与知识生成均调用豆包/火山引擎付费接口（按音频时长、图片张数、Token 计费）。`--path 3` 一次会同时触发 ASR + OCR + 知识生成三类调用；帧提取（density 取帧）、预处理、缓存转 mp4 等本地操作免费。
>
> **付费 API 需先确认**：按项目约定（`.cursor/rules/pay-api-confirmation.mdc`），执行任何会触发付费 API 的命令前，必须先确认将调用的付费接口与预估数据量（帧数 / 音频时长）。批量入口 `batch_process.py` 默认只跑本地免费步骤，加 `--with-asr` 前应确认批量总时长与预估费用。

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

### 批量处理 B 站缓存（batch_process.py）

一次性处理全部（或部分）B 站缓存：逐个转 mp4 → density 取帧 → 黑板预处理。**默认只做本地免费步骤，不调用任何付费 API**；ASR 是付费接口（按音频时长计费），需显式 `--with-asr` 开启。

```bash
# 预览将处理的缓存（只列出，不执行）
python3 scripts/batch_process.py --list-only

# 批量处理（本地免费步骤：转换 mp4 + density 取帧 + 黑板预处理）
python3 scripts/batch_process.py

# 小规模调试：只处理前 2 个缓存
python3 scripts/batch_process.py --limit 2

# 只处理指定 cid（可多次 --cid）
python3 scripts/batch_process.py --cid 25685856471

# 开启 ASR 语音转写（付费；批量前请确认总时长与预估费用）
python3 scripts/batch_process.py --with-asr

# 中断后重跑同一命令即可续跑（已有产物自动跳过）；--force 强制全部重跑
python3 scripts/batch_process.py --force
```

### 标定 density 取帧阈值（analyze_frame_density.py）

对单个视频分析板区文字密度分布（分位数 / 直方图 / 空板区间估计），用于标定 `--density-floor` / `--density-min-increment`。纯本地操作，不调用付费 API：

```bash
python3 scripts/analyze_frame_density.py --video lecture.mp4 --sampling-fps 1.0
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

## batch_process.py 参数说明

批量处理入口只面向 B 站缓存。默认行为是「转换 mp4 + density 取帧 + 黑板预处理」全部本地免费步骤；ASR 付费，需显式 `--with-asr` 才开启。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--cache-dir <path>` | `/Users/duanp/Movies/bilibili` | B 站缓存根目录 |
| `--out-dir <path>` | 项目根目录 `output/` | 输出根目录；`<out>/converted/` 放转换后的 mp4，`<out>/<标题>/` 放每个视频的产物 |
| `--limit <n>` | 全部 | 只处理前 N 个完整缓存（小规模调试用） |
| `--cid <cid>` | 全部 | 只处理指定 cid（可多次指定） |
| `--list-only` | 关 | 只列出将处理的缓存并退出，不执行任何转换/分析 |
| `--with-asr` | 关 | 开启 ASR 语音转写（付费接口，按音频时长计费；批量前请确认总时长与费用） |
| `--language <code>` | `zh-CN` | ASR 语言代码（`zh-CN`/`en-US`/`ja-JP` 等） |
| `--skip-preprocess` | 关 | 跳过帧预处理，只转 mp4 + 取帧 |
| `--no-crop` | 关 | 预处理时不裁剪板面，只做增强 |
| `--force` | 关 | 忽略已有产物，全部重跑 |
| `--density-*` / `--max-frames` | 同 run_pipeline | 透传给 `extract_frames.py` 的 density 取帧参数（含义见下节） |

> 已存在的产物自动跳过（复用 `converted/` 下的 mp4，以及每个视频的 `frames.json` / `preprocessed/frames.json` / `subtitles.json`），中断后重跑同一命令即可续跑；`--force` 强制全部重跑。单个视频任一步骤失败不中断整体，汇总写入 `<out-dir>/batch_summary.txt`（成功/失败列表 + 完成步骤）。

## density 取帧模式（文字密度增量采样）

density 是当前**唯一**的取帧模式：用「板区文字密度」为主、「内容指纹」为辅来决定保留哪些帧，专为教学/板书视频设计。

### 核心信号：板区文字密度

- 只统计画面 **15%~60% 高度**的板区（`BOARD_CROP_GRAY` 滤镜），画面底部教师身体/讲台等高动态干扰不参与统计；
- 板区灰度高于 `--density-bright-threshold`（默认 200）的像素计为板书/粉笔像素，其占比即「板区密度」；
- 实测范围：空板 ≈ 0.06%~0.13%，有板书 0.3%~4%，密度随书写单调递增——是「板书是否在变化」的可靠信号。

### 为什么不用纯指纹去重

对真实教学视频实测，单纯 dHash 指纹去重会失真：

- 教师只讲不写 → 出现数百秒大空档（既没有帧可去重，也取不到帧）；
- 开头空板期被人物走动触发，误保留大量废帧；
- 连续书写时每一笔都触发，帧过密。

因此 density 改为「密度为主、指纹为辅」的增量采样，并配两个兜底机制。

### 选择规则

1. **书写增量**：密度相对上一保留帧净增 ≥ `--density-min-increment`（默认 0.35 个百分点）→ 保留（捕捉书写增量）；
2. **内容改写**：密度相近但 64-bit dHash 指纹汉明距离 ≥ `--density-fingerprint-hamming`（默认 10）→ 保留（捕捉改写/换公式等不显著改变密度的内容变化，修掉「跳跃漏帧」）；
3. **空板过滤**：密度 < `--density-floor`（默认 0.3%）视为空板，一律不保留（修掉「开头空板多帧」）；
4. **最小间隔**：保留帧间隔 < `--density-min-interval`（默认 12s）跳过（修掉「连续书写 1~3s 过密」）；
5. **擦板重写**：密度相对上一保留帧骤降 ≥ `--density-erase-drop`（默认 2.0 个百分点）判定为擦板并进入恢复期，之后密度回升至擦前密度的 `--density-erase-recover`（默认 0.7）比例即强制保留一帧——防止整板擦掉重写被跳过；
6. **低对比度兜底**：一次提取帧数 < `--density-min-frames`（默认 15）且视频时长 > 600s 时，自动把亮像素阈值降到 180→160→150 重跑，取首个帧数达标（或帧数最多）的结果（粉笔亮度偏低的视频在默认阈值下密度信号会被压平）；
7. **静止有板期兜底**：相邻保留帧间隔 > `--density-max-gap`（默认 300s）时，补入目标时刻之后首个有内容（密度 ≥ floor）的采样帧（`0`=关闭）——教师长时间只讲不写、板面静止时会产生数百秒大空档。

最后若保留帧数超过 `--max-frames`（默认 120），均匀降采样到该上限。指纹为 64-bit dHash，取帧通过单条 ffmpeg 管道流式输出板区原始分辨率灰度图，同时计算密度与指纹，避免全分辨率整批载入内存。产物为 `<out-dir>/frame_%06d.jpg` + `frames.json` manifest（记录每帧 `t` / `density` / `hamming` / `forced` 字段），供 `preprocess_frames.py` 与 `ocr_doubao.py` 使用。每保留帧对应一次 OCR 计费，`--max-frames` 即费用上限。

## 脚本清单

| 脚本 | 职责 | 是否调用付费 API |
|---|---|---|
| `run_pipeline.py` | 一键流水线入口：取帧 → 并行 ASR+OCR → 融合 → 知识文档 | 是（视 path） |
| `batch_process.py` | 批量处理 B 站缓存：全量转 mp4 → 逐视频 density 取帧 → 黑板预处理 →（可选）ASR，写 `batch_summary.txt` | 是（仅 `--with-asr` 时） |
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

### 批量模式（batch_process.py）

```
output/
├── batch_summary.txt        # 批量汇总：成功/失败列表（[ok]/[err] + cid + 标题 + 完成步骤）
├── converted/               # 转换后的 mp4（按标题命名 <标题>.mp4）
└── <标题>/                  # 每个视频一个目录
    ├── frames/              # density 取帧：frame_%06d.jpg + frames.json
    ├── preprocessed/        # 黑板裁剪+增强后的帧 + frames.json（--skip-preprocess 时无）
    └── subtitles.{srt,vtt,json}   # 仅 --with-asr 时生成
```
