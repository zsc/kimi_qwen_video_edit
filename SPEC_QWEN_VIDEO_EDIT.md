# SPEC: Qwen-Image-Edit → 视频编辑（仅推理改动）

面向：用 `gemini-cli` / `codex` 在本仓库内实现改动的工程师/代理。  
约束：**只改推理算法**（不训练、不改权重）；纯 Python CLI；默认运行在 GPU（CUDA）环境。

---

## 0. 现状说明（重要）
本 SPEC **只走“独立 CLI”路线**：
- 本仓库**不内置/不修改** `diffusers` 源码，也不在仓库里新增 `src/diffusers/...` 之类的补丁文件。
- 运行时依赖你本机 `pip` 环境中的新版 `diffusers`（需能 `from diffusers import QwenImageEditPlusPipeline`，并支持 `callback_on_step_end` 类回调来修改 `latents`）。
- “TTLG（loss guidance）+ 视频/帧读写 + 指标统计（edge/gram/LPIPS）”全部放在仓库 `scripts/` 下完成。

> 备注：如果你当前安装的 `diffusers/Qwen` pipeline 不支持回调修改 `latents`，MVP 的兜底做法是：在 `scripts/` 内复制一份该 pipeline 的 denoising loop 作为本地 wrapper（仍然不改 site-packages，不改仓库内 diffusers 源码）。


---

## 1. 目标与非目标

### 1.1 目标（MVP）

给定：
- **输入首帧**：`frame_rgb[0]`（RGB，作为身份/风格锚点）
- **输入 edge 序列**：`frame_edge[i]`，`1 <= i <= T-1`（每一帧的结构控制信号；与 Edge loss 里所用 **edge** 相同（Sobel/Laplacian））

要求输出：
- **输出 RGB 序列**：`o_frame_rgb[i]`，`0 <= i <= T-1`：
  - `o_frame_rgb[0] = frame_rgb[0]`（逐像素复制，不做生成）
  - 对 `1 <= i <= T-1`：`edge(o_frame_rgb[i]) ≈ frame_edge[i]`
  - 风格尽量与 `frame_rgb[0]` 一致（Gram guidance），并尽量保持时间一致性（可选 temporal loss）

实现方式（只推理）：
- 在扩散去噪迭代中加入 **test-time loss guidance (TTLG)**：
  - 对第 i 帧的生成，使用 Qwen-Image-Edit-2509 的**原生多图输入**：`image=[图1, 图2]`（见 3.3），其中：
    - 图1：本帧 edge（Laplacian/Sobel 图，0~255）
    - 图2：上一帧 RGB（`o_frame_rgb[i-1]`；第 1 帧时为 `frame_rgb[0]`）
    - 默认提示词：`将图 2 按图 1 所勾勒出的精致形状进行变形，生成一张图像输出`
  - **Edge loss guidance**：对 denoising loop 内的某个“可解码图像表征”（优先 `pred_x0`，否则退化为当前 `latents`；见 4.5）做可微 decode 得到 RGB，提取可微 edge（Sobel/Laplacian），与目标 edge 计算 loss，对 `latents` 做梯度步进形成 guidance。
  - **Gram style guidance**：用 VGG 特征的 Gram 矩阵约束生成图与 `frame_rgb[0]` 在风格上的一致性，同样以梯度方式指导 latents。
- 支持**流式推理**：逐帧读取 edge、逐帧生成并落盘；除 `init_frame` 与上一帧外不常驻保存全视频到内存。
- 指标（用于验收/回归）：输出每帧的 `edge_loss / gram_loss / lpips_to_init / lpips_to_prev`（见第 6 章的阈值与统计方式）。

### 1.2 非目标（明确不做）

- 不引入训练/微调（LoRA、DreamBooth、ControlNet 训练等都不做）。
- 不做复杂的时序一致性训练方法（如训练一个时序模块）。
- 不要求达到 SOTA 视频一致性；MVP 先保证“能跑 + 可控 + 有 edge/style TTLG”。

---

## 2. CLI 形态（纯 Python）

新增一个可直接运行的脚本（建议路径）：
- `scripts/qwen_video_edit.py`

### 2.1 输入输出

必须支持两种数据源（至少一种即可先落地，另一种作为可选）：

1) **帧目录模式（最稳，不依赖 ffmpeg）**
   - `--init_frame /path/to/init.png`（对应 `frame_rgb[0]`）
   - `--edge_frames_dir /path/to/edge_frames/`
   - 输出：`--out_frames_dir /path/to/out_frames/`

2) **视频文件模式（可选，依赖 imageio/ffmpeg 或 PyAV）**
   - `--rgb_video input.mp4`（只读取首帧作为 `frame_rgb[0]`）
   - `--edge_video edge.mp4`（帧数应为 `T-1`；其第 0 帧对应 `frame_edge[1]`）。如缺失，则由 input.mp4 的除首帧跑 Sobel/Laplacian 生成备用。
   - 输出：`--out_video out.mp4`

两种模式都要满足：
- edge 序列长度为 `T-1`；输出序列长度为 `T`（`o_frame_rgb[0]` 直接复制 `init_frame`）。
- edge 帧尺寸若与 `init_frame` 不一致：默认把 edge resize 到 `init_frame` 尺寸（插值：nearest/area，可配置）。

### 2.2 核心参数（建议）

通用：
- `--model_id`（默认 `Qwen/Qwen-Image-Edit-2509`）
- `--prompt`
- `--negative_prompt`（默认 `" "`）
- `--num_inference_steps`（默认 50）
- `--true_cfg_scale`（默认 4.0，与 Qwen pipeline 参数保持一致）
- `--seed`（默认 0）
- `--device`（默认 `cuda`）
- `--dtype`（默认 `bf16`）
- `--max_frames`（可选；输出总帧数上限，含首帧；例如 `max_frames=25` 则最多消费 24 张 edge 帧）

TTLG（关键）：
- `--ttlg_edge_scale`（float，默认 0.0；>0 才启用）
- `--ttlg_gram_scale`（float，默认 0.0；>0 才启用）
- `--ttlg_lr`（float，默认 0.05；latents 梯度步长）
- `--ttlg_every`（int，默认 1；每隔多少 denoise step 做一次 TTLG）
- `--ttlg_start_step` / `--ttlg_end_step`（可选；只在某个 step 区间启用 TTLG）
- `--ttlg_last_steps`（int，可选；只在最后 N 个 denoise step 启用 TTLG，用于避免早期噪声过大导致 guidance 不稳定；推荐从 10~20 起试）
- `--ttlg_edge_size`（int，可选；edge loss 计算的边长，比如 256；用于降采样提速）
- `--ttlg_gram_size`（int，可选；VGG 输入边长，比如 224/256）

可选（第二阶段）：
- `--ttlg_temporal_scale`（float，默认 0.0；>0 启用一个轻量时序 loss）
- （不提供 `--init_mode`）MVP **只支持 prev 滚动**：始终用 `o_frame_rgb[i-1]` 作为图2输入（提升时序一致性；代价是无法 batch，但本 SPEC 不做 batch）。

指标输出（用于验收/调参）：
- `--metrics_out /path/to/metrics.csv`（可选；写出 per-frame 指标）
- `--save_debug_dir /path/to/debug/`（可选；保存中间可视化，如 soft edge、下采样图等）
- 指标定义建议（避免实现/阈值不一致）：
  - `edge_loss`：使用 4.2 的 `sobel_edges` + `L1(mean)`
  - `gram_loss`：使用 4.3 的 Gram + `L1(mean)`（与 `init_frame` 对比）
  - `lpips_*`：使用 `lpips.LPIPS(net="vgg")`；输入用 `[0,1] → [-1,1]` 并 resize 到同一边长（如 256）

### 2.3 CLI 示例（帧目录模式）

```bash
python scripts/qwen_video_edit.py \
  --init_frame ./data/init.png \
  --edge_frames_dir ./data/edge_frames \
  --out_frames_dir ./outputs/out_frames \
  --model_id Qwen/Qwen-Image-Edit-2509 \
  --prompt "将图 2 按图 1 所勾勒出的精致形状进行变形，生成一张图像输出" \
  --negative_prompt " " \
  --num_inference_steps 50 \
  --true_cfg_scale 4.0 \
  --seed 0 \
  --device cuda \
  --dtype bf16 \
  --ttlg_edge_scale 1.0 \
  --ttlg_gram_scale 0.2 \
  --ttlg_lr 0.05
```

---

## 3. 推理算法：把“单图编辑”扩展为“首帧锚点 + edge 序列”

### 3.1 时间索引与长度约定

令 `edge_frames` 为按顺序读取的 edge 列表，长度为 `T-1`。则：
- 输出帧数 `T = len(edge_frames) + 1`
- `o_frame_rgb[0]` 直接等于 `init_frame`
- 对 `k in [0, T-2]`：`edge_frames[k]` 对应 `frame_edge[k+1]`，生成 `o_frame_rgb[k+1]`

### 3.2 生成策略（MVP）

MVP **不做 batch**，逐帧滚动生成（prev mode only）：

1) 读入 `init_frame`（RGB），并把它作为 `o_frame_rgb[0]` 原样复制到输出。
2) 初始化 `prev_rgb = init_frame`。
3) 依次遍历 `edge_frames[k]`（对应目标输出帧 `o_frame_rgb[k+1]`）：
   - 读入当前 `edge_img = edge_frames[k]`（必要时 resize 到与 `init_frame` 同尺寸）。
   - 构造 Qwen pipeline 的多图输入：`image=[edge_img, prev_rgb]`（见 3.3）。
   - 调用 pipeline 生成 `out_rgb`：
     - 若启用 TTLG：通过 `callback_on_step_end` 在 denoising loop 内对 `latents` 做梯度更新（见第 4 章）。
   - 保存 `out_rgb` 到磁盘，并令 `prev_rgb = out_rgb`（作为下一帧的图2输入）。
4) 可选：在生成过程中同步写出 `metrics_out`（每帧 edge/gram/LPIPS），用于验收与调参（见第 6 章）。

> 说明：由于“图2=上一帧输出”这一依赖，天然是串行时序；因此不支持 `--batch_frames`，也不提供 `first` 模式开关。

### 3.3 “多图输入 + edge 控制”的参数约定

Qwen-Image-Edit-2509（`QwenImageEditPlusPipeline`）原生支持**多图输入**：
- 参数名：`image`
- 类型：`PIL.Image.Image` 或 `list[PIL.Image.Image]`
- 约定：当 `image` 是 list 时，提示词中的“图 1/图 2”分别对应 `image[0] / image[1]`

本任务的约定（视频编辑）：
- `image[0]`（图1）：edge/Laplacian 图（建议已 normalize 到 0~255 的 8bit 图；`mode="L"` 或 `mode="RGB"` 均可）
- `image[1]`（图2）：初始/上一帧 RGB
  - 对帧 `i=1`：`image[1] = frame_rgb[0]`
  - 对帧 `i>=2`：`image[1] = o_frame_rgb[i-1]`（上一帧滚动）

示例（Qwen-Image-Edit-2509 原生多图输入；未加 guidance loss）：
```python
import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)
image1 = Image.open("input1.png")  # laplacian 图, normalize 成 0~255
image2 = Image.open("input2.png")  # 初始值，为 rgb[0] 或滚动生成的上一帧 rgb
prompt = "将图 2 按图 1 所勾勒出的精致形状进行变形，生成一张图像输出"
inputs = {
    "image": [image1, image2],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit_plus.png")
    print("image saved at", os.path.abspath("output_image_edit_plus.png"))
```

注意：上面示例使用 `torch.inference_mode()` 仅适用于**不启用 TTLG** 的情况；当启用 TTLG 时必须移除 `inference_mode`（否则无法在回调里开启梯度）。


## 4. Test-Time Loss Guidance (TTLG) 设计

### 4.1 为什么需要可微 edge

TTLG 要从 loss 反传到 latents，所以 “edge 提取” 必须可微。
- **不要**用 OpenCV Canny 直接做 loss（不可微，会导致梯度为 0）。
- **推荐**：Sobel（卷积）或 Laplacian（卷积）得到 soft edge map。

### 4.2 Edge loss（Sobel 示例）

输入：解码后的预测图 `img`，形状 `[B,3,H,W]`，值域 `[0,1]`。  
步骤：
1) 灰度：`gray = 0.2989*R + 0.5870*G + 0.1140*B` → `[B,1,H,W]`
2) Sobel 卷积得到 `dx, dy`
3) `edge = sqrt(dx^2 + dy^2 + eps)`，再做归一化/截断（可选）
4) loss：`L_edge = mean(|edge_pred - edge_target|)` 或 MSE

`edge_target` 的来源：
- 若输入的 edge 帧本身就是 edge 图：直接转成 `[0,1]` 灰度并 resize 到一致即可；
- 或者同样跑一遍 Sobel，保证“度量空间一致”。

### 4.3 Gram style loss（VGG16 特征）

依赖：`torchvision`（建议作为可选依赖，缺失则禁用 Gram guidance 并给出清晰报错）。

步骤：
1) 用 VGG16 的若干卷积层特征（如 `relu1_2, relu2_2, relu3_3, relu4_3`）。
2) 对每层特征 `F`（`[B,C,H,W]`）展平为 `[B,C,HW]`，Gram：`G = (F @ F^T) / (C*H*W)`。
3) 对同层 Gram 做 `L1/MSE`：
   - `L_gram = Σ_l w_l * mean(|G_pred_l - G_ref_l|)`
4) 输入归一化：
   - VGG 输入使用 `float32`，并做 ImageNet mean/std normalize（避免不同 dtype 下数值不稳定）。

优化：
- `G_ref_l`（来自 `frame_rgb[0]`）在推理前预计算并缓存（只算一次，供所有帧复用）。
- VGG 权重冻结：`requires_grad_(False)`；只对 latents 求梯度。

### 4.4 TTLG 的插入点（独立 CLI）

独立 CLI 路线下 **不修改 diffusers 源码**。TTLG 的推荐插入方式是使用 diffusers 的回调：
- `callback_on_step_end`：在每个 denoise step 结束时拿到 `latents`，并可返回修改后的 `latents` 继续后续 step（不同 pipeline 的具体参数名以你安装的 diffusers 为准）。

推荐时机（对应“step end”回调）：
- 以回调拿到的 `latents` 为当前状态做 1 次（或 K 次）梯度更新，然后把更新后的 `latents` 返回给 pipeline。

关键点：
- 很多 pipeline 的 `__call__` 会在 `@torch.no_grad()` 下执行，但 TTLG 需要在回调内部局部开启梯度：`with torch.enable_grad():`。
- **外部调用者（CLI）不要用 `torch.inference_mode()`** 包裹整次推理（`inference_mode` 不能被 `enable_grad` 反转，会导致梯度永远为 0）。
- 多数 pipeline 的 `decode_latents()` 会把结果 `.cpu().numpy()`（不可微）。TTLG 必须用“可微 decode”：
  - 直接调用 `pipe.vae.decode(...)`，并保持输出为 **torch tensor**（不转 numpy、不搬 CPU）。
- 性能/显存：在 pipeline 初始化后一次性冻结权重：
  - `unet/vae/text_encoder/(可选)vgg` 全部 `requires_grad_(False)`；TTLG 里只让 `latents` `requires_grad_(True)`。

### 4.5 TTLG 的更新规则（推荐实现）

独立 CLI + 回调场景下，为了避免对 scheduler/预测类型做过强假设，MVP 推荐采用**scheduler-agnostic**的更新规则：直接对回调拿到的 `latents` 做可微 decode，并在 **最后若干步**（`--ttlg_last_steps`）启用 guidance。

设回调拿到的当前状态为 `latents`（不强行标注为 `x_t` 或 `x_{t-1}`；以 pipeline 实际回调语义为准）。

1) 在 guidance block 内构建需要梯度的 latents：
   - `latents_g = latents.detach().float().requires_grad_(True)`
2) 选取用于解码并计算 loss 的 latent 表征（优先级从高到低）：
   - 若 pipeline 在回调 kwargs 中提供 `pred_original_sample`/`pred_x0`（名字以实际为准）：优先用它；
   - 否则退化为 `latents_g`（仅在 `--ttlg_last_steps` 内启用，避免早期噪声过大）。
3) 解码（可微）并归一化到 `[0,1]`：
   - `latents_vae = latent_to_decode / pipe.vae.config.scaling_factor`（若该 scaling_factor 存在；否则按 pipeline 习惯处理）
   - `img_pred = pipe.vae.decode(latents_vae).sample` → `[B,3,H,W]`（通常在 `[-1,1]`）
   - `img_pred = (img_pred / 2 + 0.5).clamp(0, 1)`（用于 edge/VGG/LPIPS）
4) 计算 loss：
   - `L = ttlg_edge_scale * L_edge(img_pred, edge_target) + ttlg_gram_scale * L_gram(img_pred, gram_ref)`
   - （可选）+ `ttlg_temporal_scale * L_temporal(img_pred, img_prev)`
5) 求梯度并更新（一次更新即可，K 次迭代先不做）：
   - `grad = autograd.grad(L, latents_g)[0]`
   - `grad = grad / (||grad|| + eps)`（按 sample 归一化，增强稳定性）
   - `latents = (latents_g - ttlg_lr * grad).to(original_dtype).detach()`

TTLG 触发条件：
- `i % ttlg_every == 0`
- `ttlg_start_step <= i < ttlg_end_step`（若提供）
- 若提供 `ttlg_last_steps`：只在最后 `ttlg_last_steps` 个 step 内启用
- `ttlg_edge_scale > 0 or ttlg_gram_scale > 0 or ttlg_temporal_scale > 0`

### 4.6 可选：轻量时序一致性（Phase 2）

不引入额外模型训练的情况下，提供一个“弱约束”开关即可：

- `L_temporal = mean(|img_pred[t] - img_pred[t-1]|)`（在低分辨率上做；或在 VGG feature 上做）

注意：
- 由于本任务是逐帧串行生成，可直接缓存上一帧的 `img_pred`/VGG feature 作为 `img_prev`（无需 batch）。
- 没有光流对齐会产生“抹平运动”的副作用；因此默认 `ttlg_temporal_scale=0`，并在 CLI help 里明确风险。

---

## 5. 代码改动清单（建议）

### 5.1 新增：视频 CLI（主入口）

新增文件：
- `scripts/qwen_video_edit.py`

职责：
- 解析参数、加载 `init_frame` 与 edge 帧、逐帧滚动调用 pipeline、保存输出帧/视频（并确保首帧原样复制）。
- （可选）输出 `metrics_out`（edge/gram/LPIPS）用于验收与调参。

### 5.2 新增：TTLG/指标工具模块（放在 scripts/ 下）

新增文件（建议）：
- `scripts/qwen_video_edit_ttlg_guidance.py`

包含（按需实现，优先满足 MVP）：
- `LossGuidanceConfig`（dataclass）
- `sobel_edges(x: torch.Tensor) -> torch.Tensor`
- `gram_matrix(feat: torch.Tensor) -> torch.Tensor`
- `VGGStyleExtractor`（封装 torchvision vgg + 选层输出；冻结权重）
- `make_ttlg_callback(...)`（构造 `callback_on_step_end`，在回调里更新 `latents` 并回传）
- `compute_metrics(...)`（edge/gram/LPIPS；用于 `--metrics_out`）

依赖处理：
- `torchvision` 作为可选依赖；缺失且 `--ttlg_gram_scale>0` 时给出可读报错。
- `lpips` 作为可选依赖；缺失则跳过 LPIPS 指标但不影响主流程。

（可选）新增文档：
- `docs/source/en/using-diffusers/qwen_video_edit.md`（或放到 `docs/` 对应结构）

---

## 6. 测试与验收

### 6.1 自检（必须，独立 CLI 形态）

由于本路线不改 diffusers 仓库代码，建议在 `scripts/qwen_video_edit.py` 增加 `--self_test`：
- 在不加载大模型的情况下，用随机 tensor 跑一遍 `sobel_edges/gram_matrix` 并 `backward()`，确认可反传。
- 若启用 `--ttlg_gram_scale`：检查 `torchvision` 可用；否则给出清晰报错与安装提示。

### 6.2 Pipeline 快测（建议）

用真实 Qwen pipeline 进行最小冒烟测试（不追求画质）：
- 输入 1 张 `init_frame` + 2 张 `edge_frame`
- `num_inference_steps=2`，并打开 `--ttlg_edge_scale>0`（且 `--ttlg_last_steps=1`）
- 只验证：
  - 能跑通、无 NaN
  - `metrics_out` 能写出每帧指标
  - 首帧输出与 `init_frame` 完全一致

### 6.3 验收标准（MVP）

1) `scripts/qwen_video_edit.py` 能处理 `init_frame + edge_frames_dir` 输入，输出帧数为 `len(edge_frames)+1`，且 `o_frame_rgb[0]` 与 `init_frame` 完全一致。
2) TTLG 的效果用**可测指标**验收（建议固定同一输入、同一 seed，分别跑 TTLG on/off 两次，并写出 `metrics_out`）：
   - Edge（开 `--ttlg_edge_scale>0`）：
     - 统计前 `N=min(8, T-1)` 帧的 `edge_loss` 均值，要求
       - `mean(edge_loss_on) <= mean(edge_loss_off) * 0.95`（至少 5% 相对下降）
     - 同时要求 `edge_loss_on` 在至少 60% 的帧上不大于对应的 `edge_loss_off`（避免单帧偶然波动）。
   - Gram（开 `--ttlg_gram_scale>0`）：
     - 同样统计 `gram_loss`，要求
       - `mean(gram_loss_on) <= mean(gram_loss_off) * 0.98`（至少 2% 相对下降；Gram 通常更“软”，阈值放宽）
   - LPIPS（若安装了 `lpips`，作为“合理性 + 时序”约束）：
     - 计算每帧 `lpips_to_prev` 与 `lpips_to_init`（对 `i>=1`）
     - 要求 `mean(lpips_to_prev) < mean(lpips_to_init)`（时间上更接近上一帧而非始终贴着首帧）
     - 要求 `mean(lpips_to_prev) > 0.005` 且 `mean(lpips_to_init) > 0.01`（避免退化成逐帧完全复制）
   - 说明：GPU 上可能存在轻微非确定性；如偶发不满足，可改为跑 `seeds=[0,1,2]`，用 3 次结果的 **median** 再判定上述阈值。
3) CLI 给出清晰报错：
   - edge 帧序列为空/帧数不足（例如与 `--max_frames` 不匹配）
   - `torchvision` 缺失但 `--ttlg_gram_scale>0`
   - 用户传了 `torch.inference_mode()` 导致梯度不可用（可通过检测 `torch.is_inference_mode_enabled()` 给提示）
