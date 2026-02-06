# Qwen Video Edit with TTLG

基于 Qwen-Image-Edit-2509 的视频编辑工具，使用 Test-Time Loss Guidance (TTLG) 技术实现无需训练的视频风格化和结构控制。

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)

## 功能特性

- 🎬 **视频帧生成**：从首帧 + 边缘序列生成完整视频
- 🎨 **TTLG 引导**：测试时损失引导，无需训练
  - Edge Loss：边缘结构控制
  - Gram Loss：风格一致性
  - Temporal Loss：时序平滑（可选）
- 🔄 **流式处理**：逐帧生成，内存友好
- 📊 **质量评估**：自动生成 edge/gram/LPIPS 指标
- 🚀 **断点续传**：支持中断后恢复生成

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 准备模型

确保 Qwen-Image-Edit-2509 模型已下载：

```bash
# 默认路径
/.autodl-model/data/Qwen/Qwen-Image-Edit-2509

# 或使用 Hugging Face 自动下载
export MODEL_ID=Qwen/Qwen-Image-Edit-2509
```

### 基本使用

```bash
# 单帧测试
python scripts/qwen_video_edit.py \
  --init_frame data/init.png \
  --edge_frames_dir data/edge_frames \
  --out_frames_dir outputs \
  --num_inference_steps 10 \
  --ttlg_edge_scale 1.0 \
  --ttlg_gram_scale 0.3 \
  --seed 42
```

### 视频处理流程

#### 1. 提取视频帧和边缘

```bash
python3 << 'EOF'
import cv2
import numpy as np
from PIL import Image
import os

os.makedirs('data_video/init', exist_ok=True)
os.makedirs('data_video/edges', exist_ok=True)

video = cv2.VideoCapture('input.mp4')

# 提取首帧
ret, frame = video.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_square = cv2.resize(frame_rgb, (512, 512))
Image.fromarray(frame_square).save('data_video/init/init.png')

# 提取边缘帧（1/4 帧率）
frame_idx = 0
edge_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    frame_idx += 1
    if frame_idx % 4 == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_square = cv2.resize(frame_rgb, (512, 512))
        frame_gray = cv2.cvtColor(frame_square, cv2.COLOR_RGB2GRAY)
        
        # Sobel 边缘检测
        sobelx = cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.sqrt(sobelx**2 + sobely**2)
        edge = np.uint8(255 * edge / edge.max())
        edge_rgb = np.stack([edge, edge, edge], axis=-1)
        
        Image.fromarray(edge_rgb).save(f'data_video/edges/edge_{edge_count+1:04d}.png')
        edge_count += 1

video.release()
print(f"Extracted: 1 init + {edge_count} edges")
EOF
```

#### 2. 生成视频帧

```bash
python scripts/qwen_video_edit.py \
  --init_frame data_video/init/init.png \
  --edge_frames_dir data_video/edges \
  --out_frames_dir outputs_video \
  --num_inference_steps 10 \
  --ttlg_edge_scale 1.0 \
  --ttlg_gram_scale 0.3 \
  --ttlg_lr 0.05 \
  --ttlg_last_steps 5 \
  --seed 42 \
  --dtype bf16 \
  --metrics_out outputs_video/metrics.csv
```

#### 3. 合成视频

```bash
ffmpeg -framerate 4.48 -i outputs_video/frame_%04d.png \
  -c:v libx264 -pix_fmt yuv420p output_video.mp4
```

## 参数说明

### 输入输出
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--init_frame` | 初始化帧（风格参考） | 必填 |
| `--edge_frames_dir` | 边缘帧目录 | 必填 |
| `--out_frames_dir` | 输出帧目录 | 必填 |
| `--metrics_out` | 指标 CSV 输出路径 | 可选 |

### 模型参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_id` | 模型路径或 HuggingFace ID | `/.autodl-model/data/Qwen/Qwen-Image-Edit-2509` |
| `--num_inference_steps` | 去噪步数 | 50 |
| `--true_cfg_scale` | CFG 缩放 | 4.0 |
| `--seed` | 随机种子 | 0 |
| `--dtype` | 数据类型 (fp16/bf16/fp32) | bf16 |

### TTLG 参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ttlg_edge_scale` | 边缘损失权重 (0=禁用) | 0.0 |
| `--ttlg_gram_scale` | Gram 风格损失权重 | 0.0 |
| `--ttlg_temporal_scale` | 时序损失权重 | 0.0 |
| `--ttlg_lr` | 潜变量学习率 | 0.05 |
| `--ttlg_last_steps` | 仅最后 N 步启用 TTLG | 无 |
| `--ttlg_every` | 每隔多少步应用 TTLG | 1 |

## 项目结构

```
.
├── scripts/
│   ├── qwen_video_edit.py              # 主 CLI 脚本
│   ├── qwen_video_edit_ttlg_guidance.py # TTLG 工具模块
│   ├── create_test_data.py             # 测试数据生成
│   └── process_full_video.py           # 批量视频处理
├── data/                               # 输入数据（gitignore）
├── outputs/                            # 输出结果（gitignore）
├── experiment_report.html              # 实验报告
├── requirements.txt                    # Python 依赖
├── README.md                           # 本文件
└── .gitignore                          # Git 忽略规则
```

## 技术原理

### TTLG (Test-Time Loss Guidance)

在扩散模型推理阶段，通过损失函数梯度引导潜变量更新：

```
latents = latents - lr * ∇(L_edge + L_gram + L_temporal)
```

### Qwen-Image-Edit-2509 适配

- 潜变量格式：打包格式 (B, 4096, 64) → VAE 格式 (B, 16, 1, 128, 128)
- VAE 归一化：应用 latents_mean 和 latents_std
- 多图输入：`image=[edge_img, prev_frame]`

## 实验结果

详见 [experiment_report.html](experiment_report.html)（独立 HTML 文件，含完整结果）

### 测试视频
- **来源**: `test.mp4` (1280×704, 361帧, 20秒)
- **内容**: 扫地机器人上的猫
- **处理**: 1/4 帧率采样 → 91 帧
- **指标**:
  - Edge Loss: ~0.09-0.11
  - Gram Loss: ~0.007-0.008

## 常见问题

### Q: 显存不足？
A: 尝试减少 `--num_inference_steps` 或使用 `--dtype fp16`

### Q: 生成速度慢？
A: 每帧约 20-30 秒是正常的。可使用 `process_full_video.py` 断点续传

### Q: 边缘引导不明显？
A: 增加 `--ttlg_edge_scale` 到 2.0 或更多，或减少 `--ttlg_last_steps`

### Q: 风格漂移？
A: 增加 `--ttlg_gram_scale` 到 0.5 或更高

## 许可证

MIT License

## 引用

基于 [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) 开发
