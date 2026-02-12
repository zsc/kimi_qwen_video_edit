qwen-image-edit 是图像编辑，我想改成视频编辑，方法如下（只涉及到推理算法的变化）。

Qwen-Image-Edit-2509（以下简称Q09） 支持多图输入，原生支持ControlNet: 包括深度图、边缘图、关键点图等。
我们的主要使用场景，是给定图1为普通 RGB，图2 为 edge 图，要求 Q09 输出图1 按图2 变化的结果。为了提升一致性，我们采用 test time loss guidance 的方法，即对扩散迭代的生成图的中间结果，提取 edge 和图2的 edge 算 loss 产生 guidance。进一步的，我们会用风格迁移里的 Gram 矩阵约束扩散迭代的生成图和图1 来提升风格的一致性。

---
import os
from PIL import Image
import torch

from diffusers import QwenImageEditPipeline

pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
print("pipeline loaded")
pipeline.to(torch.bfloat16)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)
image = Image.open("./input.png").convert("RGB")
prompt = "Change the rabbit's color to purple, with a flash light background."
inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit.png")
    print("image saved at", os.path.abspath("output_image_edit.png"))
----

整个程序为纯 python CLI。
结合本目录的 diffusers 相关代码，制订 gemini-cli/codex 能用的 SPEC markdown 来实现上述修改。（本机暂无 gpu，所以先只制订 spec）
