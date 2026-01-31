# TimeLens 时间戳编码方式详解

本文档详细说明 TimeLens 模型如何在输入中添加时间戳信息。

## 1. 基础 Messages 格式（JSON 风格）

所有模型共用的基础输入格式：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "video",
          "video": "/path/to/video.mp4",
          "min_pixels": 50176,
          "total_pixels": 11239424,
          "fps": 2
        },
        {
          "type": "text",
          "text": "Please find the visual event described by the sentence 'a person opens a door', determining its starting and ending times. The format should be: 'The event happens in <start time> - <end time> seconds'."
        }
      ]
    }
  ]
}
```

## 2. TimeLens-7B（Qwen2.5-VL）- Interleaved Textual Timestamps

### 特点
- 使用 **交织式文本时间戳**（Interleaved Textual Timestamps）
- 时间戳作为文本插入到每一帧图像之前
- 通过 `return_video_metadata=True` 获取视频元数据

### Prompt 格式
```
"You are given a video with multiple frames. The numbers before each video frame indicate its sampling timestamp (in seconds). Please find the visual event described by the sentence '{query}', determining its starting and ending times. The format should be: 'The event happens in <start time> - <end time> seconds'."
```

### 处理后的输入序列（概念示意）
```
[System Prompt]
<|im_start|>user
You are given a video with multiple frames. The numbers before each video frame indicate its sampling timestamp (in seconds). ...

0.0 <video_frame_1> 0.5 <video_frame_2> 1.0 <video_frame_3> 1.5 <video_frame_4> ...

Please find the visual event described by the sentence 'a person opens a door'...
<|im_end|>
<|im_start|>assistant
```

### 代码调用方式
```python
from qwen_vl_utils import process_vision_info

# TimeLens-7B 使用 return_video_metadata=True
images, videos = process_vision_info(messages, return_video_metadata=True)

inputs = processor(
    text=[text],
    images=images,
    videos=videos,  # videos 包含了时间戳元数据，会自动交织到文本中
    padding=True,
    return_tensors="pt",
)
```

## 3. TimeLens-8B（Qwen3-VL）- Video Metadata 方式

### 特点
- 使用 **video_metadata** 参数传递时间戳信息
- 时间戳信息通过独立的元数据字段传入
- 模型内部处理时间戳与帧的对应关系

### Prompt 格式（不含时间戳前缀说明）
```
"Please find the visual event described by the sentence '{query}', determining its starting and ending times. The format should be: 'The event happens in <start time> - <end time> seconds'."
```

### 代码调用方式
```python
from qwen_vl_utils import process_vision_info

# TimeLens-8B / Qwen3-VL 使用 video_metadata 参数
images, videos, video_kwargs = process_vision_info(
    messages,
    image_patch_size=16,
    return_video_kwargs=True,
    return_video_metadata=True,
)

# 解包视频和元数据
videos, video_metadatas = zip(*videos)
videos, video_metadatas = list(videos), list(video_metadatas)

inputs = processor(
    text=[text],
    images=images,
    videos=videos,
    video_metadata=video_metadatas,  # 时间戳信息通过这个参数传入
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)
```

### video_metadata 结构示例
```json
{
  "video_metadata": [
    {
      "total_frames": 120,
      "fps": 2,
      "duration": 60.0,
      "frame_timestamps": [0.0, 0.5, 1.0, 1.5, 2.0, ...]
    }
  ]
}
```

### inputs 对象结构示例（processor 输出）

调用 `processor()` 后返回的 `inputs` 对象是一个 `BatchEncoding` 类型，包含以下键值：

```python
# inputs 对象的结构（假设视频有 120 帧，序列长度为 8000）
inputs = {
    "input_ids": torch.Tensor,           # 形状: [batch_size, seq_len], 例如 [1, 8000]
    "attention_mask": torch.Tensor,      # 形状: [batch_size, seq_len], 例如 [1, 8000]
    "pixel_values_videos": torch.Tensor, # 形状: [batch_size, num_frames, channels, height, width]
                                         # 例如 [1, 120, 3, 224, 224]
    "video_grid_thw": torch.Tensor,      # 视频网格信息 [temporal, height, width]
}
```

**详细示例**（假设一个 60 秒视频，采样率 2 fps）：

```python
print(f"inputs 类型: {type(inputs)}")
# 输出: inputs 类型: <class 'transformers.tokenization_utils_base.BatchEncoding'>

print(f"inputs 包含的键: {inputs.keys()}")
# 输出: inputs 包含的键: dict_keys(['input_ids', 'attention_mask', 'pixel_values_videos', 'video_grid_thw'])

print(f"input_ids 形状: {inputs['input_ids'].shape}")
# 输出: input_ids 形状: torch.Size([1, 8256])

print(f"attention_mask 形状: {inputs['attention_mask'].shape}")
# 输出: attention_mask 形状: torch.Size([1, 8256])

print(f"pixel_values_videos 形状: {inputs['pixel_values_videos'].shape}")
# 输出: pixel_values_videos 形状: torch.Size([1, 120, 3, 224, 224])
# 解释: [batch_size=1, num_frames=120, channels=3, height=224, width=224]

print(f"video_grid_thw 形状: {inputs['video_grid_thw'].shape}")
# 输出: video_grid_thw 形状: torch.Size([1, 3])
# 解释: 每个视频的 [temporal_patches, height_patches, width_patches]

# 查看 input_ids 的部分内容（包含特殊标记）
print(f"input_ids 前 50 个 token: {inputs['input_ids'][0, :50].tolist()}")
# 输出示例: [151643, 151644, 8948, 198, 2610, 525, 264, ...]
# 这些是 tokenized 的文本，包含系统提示和用户输入
```

**完整的 inputs 对象使用示例**：

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

# 加载模型和处理器
model = AutoModelForImageTextToText.from_pretrained("TencentARC/TimeLens-8B")
processor = AutoProcessor.from_pretrained("TencentARC/TimeLens-8B")

# 构建 messages
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/path/to/video.mp4",
                "min_pixels": 64 * 28 * 28,
                "total_pixels": 14336 * 28 * 28,
                "fps": 2
            },
            {
                "type": "text",
                "text": "Please find the visual event described by the sentence 'a person opens a door'..."
            }
        ]
    }
]

# 应用聊天模板
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 处理视觉信息
images, videos, video_kwargs = process_vision_info(
    messages,
    image_patch_size=16,
    return_video_kwargs=True,
    return_video_metadata=True,
)

# 解包视频和元数据
videos, video_metadatas = zip(*videos)
videos, video_metadatas = list(videos), list(video_metadatas)

# 调用 processor 获取 inputs
inputs = processor(
    text=[text],
    images=images,
    videos=videos,
    video_metadata=video_metadatas,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)

# 打印 inputs 结构
print("=" * 50)
print("inputs 对象详细信息:")
print("=" * 50)
for key, value in inputs.items():
    if hasattr(value, 'shape'):
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"{key}: {type(value)}")

# 输出示例:
# ==================================================
# inputs 对象详细信息:
# ==================================================
# input_ids: shape=torch.Size([1, 8256]), dtype=torch.int64
# attention_mask: shape=torch.Size([1, 8256]), dtype=torch.int64
# pixel_values_videos: shape=torch.Size([1, 120, 3, 224, 224]), dtype=torch.float32
# video_grid_thw: shape=torch.Size([1, 3]), dtype=torch.int64

# 将 inputs 移动到 GPU 并进行推理
inputs = inputs.to("cuda")
output_ids = model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=512,
)

# 解码输出
generated_ids_trimmed = output_ids[0, inputs.input_ids.shape[1]:]
answer = processor.decode(generated_ids_trimmed, skip_special_tokens=True)
print(f"模型输出: {answer}")
# 输出示例: 模型输出: The event happens in 0.0 - 5.0 seconds.
```

## 4. Qwen2.5-VL 原始模型（无时间戳增强）

### 特点
- 不使用时间戳增强
- 仅依赖视频帧的顺序信息

### 代码调用方式
```python
from qwen_vl_utils import process_vision_info

# Qwen2.5-VL 原始模型不使用 video_metadata
images, videos, video_kwargs = process_vision_info(
    messages, 
    return_video_kwargs=True
)

inputs = processor(
    text=[text],
    images=images,
    videos=videos,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)
```

## 5. 完整示例：构建 Grounding 任务输入

```python
import json

# 定义查询
query = "a man wearing a blue jacket approaches a blue car"
video_path = "/data/videos/example.mp4"

# 构建 messages
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "min_pixels": 64 * 28 * 28,      # min_tokens * 28 * 28
                "total_pixels": 14336 * 28 * 28, # total_tokens * 28 * 28
                "fps": 2
            },
            {
                "type": "text",
                "text": f"You are given a video with multiple frames. "
                        f"The numbers before each video frame indicate its sampling timestamp (in seconds). "
                        f"Please find the visual event described by the sentence '{query}', "
                        f"determining its starting and ending times. "
                        f"The format should be: 'The event happens in <start time> - <end time> seconds'."
            }
        ]
    }
]

print(json.dumps(messages, indent=2, ensure_ascii=False))
```

输出：
```json
[
  {
    "role": "user",
    "content": [
      {
        "type": "video",
        "video": "/data/videos/example.mp4",
        "min_pixels": 50176,
        "total_pixels": 11239424,
        "fps": 2
      },
      {
        "type": "text",
        "text": "You are given a video with multiple frames. The numbers before each video frame indicate its sampling timestamp (in seconds). Please find the visual event described by the sentence 'a man wearing a blue jacket approaches a blue car', determining its starting and ending times. The format should be: 'The event happens in <start time> - <end time> seconds'."
      }
    ]
  }
]
```

## 6. 模型对比总结

| 模型 | 时间戳编码方式 | 关键参数 | Prompt 特点 |
|------|---------------|----------|-------------|
| **TimeLens-7B** | Interleaved Textual Timestamps | `return_video_metadata=True` | 包含时间戳说明前缀 |
| **TimeLens-8B** | Video Metadata | `video_metadata=video_metadatas` | 标准 Prompt |
| **Qwen2.5-VL** | 无时间戳增强 | `return_video_kwargs=True` | 标准 Prompt |
| **Qwen3-VL** | Video Metadata | `video_metadata=video_metadatas` | 标准 Prompt |

## 7. 模型输出格式

所有模型的输出格式统一为：
```
The event happens in <start_time> - <end_time> seconds.
```

例如：
```
The event happens in 0.0 - 5.0 seconds.
```
