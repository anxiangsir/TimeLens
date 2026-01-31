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
