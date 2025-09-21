# IndexTTS API 服务器

基于 IndexTTS2 模型的 REST API 服务器，提供文本转语音功能。

## 功能特性

- **异步和同步TTS生成**：支持异步任务和同步请求
- **多种情感控制方式**：
  - 与音色参考音频相同
  - 使用情感参考音频
  - 使用情感向量控制（8维情感向量）
  - 使用情感描述文本控制
- **文件上传管理**：支持音频文件上传和管理
- **任务状态跟踪**：实时查看生成进度和状态
- **高级生成参数**：支持采样参数、束搜索等配置

## 安装依赖

本项目使用 `uv` 进行依赖管理。安装API服务器依赖：

```bash
# 安装API服务器依赖
uv sync --extra api

# 或者安装所有可选依赖（包括webui和deepspeed）
uv sync --all-extras
```

如果您使用传统的pip，也可以手动安装：

```bash
pip install "fastapi>=0.104.0" "uvicorn[standard]>=0.24.0" "pydantic>=2.0.0" "python-multipart>=0.0.6"
```

## 启动服务器

```bash
python api.py --model_dir ./checkpoints --port 8000 --host 0.0.0.0
```

### 启动参数

- `--host`: 服务器主机地址（默认: 0.0.0.0）
- `--port`: 服务器端口（默认: 8000）
- `--model_dir`: 模型检查点目录（默认: ./checkpoints）
- `--fp16`: 使用FP16推理
- `--deepspeed`: 使用DeepSpeed加速
- `--cuda_kernel`: 使用CUDA内核
- `--workers`: 工作进程数（默认: 1）
- `--reload`: 开发模式热重载

## API 文档

启动服务器后，访问 `http://localhost:8000/docs` 查看完整的 API 文档。

## 主要 API 端点

### 1. 健康检查
```
GET /api/v1/health
```

### 2. 异步TTS生成
```
POST /api/v1/tts
```

### 3. 同步TTS生成
```
POST /api/v1/tts/sync
```

### 4. 查询任务状态
```
GET /api/v1/task/{task_id}
```

### 5. 获取生成的音频
```
GET /api/v1/audio/{task_id}
```

### 6. 上传音频文件
```
POST /api/v1/upload/audio
```

### 7. 获取文件信息
```
GET /api/v1/file/{file_id}
```

### 8. 任务管理
```
GET /api/v1/tasks          # 列出任务
DELETE /api/v1/task/{task_id}  # 删除任务
```

## 使用示例

### Python 客户端示例

```python
import requests
import time

# 服务器地址
BASE_URL = "http://localhost:8000"

# 1. 上传音色参考音频
with open("prompt.wav", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/api/v1/upload/audio",
        files={"file": f}
    )
    prompt_file_id = response.json()["file_id"]

# 2. 验证文件上传成功（可选）
file_info = requests.get(f"{BASE_URL}/api/v1/file/{prompt_file_id}")
print(f"上传的文件信息: {file_info.json()}")

# 3. 异步生成TTS
tts_request = {
    "text": "你好，这是一个测试语音。",
    "prompt_audio_id": prompt_file_id,
    "emo_control_method": 0,
    "max_text_tokens_per_segment": 120
}

response = requests.post(f"{BASE_URL}/api/v1/tts", json=tts_request)
task_id = response.json()["task_id"]

# 4. 查询任务状态
while True:
    response = requests.get(f"{BASE_URL}/api/v1/task/{task_id}")
    status = response.json()
    
    if status["status"] == "completed":
        print("生成完成！")
        # 下载音频
        audio_response = requests.get(f"{BASE_URL}/api/v1/audio/{task_id}")
        with open("output.wav", "wb") as f:
            f.write(audio_response.content)
        break
    elif status["status"] == "failed":
        print(f"生成失败: {status['message']}")
        break
    else:
        print(f"进度: {status['progress']*100:.1f}%")
        time.sleep(1)
```

### curl 示例

```bash
# 先上传音频文件
curl -X POST "http://localhost:8000/api/v1/upload/audio" \
  -F "file=@prompt.wav"

# 使用返回的file_id进行TTS生成
curl -X POST "http://localhost:8000/api/v1/tts/sync" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是一个测试语音。",
    "prompt_audio_id": "your-file-id-here",
    "emo_control_method": 0,
    "max_text_tokens_per_segment": 120
  }'
```

## 情感控制说明

### 情感控制方式

- `0`: 与音色参考音频相同
- `1`: 使用情感参考音频（需要提供 `emo_audio_path`）
- `2`: 使用情感向量控制（需要提供 `emo_vector`）
- `3`: 使用情感描述文本控制（需要提供 `emo_text`）

### 情感向量

8维向量，分别对应：[喜, 怒, 哀, 惧, 厌恶, 低落, 惊喜, 平静]
每个值范围为 0.0-1.0。

示例：
```json
{
  "emo_control_method": 2,
  "emo_vector": [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.3]
}
```

## 安全性和去重说明

### 🔒 安全性
- **SHA哈希ID系统**：API使用SHA256哈希值作为文件ID，确保文件唯一性和安全性
- **文件验证**：所有文件ID都必须是64位十六进制SHA256哈希值，经过严格验证
- **路径安全**：内部自动拼接安全的文件路径，防止路径遍历攻击
- **文件格式限制**：只允许上传特定格式的音频文件（.wav, .mp3, .m4a, .flac）

### 🔄 文件去重
- **内容哈希**：使用文件内容的SHA256哈希值作为唯一标识
- **自动去重**：相同内容的文件只会保存一份，节省存储空间
- **重复检测**：上传重复文件时会返回现有文件的信息，标记为`is_duplicate: true`

## 注意事项

1. **文件管理**：上传的文件会在1小时后自动清理，生成的音频文件会在24小时后清理
2. **并发限制**：建议根据GPU显存调整worker数量
3. **音频格式**：支持上传 .wav, .mp3, .m4a, .flac 格式的音频文件
4. **文本长度**：长文本会自动分句处理，可通过 `max_text_tokens_per_segment` 调整分句长度

## 错误处理

API 使用标准的 HTTP 状态码：

- `200`: 成功
- `400`: 请求参数错误
- `404`: 资源不存在
- `500`: 服务器内部错误

错误响应格式：
```json
{
  "success": false,
  "message": "错误描述"
}
```
