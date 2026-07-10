# freebuff2api

Codebuff Freebuff 的 OpenAI-compatible API

## 接口

| 端点 | 协议 | 认证方式 |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | `Authorization: Bearer <key>` |
| `POST /v1/messages` | Anthropic | `x-api-key: <key>` |
| `GET /v1/models` | 通用 | — |
| `GET /healthz` | 通用 | — |

模型名支持简写（如 `deepseek-v4-flash` 等同于 `deepseek/deepseek-v4-flash`）。

## 配置

### 获取 Token

无需安装 Freebuff / Codebuff CLI，可以直接打开公开页面自动获取 token：

```text
https://freebuff.071129.xyz/
```

使用方式：

1. 打开上面的地址
2. 选择 Freebuff
3. 点击“开始认证”，在跳转页面完成授权
4. 回到页面复制展示的 token
5. 将复制结果写入本项目 `.env`

示例：

```dotenv
FREEBUFF_TOKEN=你的 Freebuff Bearer token
```

多账号可用英文逗号分隔；并发请求会优先分配到空闲账号，避免单个
Freebuff 账号的全局 active free session 被并发切模型请求互相覆盖：

```dotenv
FREEBUFF_TOKEN=token-a,token-b,token-c
```

复制 `.env.example` 为 `.env`，然后填写上游 token：

```powershell
Copy-Item .env.example .env
```

`.env` 示例：

```dotenv
FREEBUFF_TOKEN=你的 Freebuff Bearer token
FREEBUFF_API_KEY=本地 OpenAI API key，可留空
FREEBUFF_AD_PROVIDERS=gravity,zeroclick
FREEBUFF_PROXY_ENABLED=false
FREEBUFF_PROXY_URL=
FREEBUFF_DEBUG=false
FREEBUFF_LOG_LEVEL=INFO
FREEBUFF_LOG_BODY_CHARS=2000
FREEBUFF_LOG_COLOR=true
FREEBUFF_HOST=0.0.0.0
FREEBUFF_PORT=8000
```

默认不启用代理，所有上游请求直连，且不会读取系统 `HTTP_PROXY` / `HTTPS_PROXY`。

需要让所有上游请求经过代理时，在 `.env` 中开启：

```dotenv
FREEBUFF_PROXY_ENABLED=true
FREEBUFF_PROXY_URL=http://127.0.0.1:7890
```

支持 HTTP 和 SOCKS 代理，例如：

```dotenv
FREEBUFF_PROXY_URL=http://127.0.0.1:7890
FREEBUFF_PROXY_URL=socks5://127.0.0.1:1080
FREEBUFF_PROXY_URL=socks5h://127.0.0.1:1080
```

当前内置 Freebuff 模型：

- `deepseek/deepseek-v4-flash`
- `deepseek/deepseek-v4-pro`
- `moonshotai/kimi-k2.6`
- `minimax/minimax-m2.7`
- `minimax/minimax-m3`
- `google/gemini-2.5-flash-lite`
- `google/gemini-3.1-flash-lite-preview`
- `google/gemini-3.1-pro-preview`
- `mimo/mimo-v2.5`
- `mimo/mimo-v2.5-pro`

调试空返回或上游异常时：

```dotenv
FREEBUFF_DEBUG=true
FREEBUFF_LOG_LEVEL=DEBUG
FREEBUFF_LOG_BODY_CHARS=0
```

## 运行

```powershell
uv sync
uv run freebuff2api
```

或：

```powershell
python -m pip install -e .
python main.py
```

## Docker 部署

```yaml
# docker-compose.yml
services:
  freebuff2api:
    build: .
    image: freebuff2api:latest
    container_name: freebuff2api
    restart: always
    ports:
      - "8000:8000"
    volumes:
      - ./.env:/app/.env
```

启动：

```bash
docker compose up -d
```

### GitHub Actions 自动构建

推送 `main` 分支或打 `v*` tag 时自动构建并推送到 Docker Hub。

在仓库 Secrets 添加：

| Secret | 说明 |
|---|---|
| `DOCKER_USERNAME` | Docker Hub 用户名 |
| `DOCKER_PASSWORD` | Docker Hub 密码或 token |

## 调用示例

```powershell
curl http://127.0.0.1:8000/v1/chat/completions `
  -H "Authorization: Bearer $env:FREEBUFF_API_KEY" `
  -H "Content-Type: application/json" `
  -d '{
    "model": "deepseek/deepseek-v4-flash",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": false
  }'
```

流式：

```powershell
curl -N http://127.0.0.1:8000/v1/chat/completions `
  -H "Authorization: Bearer $env:FREEBUFF_API_KEY" `
  -H "Content-Type: application/json" `
  -d '{
    "model": "deepseek/deepseek-v4-flash",
    "messages": [{"role": "user", "content": "写一个 Python 快排"}],
    "stream": true
  }'
```

### Python (Anthropic SDK)

```python
from anthropic import Anthropic

client = Anthropic(
    api_key="你的 FREEBUFF_API_KEY",
    base_url="http://127.0.0.1:8000",
)

# 非流式
msg = client.messages.create(
    model="deepseek/deepseek-v4-flash",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "你好"}],
)
print(msg.content[0].text)

# 流式
with client.messages.stream(
    model="deepseek-v4-flash",
    max_tokens=1024,
    messages=[{"role": "user", "content": "数到3"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="")
```

## 感谢

> [FreeBuff](https://freebuff.com)
