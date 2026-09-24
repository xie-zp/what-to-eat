"""第一课：不依赖 LangChain，直接调用 OpenAI 兼容 API。"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def build_client() -> OpenAI:
    """从环境变量创建 OpenAI 兼容客户端。"""

    api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("缺少 LLM_API_KEY 或 DEEPSEEK_API_KEY。请先根据 .env.example 创建 .env。")

    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
    )


def ask(question: str) -> str:
    """发送一条用户消息，并返回模型文本回答。"""

    client = build_client()
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "deepseek-v4-flash"),
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        temperature=0,
        max_tokens=2000,
    )
    return response.choices[0].message.content or ""


def main() -> None:
    """运行第一课示例。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question",
        nargs="?",
        default="用一句话说明什么是大语言模型。",
    )
    args = parser.parse_args()

    print(f"用户：{args.question}")
    print(f"模型：{ask(args.question)}")


if __name__ == "__main__":
    main()
