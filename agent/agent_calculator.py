"""
LangChain Agent with Calculator Tool

使用LangChain框架创建一个带计算器工具的Agent，通过vLLM提供的OpenAI兼容API进行推理。

Usage:
    1. 首先启动vLLM服务器:
       python agent/serve_vllm.py
    
    2. 然后运行Agent:
       python agent/agent_calculator.py
"""
import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor


# =========================
# 可配置参数
# =========================
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "lora-adapter")  # 使用LoRA模型，或 "qwen" 使用基础模型
TEMPERATURE = 0.7
MAX_TOKENS = 1024


# =========================
# 工具定义
# =========================
@tool
def calculator(expression: str) -> str:
    """
    计算数学表达式。
    
    输入应为有效的数学表达式字符串，例如:
    - "2 + 3"
    - "10 * 5"
    - "(12 + 8) * 3"
    - "100 / 4 - 5"
    - "2 ** 10" (幂运算)
    
    Args:
        expression: 数学表达式字符串
        
    Returns:
        计算结果或错误信息
    """
    # 安全的数学操作
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sum": sum,
    }
    
    try:
        # 使用受限的eval环境，只允许数学运算
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except ZeroDivisionError:
        return "错误: 除数不能为零"
    except SyntaxError:
        return f"错误: 表达式语法不正确 - {expression}"
    except Exception as e:
        return f"计算错误: {str(e)}"


# =========================
# Agent 创建
# =========================
def create_agent(
    base_url: str = VLLM_BASE_URL,
    model_name: str = MODEL_NAME,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> AgentExecutor:
    """
    创建带计算器工具的Agent。
    
    Args:
        base_url: vLLM服务器地址
        model_name: 模型名称（使用LoRA时为 "lora-adapter"）
        temperature: 生成温度
        max_tokens: 最大生成token数
        
    Returns:
        AgentExecutor 实例
    """
    # 1. 创建LLM连接
    llm = ChatOpenAI(
        base_url=base_url,
        api_key="not-needed",  # vLLM不需要真实的API key
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # 2. 定义工具列表
    tools = [calculator]
    
    # 3. 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个有用的AI助手，可以使用工具来帮助用户解决问题。

当用户提出数学计算问题时，请使用calculator工具来计算结果。
在回答时，先说明你要做什么，然后使用工具，最后给出完整的答案。"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # 4. 创建Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # 5. 创建AgentExecutor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # 显示详细执行过程
        handle_parsing_errors=True,  # 处理解析错误
        max_iterations=5,  # 最大迭代次数
    )
    
    return agent_executor


def run_single_query(agent: AgentExecutor, query: str) -> str:
    """
    运行单个查询。
    
    Args:
        agent: AgentExecutor实例
        query: 用户输入
        
    Returns:
        Agent的响应
    """
    try:
        result = agent.invoke({"input": query})
        return result.get("output", "无法获取响应")
    except Exception as e:
        return f"执行错误: {str(e)}"


def interactive_chat():
    """交互式聊天循环。"""
    print("=" * 60)
    print("LangChain Agent with Calculator Tool")
    print("=" * 60)
    print(f"vLLM Server: {VLLM_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)
    print("Commands:")
    print("  - Type your question to interact with the agent")
    print("  - Type 'quit' or 'exit' to quit")
    print("  - Type 'help' for example queries")
    print("=" * 60)
    
    # 创建Agent
    print("\nInitializing agent...")
    try:
        agent = create_agent()
        print("Agent ready!\n")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Make sure the vLLM server is running: python agent/serve_vllm.py")
        return
    
    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit"]:
            print("\nGoodbye!")
            break
        
        if user_input.lower() == "help":
            print("\nExample queries:")
            print("  - 计算 (12 + 8) * 3")
            print("  - 100除以4再减去5等于多少？")
            print("  - 2的10次方是多少？")
            print("  - 帮我算一下 15 * 7 + 23")
            print()
            continue
        
        print()
        response = run_single_query(agent, user_input)
        print(f"\nAssistant: {response}\n")


def main():
    """主入口函数。"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LangChain Agent with Calculator Tool")
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to run (non-interactive mode)"
    )
    parser.add_argument(
        "--base-url",
        default=VLLM_BASE_URL,
        help=f"vLLM server base URL (default: {VLLM_BASE_URL})"
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Model name (default: {MODEL_NAME})"
    )
    
    args = parser.parse_args()
    
    if args.query:
        # 非交互模式：执行单个查询
        agent = create_agent(base_url=args.base_url, model_name=args.model)
        result = run_single_query(agent, args.query)
        print(result)
    else:
        # 交互模式
        interactive_chat()


if __name__ == "__main__":
    main()
