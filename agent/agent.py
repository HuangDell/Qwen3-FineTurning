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
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
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
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
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


@tool
def get_time(zone: str) -> str:
    """
    获取当前北京时间。
     
    此工具返回当前的北京时间（东八区），格式为：YYYY-MM-DD HH:MM:SS
    
    Returns:
        当前北京时间的字符串表示
    """
    try:
        # 获取北京时间（东八区 UTC+8）
        beijing_tz = ZoneInfo(zone)
        current_time = datetime.now(beijing_tz)
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][current_time.weekday()]
        return f"{formatted_time} {weekday}"
    except Exception as e:
        return f"获取时间失败: {str(e)}"


@tool
def get_weather(location: str, forecast_days: int = 0) -> str:
    """
    获取指定城市的天气信息。
    
    此工具查询任意城市的实时天气或未来天气预报。
    支持中文和英文城市名称。
    
    Args:
        location: 城市名称（中文或英文），例如："Shanghai", "Beijing", "上海", "New York", "东京"
        forecast_days: 预报天数，0表示当前天气，1-5表示未来第N天的预报（可选，默认0）
    
    Returns:
        指定城市的天气信息，包括温度、天气状况、湿度等
    """
    if not OPENWEATHER_API_KEY:
        return "错误: 未设置OpenWeatherMap API密钥。请设置环境变量 OPENWEATHER_API_KEY"
    
    try:
        # 参数验证
        if forecast_days < 0 or forecast_days > 5:
            return "错误: forecast_days 参数必须在 0-5 之间"
        
        # 步骤1: 使用 Geocoding API 将城市名转换为经纬度
        geocoding_url = "http://api.openweathermap.org/geo/1.0/direct"
        geocoding_params = {
            "q": location,
            "limit": 1,  # 只获取最匹配的一个结果
            "appid": OPENWEATHER_API_KEY
        }
        
        geo_response = requests.get(geocoding_url, params=geocoding_params, timeout=5)
        
        if geo_response.status_code != 200:
            if geo_response.status_code == 401:
                return "错误: API密钥无效，请检查 OPENWEATHER_API_KEY"
            elif geo_response.status_code == 429:
                return "错误: API调用次数超过限制，请稍后再试"
            else:
                return f"无法获取城市坐标，服务返回状态码: {geo_response.status_code}"
        
        geo_data = geo_response.json()
        
        if not geo_data or len(geo_data) == 0:
            return f"错误: 找不到城市 '{location}'，请检查城市名称是否正确"
        
        # 提取经纬度和城市信息
        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]
        city_name = geo_data[0]["name"]
        country = geo_data[0].get("country", "")
        
        # 获取中文城市名（如果有）
        local_names = geo_data[0].get("local_names", {})
        city_display_name = local_names.get("zh", city_name)  # 优先使用中文名
        
        # 步骤2: 使用经纬度查询天气
        if forecast_days == 0:
            # 获取当前天气
            weather_url = "https://api.openweathermap.org/data/2.5/weather"
            weather_params = {
                "lat": lat,
                "lon": lon,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric",  # 使用摄氏度
                "lang": "zh_cn"     # 中文天气描述
            }
            
            weather_response = requests.get(weather_url, params=weather_params, timeout=5)
            
            if weather_response.status_code == 200:
                data = weather_response.json()
                
                temp = data["main"]["temp"]
                feels_like = data["main"]["feels_like"]
                humidity = data["main"]["humidity"]
                weather_desc = data["weather"][0]["description"]
                
                weather_info = (
                    f"{city_display_name}（{country}）天气：{weather_desc}\n"
                    f"温度：{temp}°C (体感温度：{feels_like}°C)\n"
                    f"湿度：{humidity}%"
                )
                return weather_info
            elif weather_response.status_code == 401:
                return "错误: API密钥无效，请检查 OPENWEATHER_API_KEY"
            elif weather_response.status_code == 429:
                return "错误: API调用次数超过限制，请稍后再试"
            else:
                return f"无法获取天气信息，服务返回状态码: {weather_response.status_code}"
        else:
            # 获取天气预报
            forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
            forecast_params = {
                "lat": lat,
                "lon": lon,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric",
                "lang": "zh_cn"
            }
            
            forecast_response = requests.get(forecast_url, params=forecast_params, timeout=5)
            
            if forecast_response.status_code == 200:
                data = forecast_response.json()
                
                # 获取目标日期的预报数据（每3小时一个数据点）
                # 选择目标日期中午12点左右的数据
                forecast_list = data["list"]
                target_index = min(forecast_days * 8, len(forecast_list) - 1)  # 每天8个数据点（3小时间隔）
                
                forecast = forecast_list[target_index]
                temp = forecast["main"]["temp"]
                feels_like = forecast["main"]["feels_like"]
                humidity = forecast["main"]["humidity"]
                weather_desc = forecast["weather"][0]["description"]
                forecast_time = forecast["dt_txt"]
                
                weather_info = (
                    f"{city_display_name}（{country}）天气预报（{forecast_time}）：{weather_desc}\n"
                    f"预计温度：{temp}°C (体感温度：{feels_like}°C)\n"
                    f"预计湿度：{humidity}%"
                )
                return weather_info
            elif forecast_response.status_code == 401:
                return "错误: API密钥无效，请检查 OPENWEATHER_API_KEY"
            elif forecast_response.status_code == 429:
                return "错误: API调用次数超过限制，请稍后再试"
            else:
                return f"无法获取天气预报，服务返回状态码: {forecast_response.status_code}"
                
    except requests.exceptions.Timeout:
        return "获取天气信息超时"
    except requests.exceptions.ConnectionError:
        return "网络连接错误，无法访问天气服务"
    except Exception as e:
        return f"获取天气失败: {str(e)}"


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
    tools = [calculator, get_time, get_weather]
    
    # 3. 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个有用的AI助手，可以使用工具来帮助用户解决问题。

重要规则：
1. 每次回答用户问题时，你必须首先调用 get_time 工具获取当前某个地区时间，并在最终回答中显示当前时间。
2. 地区你需要根据用户的问题来决定，例如用户问"现在几点了？"时，你可以使用"Asia/Shanghai"作为参数调用 get_time 工具。
2. 当用户提出数学计算问题时，使用 calculator 工具来计算结果。
3. 当用户询问天气时，使用 get_weather 工具，需要提取用户询问的城市名称作为location参数。如果用户询问未来天气，可以使用forecast_days参数（1-5天）。
4. 在回答时保持友好和专业的态度。

回答格式示例：
当前时间：[从 get_time 获取的时间]
[根据用户问题给出的具体回答]"""),
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
            print("  计算相关：")
            print("    - 计算 (12 + 8) * 3")
            print("    - 100除以4再减去5等于多少？")
            print("    - 2的10次方是多少？")
            print("    - 帮我算一下 15 * 7 + 23")
            print("  时间相关：")
            print("    - 现在几点了？")
            print("    - 今天是什么日期？")
            print("  天气相关：")
            print("    - 北京现在天气怎么样？")
            print("    - 上海明天天气如何？")
            print("    - 纽约的天气怎么样？")
            print("    - What's the weather in Tokyo?")
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
