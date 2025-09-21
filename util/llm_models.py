# ... 现有导入代码 ...
import os
import dashscope
from dashscope import Generation  # 添加DashScope Generation导入

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


def function_call_predict(messages,functions, model='qwen-turbo'):
    # 创建回答
    # DashScope对话模型调用
    response = Generation.call(
        model=model,  # 使用DashScope的对话模型，如通义千问
        messages=messages,  # 保持与OpenAI相同的messages格式
        functions=functions,  # 函数定义格式与OpenAI兼容
        function_call="auto"  # 自动函数调用参数
    )
    # print(f"函数调用返回: {response}")
    # 修复1：添加API调用失败处理
    if response.status_code != 200:
        print(f"API调用失败: {response.message}")
        return -1
    # 修复2：检查output是否存在
    if not response.output or not response.output.choices:
        print("API返回结果为空")
        return -1
    message = response.output.choices[0].message

    # 提取function_call
    function_call = message.get('function_call')
    if function_call:
        # 提取函数名和参数
        function_name = function_call.get('name')
        function_args = function_call.get('arguments')
        print(f"函数调用: {function_name}, 参数: {function_args}")
        # res = type_dict[function_name]
        return function_name
    else:
        # 普通文本回复处理
        result = message.get('content')
        print(result)
        return -1

