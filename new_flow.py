import os
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

def get_response(messages):
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=messages
    )
    return completion.choices[0].message.content

def get_score(result_str):
    """
    根据判断结果字符串返回对应分数
    :param result_str: 模型返回的判断结果（"正确"或"错误"）
    :return: 10（正确）或0（错误）
    """
    # 去除首尾空白字符后判断
    if result_str.strip() == "正确":
        return 10
    else:
        return 0
def langchain_qa_workflow_1():


    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        # 以下是北京地域base_url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    def get_response(messages):
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=messages
        )
        return completion.choices[0].message.content

    # 初始化 messages
    messages = []

    # 第 1 轮
    messages.append({"role": "user", "content": "推荐一部关于太空探索的科幻电影。"})
    print("第1轮")
    print(f"用户：{messages[0]['content']}")
    user_answer = input("请输入您的答案: ")
    # answers_str = ""
    messages.append({"role": "assistant", "content": user_answer})
    assistant_answer = ' 任务：判断用户推荐的太空探索科幻电影是否合理。判断标准：只要是合理的太空探索科幻电影即可视为正确。输出格式：只输出"正确"或"错误"，不需要额外解释。'
    messages.append({"role": "user", "content": assistant_answer})
    assistant_output = get_response(messages)
    print(f"模型：{assistant_output}\n")
    score = get_score(assistant_output)
    print(f"得分: {score}")
    total_score = 0
    total_score += score
    # messages.append({"role": "assistant", "content": assistant_output})
    # print(f"模型：{assistant_output}\n")
    #
    # # 第 2 轮
    messages.append({"role": "user", "content": "这部电影的导演是谁？"})
    print("第2轮")
    print(f"用户：""这部电影的导演是谁")
    user_answer1 = input("请输入您的答案: ")
    messages.append({"role": "assistant", "content": user_answer1})

    assistant_answer1 = ' 任务：判断用户回答的电影导演是否正确。输出格式：只输出"正确"或"错误"，不需要额外解释。'
    messages.append({"role": "user", "content": assistant_answer1})
    assistant_output = get_response(messages)
    print(f"模型：{assistant_output}\n")
    score = get_score(assistant_output)
    print(f"得分: {score}")
    total_score += score

    # # 第 3 轮
    messages.append({"role": "user", "content": "这位导演还拍过什么电影"})
    print("第3轮")
    print(f"用户：""这位导演还拍过什么电影")
    user_answer2 = input("请输入您的答案: ")
    messages.append({"role": "assistant", "content": user_answer2})

    assistant_answer2 = ' 任务：判断用户回答的导演作品是否正确。 输出格式：只输出"正确"或"错误"，不需要额外解释。'
    messages.append({"role": "user", "content": assistant_answer2})
    assistant_output = get_response(messages)
    print(f"模型：{assistant_output}\n")

    score = get_score(assistant_output)
    print(f"得分: {score}")
    total_score += score
    print(f"总得分: {total_score}/{3 * 10}")


def langchain_qa_workflow_2():

    # 初始化 messages
    messages = []
    # 统一存储问题和对应的评分提示模板（新增）
    qa_configs = [
        {
            "question": "推荐一部关于太空探索的科幻电影。",
            "assessment_prompt": "任务：判断用户推荐的太空探索科幻电影是否合理。判断标准：只要是合理的太空探索科幻电影即可视为正确。输出格式：只输出\"正确\"或\"错误\"，不需要额外解释。"
        },
        {
            "question": "这部电影的导演是谁？",
            "assessment_prompt": "任务：判断用户回答的电影导演是否正确。输出格式：只输出\"正确\"或\"错误\"，不需要额外解释。"
        },
        {
            "question": "这位导演还拍过什么电影？",
            "assessment_prompt": "任务：判断用户回答的导演作品是否正确。输出格式：只输出\"正确\"或\"错误\"，不需要额外解释。"
        }
    ]

    total_score = 0  # 初始化总分（移至循环前）
    scores = []  # 存储各题得分

    # 通过循环处理多轮问答（核心优化）
    for round_num, config in enumerate(qa_configs, 1):
        question = config["question"]
        assessment_prompt = config["assessment_prompt"]

        # 提问流程
        print(f"\n第{round_num}轮")
        messages.append({"role": "user", "content": question})
        print(f"用户：{question}")

        # 获取用户回答
        user_answer = input("请输入您的答案: ")
        messages.append({"role": "assistant", "content": user_answer})

        # 评分流程
        messages.append({"role": "user", "content": assessment_prompt})
        assistant_output = get_response(messages)
        print(f"模型：{assistant_output}\n")

        # 计算得分
        score = get_score(assistant_output)
        scores.append(score)
        total_score += score
        print(f"得分: {score}")

    # 输出最终结果
    print("\n=== 问答结束 ===")
    for i, (score, config) in enumerate(zip(scores, qa_configs), 1):
        print(f"第{i}题: {config['question']}")
        print(f"得分: {score}\n")
    print(f"总得分: {total_score}/{len(qa_configs) * 10}")

if __name__ == '__main__':
    langchain_qa_workflow_2()