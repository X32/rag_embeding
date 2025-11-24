import os
from openai import OpenAI
# from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
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
def main():
    # 第 1 轮
    messages.append({"role": "user", "content": "推荐一部关于太空探索的科幻电影。"})
    print("第1轮")
    print(f"用户：{messages[0]['content']}")
    assistant_output = get_response(messages)
    messages.append({"role": "assistant", "content": assistant_output})
    print(f"模型：{assistant_output}\n")

    # 第 2 轮
    messages.append({"role": "user", "content": "这部电影的导演是谁？"})
    print("第2轮")
    print(f"用户：{messages[-1]['content']}")
    assistant_output = get_response(messages)
    messages.append({"role": "assistant", "content": assistant_output})
    print(f"模型：{assistant_output}\n")


# 新增：LangChain问答工作流函数
def langchain_qa_workflow():
    # 初始化LangChain的ChatOpenAI
    llm = ChatOpenAI(
        model_name="qwen-plus",
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 预设问题列表
    questions = [
        "推荐一部关于太空探索的科幻电影。",
        "这部电影的导演是谁？",
        "这个导演还导演过哪些影片？"
    ]

    # 用于存储正确答案的参考（实际应用中可从知识库获取）
    reference_answers = []
    user_answers = []
    scores = []
    total_score = 0

    print("=== LangChain定制问答工作流 ===")

    for i, question in enumerate(questions, 1):
        print(f"\n问题{i}/{len(questions)}: {question}")

        # 获取模型对问题的预期答案（作为评分参考）
        if i == 1:
            # 第一个问题获取推荐电影
            ref_msg = [{"role": "user", "content": question}]
            ref_answer = get_response(ref_msg)
            reference_answers.append(ref_answer)
            print(f"模型推荐参考: {ref_answer}")

        # 获取用户回答
        user_answer = input("请输入您的答案: ")
        user_answers.append(user_answer)



        # 构建评分提示模板
        if i == 1:
            # 判断电影推荐是否合理
            score_prompt = ChatPromptTemplate.from_template("""
            任务：判断用户推荐的太空探索科幻电影是否合理。
            用户答案：{user_answer}
            参考推荐：{ref_answer}
            判断标准：只要是合理的太空探索科幻电影即可视为正确。
            输出格式：只输出"正确"或"错误"，不需要额外解释。
            """)
        elif i == 2:
            # 判断导演是否正确（基于第一个问题的推荐）
            score_prompt = ChatPromptTemplate.from_template("""
            任务：判断用户回答的电影导演是否正确。
            电影名称：{movie_name}
            用户答案：{user_answer}
            输出格式：只输出"正确"或"错误"，不需要额外解释。
            """)
        else:  # i == 3
            # 判断导演其他作品是否正确
            score_prompt = ChatPromptTemplate.from_template("""
            任务：判断用户回答的导演作品是否正确。
            导演名称：{director_name}
            用户答案：{user_answer}
            输出格式：只输出"正确"或"错误"，不需要额外解释。
            """)

        # 创建评分链并执行
        # score_chain = LLMChain(llm=llm, prompt=score_prompt)
        score_chain = score_prompt | llm
        if i == 1:
            # score_result = score_chain.run(user_answer=user_answer, ref_answer=ref_answer)
            score_result = score_chain.invoke({  # 使用invoke代替run
                "user_answer": user_answer,
                "ref_answer": ref_answer
            }).content  # 获取content属性
        elif i == 2:
            # 从参考答案中提取电影名称（简单处理，实际可能需要NLP提取）
            movie_name = reference_answers[0].split("《")[1].split("》")[0] if "《" in reference_answers[0] else \
            reference_answers[0]
            # 原代码：score_result = score_chain.run(movie_name=movie_name, user_answer=user_answer)
            score_result = score_chain.invoke({  # 使用invoke代替run
                "movie_name": movie_name,
                "user_answer": user_answer
            }).content  # 获取content属性
            # 保存导演名称用于第三个问题
            director_name = get_response([{"role": "user", "content": f"{movie_name}的导演是谁？"}])
            reference_answers.append(director_name)
        else:
            # score_result = score_chain.run(director_name=director_name, user_answer=user_answer)
            # 原代码：score_result = score_chain.run(director_name=director_name, user_answer=user_answer)
            score_result = score_chain.invoke({  # 使用invoke代替run
                "director_name": director_name,
                "user_answer": user_answer
            }).content  # 获取content属性

        # 判断结果并计分
        if "正确" in score_result:
            scores.append(10)
            total_score += 10
            print(f"回答正确！+10分")
        else:
            scores.append(0)
            print(f"回答错误！0分")
            # 显示正确答案
            # if i > 1:
            #     print(f"正确答案参考: {reference_answers[i - 1]}")

    # 输出最终得分
    print("\n=== 问答结束 ===")
    print("您的回答与得分:")
    for i in range(len(questions)):
        print(f"问题{i + 1}: {questions[i]}")
        print(f"您的答案: {user_answers[i]}")
        print(f"得分: {scores[i]}\n")
    print(f"总得分: {total_score}/{len(questions) * 10}")
# def langchain_memory():
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
    import os
    from openai import OpenAI

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



if __name__ == '__main__':
    langchain_qa_workflow_1()
    # main()