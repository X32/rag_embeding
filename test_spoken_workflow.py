import os
from openai import OpenAI

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

def fce_qa_workflow_part1():

    # 初始化 messages
    messages = []
    assessment_prompt = "任务:根据FCE口语part1的要求，给用户的回答进行评分 判断标准：是否体现出英语水平。满分15分 输出格式：只输出分数数字，不需要额外解释。"

    assessment_prompt_explan = "对刚才的评分进行解释，说明用户回答的英语水平是否符合要求。"
    # 统一存储问题和对应的评分提示模板（新增）
    qa_configs = [
        {
            "question": "Do you prefer to travel by bus or by car?",
            "assessment_prompt":""
        },
        {
            "question": "What do you like to do on a long journey?",
            "assessment_prompt": ""
        },
        {
            "question": "What's the best way to travel around the place where you live?",
            "assessment_prompt": ""
        }
    ]

    total_score = 0  # 初始化总分（移至循环前）
    scores = []  # 存储各题得分
    scores_explanations = []
    prompt_start = "任务：我们将进行英语口语问答。用于模拟FCE英语口语考试。以下将进行3轮问答，每轮问答后会根据用户回答和一些标准进行评分，最后输出总得分。"
    messages.append({"role": "user", "content": prompt_start})


    # 通过循环处理多轮问答（核心优化）
    for round_num, config in enumerate(qa_configs, 1):
        question = config["question"]
        # assessment_prompt = config["assessment_prompt"]

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
    print(f"得分: {assistant_output}\n")


    total_score += int(assistant_output)

    # 评分流程
    messages.append({"role": "user", "content": assessment_prompt_explan})
    assistant_output = get_response(messages)
    scores_explanations.append(assistant_output)
    print(f"解释：{assistant_output}\n")

def fce_qa_workflow_part2():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://www.qplay2.com/images/part2.jpeg"
                    },
                },
                {"type": "text",
                 "text": "图中是两张FCE口语考试模拟 Part 2 的图片，现在模拟英语口语考试，你是考官，请根据用户回答的问题评分"},
            ],
        }
    ]
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Which of these activities would you prefer to learn?"
            }
        ]
    })

    user_answer = input("请输入您的答案: ")

    messages.append({"role": "user", "content": user_answer})

    assessment_prompt = "任务:根据FCE口语part2的要求，给用户的回答进行评分 判断标准：是否体现出英语水平。满分15分 输出格式：只输出分数数字，不需要额外解释。"
    messages.append({"role": "user", "content": assessment_prompt})

    completion = client.chat.completions.create(
        model="qwen3-vl-plus",  # 可按需更换为其它多模态模型，并修改相应的 messages
        messages=messages,
    )

    print(f"第一轮输出得分：{completion.choices[0].message.content}")
    # messages.append({"role": "user", "content": "对刚才的评分进行分析解释"})

    assistant_message = completion.choices[0].message
    messages.append(assistant_message.model_dump())
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "对刚才的评分进行分析解释"
            }
        ]
    })
    completion = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=messages,
    )

    print(f"第二轮输出解释：{completion.choices[0].message.content}")
    assistant_message = completion.choices[0].message
    messages.append(assistant_message.model_dump())
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "请给用户刚才的回答，做一个优化版本，1 分钟之内可以说完的"
            }
        ]
    })
    completion = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=messages,
    )
    print(f"第三轮输出解释：{completion.choices[0].message.content}")

def fce_qa_workflow_part3():
    """ 第三轮问答 """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://www.qplay2.com/images/part1.jpeg"
                    },
                },
                {"type": "text",
                 # "text": "图中是两张FCE口语考试模拟 Part 3 的图片，现在模拟英语口语考试，你来当一个考生，我是另一个考生，咱们两个一起讨论这幅图片，并且最终达成一致，主动结束对话。你先开始"
                 "text": "图中是两张FCE口语考试模拟 Part 3 的图片，现在模拟英语口语考试：\n"
                         "1. 你我作为考生需依次讨论以下问题并最终达成一致\n"
                         "2. 所有问题讨论完毕后自动结束对话输出'结束对话'。你先开始"
                 },
            ],
        }
    ]

    completion = client.chat.completions.create(
        model="qwen3-vl-plus",  # 可按需更换为其它多模态模型，并修改相应的 messages
        messages=messages,
    )

    print(f"模型讨论：{completion.choices[0].message.content}")
    assistant_message = completion.choices[0].message
    messages.append(assistant_message.model_dump())
    # messages.append({"role": "user", "content": "对刚才的评分进行分析解释"})

    user_answer = input("请输入您的答案: ")

    messages.append({"role": "user", "content": user_answer})

    messages.append({"role": "user", "content": "轮到你了"})

    completion = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=messages,
    )

    print(f"第二轮，模型讨论：{completion.choices[0].message.content}")
    assistant_message = completion.choices[0].message
    messages.append(assistant_message.model_dump())

    user_answer = input("请输入您的答案: ")

    messages.append({"role": "user", "content": user_answer})

    messages.append({"role": "user", "content": "轮到你了"})

    completion = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=messages,
    )

    print(f"第三轮：{completion.choices[0].message.content}")

    return 0


def fce_qa_workflow_part3_by_pic( mage_url: str,degree: str):
    """ 第三轮问答：循环对话直至双方达成一致 """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": mage_url}
                },
                {"type": "text",
                 # "text": "图中是两张FCE口语考试模拟 Part 3 的图片，现在模拟英语口语考试，你来当一个考生，我是另一个考生，咱们两个一起讨论这幅图片并最终达成一致。每次回答后若同意对方观点请明确说出'同意'。你先开始"
                 "text": "图中是两张FCE口语考试模拟 Part 3 的图片，现在模拟英语口语考试：\n"
                         "1. 你我作为考生需依次讨论以下问题并最终达成一致\n"
                         "2. 所有问题讨论完毕后自动结束对话输出'结束对话'。你先开始"

                 }
            ],
        }
    ]

    round_num = 1
    agreement_reached = False  # 一致标志
    prompt_degree = f"根据FCE {degree} 考试的评分标准将以上讨论中‘用户’的回答进行评分，满分为15分"

    while not agreement_reached:  # 循环直至达成一致
        print(f"\n第{round_num}轮讨论")

        # 模型生成回答
        completion = client.chat.completions.create(
            model="qwen3-vl-plus",
            messages=messages
        )
        model_response = completion.choices[0].message.content
        print(f"模型讨论：{model_response}")
        messages.append({"role": "assistant", "content": model_response})

        # 检测模型是否同意
        model_agreed = "结束对话" in model_response
        if model_agreed:

            messages.append({"role": "user", "content": prompt_degree})
            # 模型生成回答
            completion = client.chat.completions.create(
                model="qwen3-vl-plus",
                messages=messages
            )
            model_response = completion.choices[0].message.content
            print(f"模型评分：{model_response}")
            break

        # 用户输入回答
        user_answer = input("请输入您的答案: ")
        messages.append({"role": "user", "content": user_answer})

        # 检测用户是否同意
        user_agreed = "结束对话" in user_answer

        # 判断是否双方达成一致
        if model_agreed or user_agreed:
            print("\n双方达成一致！结束讨论。")
            messages.append({"role": "user", "content": prompt_degree})
            # 模型生成回答
            completion = client.chat.completions.create(
                model="qwen3-vl-plus",
                messages=messages
            )
            model_response = completion.choices[0].message.content
            print(f"模型评分：{model_response}")
            agreement_reached = True
        else:
            messages.append({"role": "user", "content": "轮到你了"})
            round_num += 1


    return 0

def fce_qa_workflow_part4_by_pic( mage_url: str,degree: str):
    """ 第三轮问答：循环对话直至双方达成一致 """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": mage_url}
                },
                {"type": "text",
                 # "text": "图中是两张FCE口语考试模拟 Part 3 的图片，现在模拟英语口语考试，你来当一个考生，我是另一个考生，咱们两个一起讨论这幅图片并最终达成一致。每次回答后若同意对方观点请明确说出'同意'。你先开始"
                 "text": "图中是FCE口语考试Part4 的问题，现在模拟FCE英语口语考试：\n"
                         "1. 你我作为考官需随机问我两个问题，等我回答完一个问题后再问下一个问题。\n"
                         "2. 等我回答完两个问题结束对话并输出'结束对话'"

                 }
            ],
        }
    ]

    round_num = 1
    agreement_reached = False  # 一致标志
    prompt_degree = f"根据FCE {degree} 考试的评分标准将以上讨论中‘用户’的回答进行评分，满分为15分"

    while not agreement_reached:  # 循环直至达成一致
        print(f"\n第{round_num}轮讨论")

        # 模型生成回答
        completion = client.chat.completions.create(
            model="qwen3-vl-plus",
            messages=messages
        )
        model_response = completion.choices[0].message.content
        print(f"模型讨论：{model_response}")
        messages.append({"role": "assistant", "content": model_response})

        # 检测模型是否同意
        model_agreed = "结束对话" in model_response
        if model_agreed:

            messages.append({"role": "user", "content": prompt_degree})
            # 模型生成回答
            completion = client.chat.completions.create(
                model="qwen3-vl-plus",
                messages=messages
            )
            model_response = completion.choices[0].message.content
            print(f"模型评分：{model_response}")
            break

        # 用户输入回答
        user_answer = input("请输入您的答案: ")
        messages.append({"role": "user", "content": user_answer})

        # 检测用户是否同意
        user_agreed = "结束对话" in user_answer

        # 判断是否双方达成一致
        if model_agreed or user_agreed:
            print("\n双方达成一致！结束讨论。")
            messages.append({"role": "user", "content": prompt_degree})
            # 模型生成回答
            completion = client.chat.completions.create(
                model="qwen3-vl-plus",
                messages=messages
            )
            model_response = completion.choices[0].message.content
            print(f"模型评分：{model_response}")
            agreement_reached = True
        else:
            messages.append({"role": "user", "content": "轮到你了"})
            round_num += 1


    return 0

if __name__ == '__main__':
    # fce_qa_workflow_part1()
    # fce_qa_workflow_part2()
    # fce_qa_workflow_part3()

    # fce_qa_workflow_part3_by_pic("https://www.qplay2.com/images/part1.jpeg","part3")
    fce_qa_workflow_part4_by_pic("https://www.qplay2.com/images/part4.jpg","part4")
