import os
import dashscope
from dashscope import Generation  # 添加DashScope Generation导入
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from util.embeddings import get_text_embedding
from util.llm_models import function_call_predict

import time
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
functions = []
type_dict = {
        "handle_savings_account_management": 1,
        "handle_loan_services": 2,
        "handle_credit_card_services": 3,
        "handle_investment_advisory": 4,
        "handle_international_transactions": 5
    }
def handle_savings_account_management():
    res = "用户需要执行储蓄账户开设与管理相关业务"
    return res
def handle_loan_services():
    res = "用户需要执行贷款服务相关业务"
    return res
def handle_credit_card_services():
    res = "用户需要执行信用卡服务相关业务"
    return res
def handle_investment_advisory():
    res = "用户需要执行投资与理财咨询业务"
    return res

def handle_international_transactions():
    res = "用户需要执行国际业务与汇款相关业务"
    return res


def function_predict(text, model='qwen-turbo'):
    messages = [
        {"role": "system", "content": "你是一个智能银行客户接待应用，输入的每个user message都是某位银行客户的需求。\
        你的每一次回答都必须调用function call来完成。请仔细甄别用户需求，并合理调用外部函数来进行回答。"},
        {"role": "user", "content": text}]


    funcName = function_call_predict(messages,functions, model)
    if funcName != -1:
        res = type_dict[funcName]
        return res
    else:
        return -1

def llm_predict_handle(x: pd.DataFrame):
    return llm_predict_handle_withName(x,"function_call_prediction_turbo")
def llm_predict_handle_withName(x: pd.DataFrame,key:str):
    for index, row in enumerate(x.itertuples()):
        try:
            #print(row.Conversation)
            # 尝试执行 function_call_predict 函数
            text = row.Conversation
            res = function_predict(text,model='qwen-turbo')
            print(f"type = {row.type}  ret = {res}")
            x.at[index, key] = res
        except Exception as e:
            # 打印错误信息并等待一分钟
            print(f"Error on row {index}: {e}")
            # time.sleep(60)  # 等待一分钟
            continue  # 继续下一次循环

        # 每10行打印一次进度
        if index % 10 == 0:
            print(f"Processed {index}/{len(x)} rows")
    return x
#初始化函数
def init_function_calling():
    global functions
    handle_savings_account_management_description = "这是一个专门用于执行储蓄账户开设与管理相关业务的函数，\
        储蓄账户开设与管理业务涉及到储蓄账户的创建和维护。客户可以在银行开设新的储蓄账户，这通常需要提供个人身份证明、地址证明以及可能的初始存款。银行还提供更新账户信息的服务，如更改联系信息、更改账户类型等。此外，客户还可以查询自己账户的余额、交易记录和其他账户活动。这类服务还可能包括网上银行和移动银行服务的设置和支持，以方便客户远程管理其账户。"
    handle_savings_account_management_function = {
        "name": "handle_savings_account_management",
        "description": handle_savings_account_management_description,
        "parameters": {}
    }

    handle_loan_services_description = "这是一个专门用于执行贷款服务相关业务的函数，\
        贷款服务包括各种类型的贷款申请和咨询服务，如住房贷款、汽车贷款、个人贷款等。银行提供详细的贷款产品信息，包括贷款金额、利率、还款期限和还款方式等。银行还会根据客户的信用评分和财务状况审核贷款申请。对于不同类型的贷款，如住房贷款或汽车贷款，银行可能需要相应的资产作为抵押。此外，银行还提供贷款计算器和专业顾问来帮助客户计划其财务。"
    handle_loan_services_function = {
        "name": "handle_loan_services",
        "description": handle_loan_services_description,
        "parameters": {}
    }

    handle_credit_card_services_description = "这是一个专门用于执行信用卡服务相关业务的函数，\
        信用卡服务涉及信用卡的申请、激活、挂失、信用额度管理和账单查询等服务。客户可以根据自己的需要选择不同类型的信用卡，如奖励卡、积分卡或商务卡等。银行提供在线服务来激活新卡、报告丢失或被盗的卡，并及时发行新卡。客户还可以调整信用额度，查询每月的账单和消费记录。此外，信用卡服务还包括各种优惠和奖励计划，如旅行奖励、现金返还等。"
    handle_credit_card_services_function = {
        "name": "handle_credit_card_services",
        "description": handle_credit_card_services_description,
        "parameters": {}
    }

    handle_investment_advisory_description = "这是一个专门用于执行投资与理财咨询服务的函数，\
        投资与理财咨询服务指的是提供关于股票、债券、基金和其他投资产品的咨询服务。银行通常会提供个性化理财规划，帮助客户根据自己的风险承受能力、投资目标和时间框架制定投资策略。此外，银行还提供退休规划服务，帮助客户规划其退休金账户和储蓄。投资顾问可帮助客户了解市场动态、资产配置以及潜在的投资机会。"
    handle_investment_advisory_function = {
        "name": "handle_investment_advisory",
        "description": handle_investment_advisory_description,
        "parameters": {}
    }

    handle_international_transactions_description = "这是一个专门用于执行国际业务与汇款服务的函数，\
        国际业务与汇款服务涵盖了与国际金融交易相关的服务，包括外汇兑换、国际汇款和外币账户管理。客户可以通过银行进行跨国货币转换和汇款，银行提供即时的汇率信息和汇款指导。对于需要频繁进行国际交易的客户，银行提供外币账户服务，允许存储和管理多种货币。此外，银行还提供企业级的国际贸易融资和汇款服务，支持企业在全球范围内的业务扩展。"
    handle_international_transactions_function = {
        "name": "handle_international_transactions",
        "description": handle_international_transactions_description,
        "parameters": {}
    }
    functions = [handle_savings_account_management_function,
                 handle_loan_services_function,
                 handle_credit_card_services_function,
                 handle_investment_advisory_function,
                 handle_international_transactions_function]

    available_functions = {
        "handle_savings_account_management": handle_savings_account_management,
        "handle_loan_services": handle_loan_services,
        "handle_credit_card_services": handle_credit_card_services,
        "handle_investment_advisory": handle_investment_advisory,
        "handle_international_transactions": handle_international_transactions
    }

def process_dataset(dataset,
                    messages=None,
                    model_name='qwen-max',
                    text_col_name='Conversation',
                    prediction_col_name='function_call_prediction_qwenMax'):
    """
    对给定的数据集应用function_call_predict进行意图识别。

    :param dataset: DataFrame, 包含需要处理的数据
    :param model_name: dict, Few-shot-messages，默认为None，表示不带入任何系统消息和提示
    :param model_name: str, 使用的模型名称
    :param text_col_name: str, 输入function calling的文本列名称
    :param prediction_col_name: str, 输出意图判别结果的列名称
    :return: 修改后的DataFrame
    """
    if messages == None:
        input_messages = []
    else:
        input_messages = messages.copy()
    data_len = len(dataset)
    for index in range(data_len):
        success = False
        while not success:
            try:
                # 尝试执行 function_call_predict 函数
                text = dataset.at[index, text_col_name]
                input_messages.append({"role": "user", "content": text})

                funcName = function_call_predict(input_messages, functions, model=model_name)
                # funcName = function_call_predict(messages, functions, model)
                result = -1
                if funcName != -1:
                    result = type_dict[funcName]
                dataset.at[index, prediction_col_name] = result
                success = True  # 如果执行成功，跳出循环
                input_messages = messages.copy()
            except Exception as e:
                # 打印错误信息并等待一分钟
                print(f"Error on row {index}: {e}")
                time.sleep(60)  # 等待一分钟后再次尝试

        # 每10行打印一次进度
        if index % 10 == 0:
            print(f"Processed {index}/{len(dataset)} rows")

    return dataset

def test():
    init_function_calling()
    text = '请问贵行的理财产品有哪些安全保障措施？'
    function_name = function_predict(text, model='qwen-turbo')
    print(function_name)
    return
def few_shot_prompting():
    init_function_calling()
    # function_name = function_predict(text, model='qwen-turbo')
    # return function_name
def get_few_shot_test():
    import pandas as pd

    # 读取训练集
    train_df = pd.read_csv('./data/train_dataset_test.csv')

    # 读取测试集
    test_df = pd.read_csv('./data/test_dataset_test.csv')
    init_function_calling()
    print("H++++++++++++++++++++++H")
    # #
    # #
    test_df = llm_predict_handle(test_df)
    test_temp_1 = test_df[test_df["function_call_prediction_turbo"] != test_df["type"]][['Conversation', 'type', 'function_call_prediction_turbo']]
    print(test_temp_1.head())
    print(test_temp_1.shape)
    test_temp_1.to_csv('./data/few_shot_dataset_test.csv', index=False)

    train_df = llm_predict_handle(train_df)
    test_temp_2 = train_df[train_df["function_call_prediction_turbo"] != train_df["type"]][['Conversation', 'type', 'function_call_prediction_turbo']]
    print(test_temp_2.head())
    print(test_temp_2.shape)
    test_temp_2.to_csv('./data/few_shot_dataset_train.csv', index=False)

    # # 读取训练集
    # test_temp_1 = pd.read_csv('./data/few_shot_dataset_train.csv')
    # # 读取测试集
    # test_temp_2 = pd.read_csv('./data/few_shot_dataset_test.csv')
    #
    # print(test_temp_1.shape)
    # print(test_temp_2.shape)

    merged_df = pd.concat([test_temp_1, test_temp_2], axis=0)
    merged_df.to_csv('./data/few_shot_dataset.csv', index=False)
    print(merged_df)
    print(merged_df.shape)

def get_key_by_value(dict, value):
    for key, val in dict.items():
        if val == value:
            return key
    return None

def get_few_shot_prompting():
    test_temp = pd.read_csv('./data/few_shot_dataset.csv')
    few_shot_messages = []
    i = 0
    for index, row in test_temp.iterrows():
        text = row['Conversation']
        intention_category = row['type']
        function_name = get_key_by_value(type_dict, intention_category)
        i = i + 1

        assistant_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "function": {  # 改为function对象
                    "name": function_name,  # 保持原变量名
                    "arguments": '{}'  # 改为parameters且为字典格式
                },
                "id": f"call_abc{i}"  # 保持原格式
            }]
        }

        tool_message = {
            "role": "tool",
            "content": '...',
            "tool_call_id": f"call_abc{i}"
        }

        few_shot_messages.append({"role": "user", "content": text})
        few_shot_messages.append(assistant_message)
        few_shot_messages.append(tool_message)

    system_message = [{"role": "system", "content": "你是一个智能银行客户接待应用，输入的每个user message都是某位银行客户的需求。\
        你的每一次回答都必须调用function call来完成。请仔细甄别用户需求，并合理调用外部函数来进行回答。"}]
    messages = system_message + few_shot_messages
    # messages
    print(messages)
    return messages

def llm_few_shot_predict(test_df: pd.DataFrame):
    few_shot_messages = get_few_shot_prompting()

    test_df = process_dataset(dataset=test_df,  # test_df。head(5)
                              messages=few_shot_messages,
                              model_name='qwen-turbo',
                              prediction_col_name='function_call_prediction_turbo_few_shot')
    return test_df

def test_few_shot_predict():
    init_function_calling()
    test_df = pd.read_csv('./data/test_dataset_test.csv')
    test_df = llm_few_shot_predict(test_df)
    print(test_df.head(10))
    print(test_df.shape)
    test_temp_3 = test_df[test_df["function_call_prediction_turbo_few_shot"] != test_df["type"]][
        ['Conversation', 'type', 'function_call_prediction_turbo_few_shot']]
    print(test_temp_3)
    print(test_temp_3.shape)
#划分数据集
def split_dataset():
    # 数据准备，划分测试集和数据集
    combined_df = pd.read_csv('./data/combined_dataset.csv')
    combined_df["embedding"] = combined_df.Conversation.apply(lambda x: get_text_embedding(x))

    # 直接对整个DataFrame进行划分
    train_df, test_df = train_test_split(
        combined_df,
        test_size=0.2,  # 测试集占20%
        random_state=42,  # 固定随机种子，确保结果可复现
        stratify=combined_df['type']  # 按目标变量分层抽样（分类任务）
    )
    train_df.to_csv('./data/train_dataset_test.csv', index=False)
    test_df.to_csv('./data/test_dataset_test.csv', index=False)
    return train_df, test_df
def collection_few_shot_dataset():
    train_df, test_df = split_dataset()

    # init_function_calling()
    print(train_df.shape)
    print(test_df.shape)

    print("H++++++++++++++++++++++H")
    #
    # 收集few_shot 数据集
    # 测试集
    test_df = llm_predict_handle(test_df)
    test_temp_1 = test_df[test_df["function_call_prediction_turbo"] != test_df["type"]][
        ['Conversation', 'type', 'function_call_prediction_turbo']]
    print(test_temp_1.head())
    print(test_temp_1.shape)
    test_temp_1.to_csv('./data/few_shot_dataset_test.csv', index=False)

    # 训练集
    train_df = llm_predict_handle(train_df)
    test_temp_2 = train_df[train_df["function_call_prediction_turbo"] != train_df["type"]][
        ['Conversation', 'type', 'function_call_prediction_turbo']]
    print(test_temp_2.head())
    print(test_temp_2.shape)
    test_temp_2.to_csv('./data/few_shot_dataset_train.csv', index=False)

#数据预测
def data_few_shot_predict(test_df: pd.DataFrame):
    init_function_calling()

    test_df = llm_few_shot_predict(test_df)
    print(test_df.head(10))
    print(test_df.shape)
    return test_df

def dataSet_llm_predict_handle():
    init_function_calling()
    test_df = pd.read_csv('./data/test_dataset_test.csv')
    ret_test_df = llm_predict_handle_withName(test_df,"function_call_prediction_turbo")
    ret_test_df.to_csv('./data/llm_predict_dataset_test.csv', index=False)


def main():
    few_shot_df = None

    #如果few_shot_dataset.csv不存在，才收集数据
    if not os.path.exists('./data/few_shot_dataset.csv'):
        #收集few_shot数据
        collection_few_shot_dataset()
    else:
        few_shot_df = pd.read_csv('./data/few_shot_dataset.csv')

    #如果few_shot_df不存在，才读取数据
    if few_shot_df is None:
        few_shot_df = pd.read_csv('./data/few_shot_dataset.csv')


    print(few_shot_df.shape)

    # 测试few_shot 预测
    # test_few_shot_predict()
    test_df =  pd.read_csv('./data/test_dataset_test.csv')
    ret_test_df = data_few_shot_predict(test_df)

    print(ret_test_df.head(10))
    print(ret_test_df.shape)

    #预测的结果
    ret_test_df1 = ret_test_df[ret_test_df["function_call_prediction_turbo_few_shot"] != ret_test_df["type"]][
        ['Conversation', 'type', 'function_call_prediction_turbo_few_shot']]
    print(ret_test_df1)
    print(ret_test_df1.shape)
    #保存few_shot 预测结果
    ret_test_df.to_csv('./data/test_dataset_with_llm_pred.csv', index=False)

    # #
    # ret_test_df = llm_predict_handle(ret_test_df)
    # ret_test_df.to_csv('./data/few_shot_dataset_test.csv', index=False)







if __name__ == '__main__':
    # test_few_shot_predict()
    main()
    # dataSet_llm_predict_handle()
    # test()
    # get_few_shot_test()
    # get_few_shot_prompting()
