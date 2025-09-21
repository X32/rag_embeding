import os
import dashscope
from dashscope import Generation  # 添加DashScope Generation导入
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from util.embeddings import get_text_embedding
from util.llm_models import function_call_predict

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
    for index, row in enumerate(x.itertuples()):
        try:
            print(row.Conversation)
            # 尝试执行 function_call_predict 函数
            x.at[index, "function_call_prediction_turbo"] = function_predict(row.Conversation,model='qwen-turbo')
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
def test():
    init_function_calling()
    text = '我的储蓄账户是否可以与支付软件直接绑定？'
    function_name = function_predict(text, model='qwen-turbo')
    print(function_name)
    return
def main():
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
    init_function_calling()

    print(train_df.head())
    print(train_df.shape)
    train_df = llm_predict_handle(train_df)
    print("H++++++++++++++++++++++H")
    print(train_df.head())
    print(train_df.shape)

    print("final")



if __name__ == '__main__':
    main()
    # test()
