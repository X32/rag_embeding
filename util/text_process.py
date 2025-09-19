import pandas as pd
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# 确保下载所需的 NLTK 数据（通常只需一次）
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)
# 初始化词形还原器
lemmatizer = WordNetLemmatizer()


# --- 自定义规则库 CustomRuleLib ---
class CustomRuleLib:
    """
    自定义文本处理规则库，包含去重、去特殊字符、文本标准化等方法。
    所有方法都设计为处理 Pandas Series 或单个字符串。
    """

    # 【去特殊字符规则】
    @staticmethod
    def remove_special_chars(text_series, method='regex', keep_chars=''):
        """
        去除文本中的特殊字符（非字母数字字符）。

        参数:
        text_series: pd.Series 或单个字符串。
        method: 清洗方法，可选 'regex'（默认，灵活） | 'translate'（高效） | 'list_comprehension'（简单）[2,4,9](@ref)。
        keep_chars: 一个字符串，指定即使不是字母数字也需要保留的字符（如空格、特定标点）。

        返回:
        清洗后的 pd.Series 或字符串。
        """
        if isinstance(text_series, str):
            input_is_series = False
            series = pd.Series([text_series])
        else:
            input_is_series = True
            series = text_series.astype(str).copy()  # 确保为字符串类型

        def _clean_string(s):
            if method == 'regex':
                # 构建模式：保留字母、数字、空格以及 keep_chars 中指定的字符
                pattern = f"[^A-Za-z0-9\\s{re.escape(keep_chars)}]"
                return re.sub(pattern, '', s)
            elif method == 'translate':
                # 创建映射表，移除非保留字符
                all_chars = s
                chars_to_remove = ''.join(set(all_chars) - set(keep_chars) - set(
                    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 '))
                trans_table = str.maketrans('', '', chars_to_remove)
                return s.translate(trans_table)
            elif method == 'list_comprehension':
                # 列表推导式，保留字母数字、空格和 keep_chars
                return ''.join(char for char in s if char.isalnum() or char.isspace() or char in keep_chars)
            else:
                return s

        cleaned_series = series.apply(_clean_string)
        return cleaned_series if input_is_series else cleaned_series.iloc[0]

    # 【文本标准化规则】
    @staticmethod
    def standardize_text(text_series, to_lower=True, remove_digits=False, lemmatize=False, remove_extra_spaces=True):
        """
        对文本进行标准化处理。

        参数:
        text_series: pd.Series 或单个字符串。
        to_lower: 是否转换为小写（默认 True）。
        remove_digits: 是否移除数字（默认 False）[12](@ref)。
        lemmatize: 是否进行词形还原（默认 False，需安装 nltk）[1,12](@ref)。
        remove_extra_spaces: 是否移除多余空格（默认 True）。

        返回:
        标准化后的 pd.Series 或字符串。
        """
        if isinstance(text_series, str):
            input_is_series = False
            series = pd.Series([text_series])
        else:
            input_is_series = True
            series = text_series.astype(str).copy()

        def _standardize_string(s):
            if to_lower:
                s = s.lower()
            if remove_digits:
                s = re.sub(r'\d+', '', s)  # 移除所有数字[12](@ref)
            if remove_extra_spaces:
                s = re.sub(r'\s+', ' ', s).strip()  # 将多个空格替换为单个并去除首尾空格[12](@ref)
            if lemmatize:
                # 分词 -> 词形还原 -> 重新组合
                words = word_tokenize(s)
                lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
                s = ' '.join(lemmatized_words)
            return s

        standardized_series = series.apply(_standardize_string)
        return standardized_series if input_is_series else standardized_series.iloc[0]

    # 【去重规则 - 基于列】
    @staticmethod
    def remove_duplicates(df, subset=None, keep='first', inplace=False):
        """
        基于指定列删除 DataFrame 中的重复行[6,7,8](@ref)。

        参数:
        df: 输入的 Pandas DataFrame。
        subset: 考虑用于识别重复项的列标签或序列，默认为 None（所有列）。
        keep: 确定保留哪些重复项，'first'（默认，保留第一次出现的）| 'last'（保留最后一次出现的）| False（删除所有重复项）。
        inplace: 是否直接修改原 DataFrame（默认 False）。

        返回:
        去重后的 DataFrame（如果 inplace=False）。
        """
        if not inplace:
            return df.drop_duplicates(subset=subset, keep=keep)
        else:
            df.drop_duplicates(subset=subset, keep=keep, inplace=True)

    # 【去重规则 - 基于文本内容相似性（简单示例）】
    @staticmethod
    def remove_similar_text_duplicates(df, text_column, threshold=0.8, keep='first'):
        """
        一个简单的基于文本相似度的去重示例（使用 Jaccard 相似度）。
        这是一个计算量较大的操作，适用于小数据集或重要任务。
        注意：这是一个示例，生产环境可能需要更高效的算法（如 MinHash, LSH）。

        参数:
        df: 输入的 Pandas DataFrame。
        text_column: 要比较相似度的文本列名。
        threshold: 相似度阈值，大于此值视为重复（默认 0.8）。
        keep: 保留哪些重复项，'first' | 'last'。

        返回:
        去重后的 DataFrame。
        """
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        # 这是一个更高效的示例（使用余弦相似度与词袋）
        vectorizer = CountVectorizer().fit_transform(df[text_column])
        cosine_sim = cosine_similarity(vectorizer)
        np.fill_diagonal(cosine_sim, 0)  # 将对角线设为0（自身比较）
        duplicates_set = set()

        for i in range(len(cosine_sim)):
            if i not in duplicates_set:
                # 找到所有与第i行相似度高于阈值的行
                similar_indices = np.where(cosine_sim[i] > threshold)[0]
                duplicates_set.update(similar_indices)

        # 根据 keep 参数决定保留哪些行
        indices_to_drop = list(duplicates_set)
        if keep == 'last':
            # 这里逻辑需简化，实际应用可能需要更复杂的策略
            df_cleaned = df.drop(index=indices_to_drop)
        else:  # 默认保留 first（最早出现的）
            df_cleaned = df.drop(index=indices_to_drop)

        return df_cleaned

def document_processing_engine(df, text_column, processing_steps):
    """
    文档处理引擎，根据 processing_steps 中的配置依次应用清洗规则。

    参数:
    df: 包含文本数据的 Pandas DataFrame。
    text_column: 要处理的文本列的名称。
    processing_steps: 一个列表，包含每个处理步骤的字典配置。
                      例如: [{'step': 'remove_special_chars', 'args': {'method': 'regex', 'keep_chars': '.,!?'}}, ...]

    返回:
    处理后的 Pandas DataFrame。
    """
    df_processed = df.copy()
    rule_lib = CustomRuleLib()

    for step_config in processing_steps:
        step_name = step_config['step']
        args = step_config.get('args', {})

        if step_name == 'remove_special_chars':
            df_processed[text_column] = rule_lib.remove_special_chars(df_processed[text_column], **args)
        elif step_name == 'standardize_text':
            df_processed[text_column] = rule_lib.standardize_text(df_processed[text_column], **args)
        elif step_name == 'remove_duplicates':
            subset_cols = args.get('subset', None)
            if subset_cols is None:
                subset_cols = [text_column] # 默认基于文本列去重
            rule_lib.remove_duplicates(df_processed, subset=subset_cols, keep=args.get('keep', 'first'), inplace=True)
        elif step_name == 'remove_similar_duplicates':
            df_processed = rule_lib.remove_similar_text_duplicates(df_processed, text_column, threshold=args.get('threshold', 0.8), keep=args.get('keep', 'first'))
        else:
            print(f"警告: 未知处理步骤 '{step_name}'，已跳过。")

    return df_processed

if __name__ == '__main__':
    # 1. 创建示例数据
    data = {
        'doc_id': [1, 2, 3, 4, 5, 6],
        'raw_text': [
            "Hello, World! This is a test. @2023 #Python",
            "Hello, World! This is a test. @2023 #Python",  # 完全重复
            "The quick brown foxes jumped over the lazy dogs.",
            "The quick brown fox jumps over the lazy dog.",  # 近似重复
            "   User    input    with    EXTRA   spaces...   and digits 123.   ",
            "Another document with %special& characters and UPPERCASE."
        ]
    }
    df_docs = pd.DataFrame(data)

    # 2. 定义处理流程
    processing_pipeline = [
        # 第一步：去特殊字符，保留基本标点
        {'step': 'remove_special_chars', 'args': {'method': 'regex', 'keep_chars': '.,!? '}},
        # 第二步：文本标准化（转小写、去多余空格、词形还原）
        {'step': 'standardize_text',
         'args': {'to_lower': True, 'remove_digits': False, 'lemmatize': True, 'remove_extra_spaces': True}},
        # 第三步：基于文本内容去重（保留第一次出现的）
        {'step': 'remove_duplicates', 'args': {'subset': ['raw_text'], 'keep': 'first'}}
        # 注意：相似去重 'remove_similar_duplicates' 计算量大，可根据需要选择使用
    ]

    # 3. 运行处理引擎
    df_cleaned = document_processing_engine(df_docs, 'raw_text', processing_pipeline)

    # 4. 查看结果
    print("原始数据形状:", df_docs.shape)
    print("处理后数据形状:", df_cleaned.shape)
    print("\n处理后数据:")
    print(df_cleaned[['doc_id', 'raw_text']].to_string(index=False))