from dashscope import TextEmbedding
import os
import dashscope

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

def get_text_embedding(text: str) -> list:
    """通用文本嵌入生成方法"""
    response = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v1,
        input=text
    )
    if response.status_code == 200:
        return response.output['embeddings'][0]['embedding']
    raise Exception(f"Embedding生成失败: {response.message}")