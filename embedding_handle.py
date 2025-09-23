from util.embeddings import get_text_embedding
from LLM_handling import split_dataset
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def main():
    train_df, test_df = split_dataset()

    #init_function_calling()
    print(train_df.shape)
    print(test_df.shape)
    train_embeddings = np.stack(train_df["embedding"].values)
    test_embeddings = np.stack(test_df["embedding"].values)

    # 训练集中计算训练集集与训练集之间的余弦相似度
    train_cos_sim_matrix = cosine_similarity(train_embeddings, train_embeddings)

    # 计算训练集中彼此最相似的三个文本
    train_most_similar_indices = np.argpartition(-(train_cos_sim_matrix - np.eye(train_cos_sim_matrix.shape[0])), 3,
                                                 axis=1)[:, :3]
    train_most_similar_indices_df = pd.DataFrame(train_most_similar_indices, columns=['sim_1', 'sim_2', 'sim_3'])

    print(train_most_similar_indices_df.head())

    train_df_final = pd.concat([train_df, train_most_similar_indices_df], axis=1)
    train_df_final.head()

    # 计算测试集与训练集之间的余弦相似度
    test_cos_sim_matrix = cosine_similarity(test_embeddings, train_embeddings)

    # 为测试集中的每个文本找到与训练集中最相似的三个文本
    test_most_similar_indices = np.argpartition(-test_cos_sim_matrix, 3, axis=1)[:, :3]
    test_most_similar_indices_df = pd.DataFrame(test_most_similar_indices, columns=['sim_1', 'sim_2', 'sim_3'])
    print(test_most_similar_indices_df.head())


    test_df_final = pd.concat([test_df.reset_index(drop=True), test_most_similar_indices_df.reset_index(drop=True)], axis=1)
    print(test_df_final.head())

    # test_df_final.to_csv('./data/embedding_test_dataset_pred.csv', index=False)
    # print('++=====================++ 1')
    #
    test_df_final['sim_1_target'] = test_df_final.sim_1.apply(lambda x: train_df_final.type[x])
    test_df_final['sim_2_target'] = test_df_final.sim_2.apply(lambda x: train_df_final.type[x])
    test_df_final['sim_3_target'] = test_df_final.sim_3.apply(lambda x: train_df_final.type[x])

    print('++=====================++ 2')
    print(test_df_final.head())
    count =  (test_df_final["sim_1_target"] != test_df_final["type"]).sum()
    print(count)
    test_df_final.to_csv('./data/embedding_test_dataset_pred.csv', index=False)



if __name__ == '__main__':
    main()