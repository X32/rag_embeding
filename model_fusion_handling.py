import pandas as pd
def main():
    test_df1 = pd.read_csv('./data/test_dataset_with_llm_pred.csv')
    test_df2 = pd.read_csv('./data/test_dataset_with_pred.csv')
    test_df3 = pd.read_csv('./data/llm_predict_dataset_test.csv')

    predictions_df =  test_df1[['Conversation','type','function_call_prediction_turbo_few_shot']]
    print(predictions_df.head())
    print(predictions_df.shape)

    predictions_df1 = test_df2[['RF','LGBM','XGB']]

    print(predictions_df1.head())
    print(predictions_df1.shape)

    predictions_df2 = test_df3[['function_call_prediction_turbo']]

    print(predictions_df2.head())
    print(predictions_df2.shape)


    merged_df = pd.concat([
        predictions_df.reset_index(drop=True),
        predictions_df1.reset_index(drop=True),
        predictions_df2.reset_index(drop=True)
    ], axis=1)
    print(merged_df.head())
    print(merged_df.shape)
    # merged_df.to_csv('./data/merged_dataset_pred.csv',index=False)
    cols = ['function_call_prediction_turbo_few_shot', 'function_call_prediction_turbo', 'RF', 'LGBM', 'XGB']
    # 使用apply方法按行计算众数
    mode_series = merged_df[cols].apply(lambda x: x.mode().iloc[0], axis=1)

    # 创建一个新列来存储每一行的众数
    merged_df['Mode_preds'] = mode_series
    print(merged_df.head())
    merged_df.to_csv('./data/merged_dataset_pred.csv', index=False)

    count = (merged_df["Mode_preds"] != merged_df["type"]).sum()
    print(count)


if __name__ == '__main__':
    main()
