from transformers import pipeline

def get_sentiments_with_pipeline(model_name, tokenizer_name, string_arr):
    
    # 封裝好開箱即用：pipelines
    # sentiment-analysis （情感分析）
    #   device
    #       指定運行 pipeline 裝置 (-1 CPU / 非負整數表示GPU，如果有可用的 GPU 可來加速計算)
    #   return_all_scores
    #       文字進行情感分析時，pipeline會回傳每個可能情感類別的得分，而不僅僅是最可能的情感類別及其得分
    #       具體的輸出將取決於使用的模型和設定，但一般情況下，對於情感分析任務，輸出將是一個清單，清單中的每個元素都是一個字典，表示一個情感類別及其對應的得分
    sentiment_analyzer = pipeline(task="sentiment-analysis", 
                                  model=model_name,
                                  tokenizer=tokenizer_name,
                                  top_k=None,
                                  return_all_scores=True
                                  )
    # Get sentiments
    results = sentiment_analyzer(string_arr)
    return results


if __name__ == "__main__":
    model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

    string_arr = [
        "我會披星戴月的想你，我會奮不顧身的前進，遠方煙火越來越唏噓，凝視前方身後的距離",
        "鯊魚寶寶 doo doo doo doo doo doo, 鯊魚寶寶"
    ]

    predictions = get_sentiments_with_pipeline(
        model_name, model_name, string_arr)
    print(predictions)