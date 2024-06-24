from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 注意：使用預定義的分詞器和預訓練模型，基本上是配套
# 例：使用預訓練模型是 bert-base-chinese，載入分詞器也必須使用 bert-base-chinese
def get_sentiments(model_name, string_arr):

    # AutoTokenizer 載入預訓練的分詞器
    # Initialize tokenizer (根據指定分詞器名稱載入)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize input strings
    # padding           長度不足 max_length 時是否進行填充
    # truncation        長度超過 max_length 時是否進行截斷
    # max_length
    # return_tensors    指定回傳資料類型 (pt：pytorch 張量 / tf：TensorFlow 張量)
    inputs = tokenizer(string_arr, padding=True,
                       truncation=True, return_tensors="pt")

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Make predictions
    # **inputs 是一個 dict
    outputs = model(**inputs)

    # Softmax to convert logits to probabilities
    #
    # Logits
    #   通常指神經網路的最後一層(輸出層)輸出，是原始預測值；通常代表每個類別的得分或機率
    # 
    # Softmax
    #   機器學習中常見的激活函數，尤其是處理分類問題
    #   作用：將 Logits（邏輯值）輸入 softmax 函數
    #   目的：將原始的未經處理的分數轉換為機率分佈 (轉換成一組介於 0-1 之間的值，這些值加起來會 1)
    #   提供：一個相對容易理解的機率分佈
    
    # dim：維度 (0,1,2,-1)
    # dim=-1 和 dim=2 的結果是一樣的 (同一陣列加總為 1)
    # https://blog.csdn.net/Will_Ye/article/details/104994504
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    return predictions


if __name__ == "__main__":
    
    # 模型：情感分析
    # https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student
    # huggingface 會將 model 儲存於此 ~/.cache/huggingface/hub/
    # 
    # 看命名 student：此模型為「知識蒸餾」(Knowledge Distillation)
    #   模型壓縮技術
    #   student 模型從可以更複雜的 teacher 模型中 "學習"
    #   如果已經透過複雜的結建立構出不錯的模型，可以用知識蒸餾訓練出較簡易版本的模型，準確度不會差太多
    model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

    string_arr = [
        "我會披星戴月的想你，我會奮不顧身的前進，遠方煙火越來越唏噓，凝視前方身後的距離",
        "鯊魚寶寶 doo doo doo doo doo doo, 鯊魚寶寶"
    ]

    predictions = get_sentiments(model_name, string_arr)
    print(predictions)
