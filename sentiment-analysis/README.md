# 練習 Python 專案

## 📝 套件
- transformers
    ```
    簡介
        主流套件
        由 Hugging Face 開發的一個 NLP 套件
        支援載入目前絕大部分的預訓練模型

    兩個著名 Transformer 模型：
        - GPT
        - BERT

    功能
        - 文字分類
            - 情感分析
            - 句子對關係判斷
        - 對文字中的詞語進行分類
            - 詞性標註 (POS)
            - 命名實體識別 (NER)
        - 文字生成
            - 填充預設的樣板 (prompt)
            - 預測文字中被遮掩掉 (masked) 的詞語
        - 文字中抽取答案：根據給定的問題從中抽取出對應的答案
        - 根據輸入文字生成新的句子
            - 文字翻譯
            - 自動摘要


    封裝好開箱即用：pipelines
        簡介
            封裝了預訓練模型和對應的前處理和後處理環節
            我們只需輸入文字，就能得到預期的答案
        常用
            - feature-extraction （獲得文字的向量化表示）
            - fill-mask （填充被遮蓋的詞、片段）
            - ner（命名實體識別）
            - sentiment-analysis （情感分析）
            - summarization （自動摘要）
            - text-generation （文字生成）
            - translation （機器翻譯）
            - zero-shot-classification （零訓練樣本分類）
    ```
- Pytorch
    ```
    簡介
        由 Facebook 人工智慧研究院於 2017 年推出
        成為主流的深度學習框架
        除去框架本身的優勢，還有著良好的生態圈

    用途
        通過 Pytorch 的 DataLoader 類來載入資料
        使用 Pytorch 的最佳化器對模型參數進行調整
    ```
- torchvision
    ```
    簡介
        來呼叫預訓練模型，載入資料集
    ```