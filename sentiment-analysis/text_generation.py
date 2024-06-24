from transformers import pipeline

def get_text_generation(model_name, tokenizer_name, string_arr):
    
    # 封裝好開箱即用：pipelines
    # text-generation （文字生成）
    #   max_length：句子最長單詞
    #   tokenizer
    #   temperature
    #   num_return_sequences
    text_generation = pipeline(task="text-generation", 
                                  model=model_name,
                                  tokenizer=tokenizer_name,
                                  max_length=100,
                                  truncation=True,
                                  eos_token_id=0,
                                  pad_token_id=0
                                  )
    # Get sentiments
    results = text_generation(string_arr)
    return results


if __name__ == "__main__":
    model_name = "achrekarom/text_generation"
    string = "The dog"

    generation_string = get_text_generation(model_name, model_name, string)
    print(generation_string[0]['generated_text'])