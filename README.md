# BERT_classification
How to use - >

    csv_ = "C:/Users/akash/Desktop/archive/IMDB Dataset.csv"

    css = textclassification(
        datatype="csv",
        column_name=("review", "sentiment"),
        no_of_labels=2,
        is_trainable=False,
        model_path=None,
        data_path=csv_,
        max_len=512,
        tokenizer_config={}
    )
    css.train()
