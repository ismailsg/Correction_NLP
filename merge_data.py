# import pandas as pd


# test_data=pd.read_csv("jfleg_test_data.csv",sep="|")

# validation_data=pd.read_csv("jfleg_validation_data.csv",sep='"')

# merge_data=pd.concat([test_data,validation_data],ignore_index=True)


# merge_data.drop_duplicates()

# df_deduplicated = merge_data.drop_duplicates()


# print(df_deduplicated.shape)

from datasets import load_dataset

import pandas as pd

# If the dataset is gated/private, make sure you have run huggingface-cli login
# train_dataset = load_dataset("jfleg", split='validation[:]')

# eval_dataset = load_dataset("jfleg", split='test[:]')

# print(train_dataset.shape)
# print(eval_dataset.shape)

# train_dataset.to_csv("train_dataset.csv",index=False)

# eval_dataset.to_csv('eval_dataset.csv',index=False)
train_dataset=pd.read_csv('train_dataset.csv')
eval_dataset=pd.read_csv('eval_dataset.csv')

merge_dataset=pd.concat([train_dataset,eval_dataset])

merge_dataset=merge_dataset.drop_duplicates()

print(merge_dataset.shape)

# import requests

# url = "https://datasets-server.huggingface.co/first-rows?dataset=jfleg&config=default&split=validation"

# response = requests.get(url)

# if response.status_code == 200:
#     print("Request successful")
#     print("Response content:")
#     print(response.text)
# else:
#     print(f"Request failed with status code: {response.status_code}")
