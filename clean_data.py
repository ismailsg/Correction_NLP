import pandas as pd

# Load the dataset
data = pd.read_csv('data/unigram_freq.csv')  # Replace 'your_dataset.csv' with your actual file path

# Extract the 'word' column
word_column = data['word']

# Print the extracted column
print(word_column)

print(word_column.shape)
word_column.to_csv('words.csv', index=False)  # Replace 'word_column.csv' with your desired file name/path
