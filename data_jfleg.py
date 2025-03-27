import pandas as pd
import datasets

# Load the JFLEG dataset validation split
validation_split = datasets.load_dataset("jfleg", split="validation")

# Extract data for CSV
data = []
for example in validation_split:
    sentence = example["sentence"]
    corrections = example["corrections"]
    data.append({"sentence": sentence, "corrections": corrections})

# Create a DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV
output_csv_file = "jfleg_validation_data.csv"
df.to_csv(output_csv_file, index=False)
