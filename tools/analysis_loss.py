import re
import pandas as pd

# Load and read the file content
modelname = 'mcta'
file_path = 'results_voc/mcta/log_dir/train-20250105-1639-VOC12.log'
with open(file_path, 'r') as file:
    log_data = file.readlines()

# Adjusting the code to use regular expressions to find and extract "epoch", "train_mct_loss", and "train_pat_loss" values

# Re-initialize lists to store data
epochs = []
train_mct_loss = []
train_pat_loss = []

# Define regex patterns for each value
epoch_pattern = re.compile(r'"epoch":\s*(\d+)')
mct_loss_pattern = re.compile(r'"train_cls_loss":\s*([\d.]+)')
pat_loss_pattern = re.compile(r'"train_pat_loss":\s*([\d.]+)')

# Process each line and extract values using regex
for line in log_data:
    # Search for each pattern in the line
    epoch_match = epoch_pattern.search(line)
    mct_loss_match = mct_loss_pattern.search(line)
    pat_loss_match = pat_loss_pattern.search(line)

    # Extract and store values if they exist in the line
    if epoch_match and mct_loss_match and pat_loss_match:
        epochs.append(int(epoch_match.group(1)) + 1)
        train_mct_loss.append(float(mct_loss_match.group(1)))
        train_pat_loss.append(float(pat_loss_match.group(1)))

# Create a DataFrame
df = pd.DataFrame({
    'epoch': epochs,
    'train_mct_loss': train_mct_loss,
    'train_pat_loss': train_pat_loss
})

print(df)
# Save to Excel
output_file_path = f'results_voc/{modelname}/{modelname}_training_loss.xlsx'
df.to_excel(output_file_path, index=False)