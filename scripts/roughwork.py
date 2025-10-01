# %%
import os
os.chdir(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
os.getcwd()
# %%
old_path = "/Users/alison/Documents/drought-indicators/analysis/data/temp_backup_19092025/los/ff/monthly_los_level0_melted.csv"
new_path = "/Users/alison/Documents/drought-indicators/analysis/data/temp/los/ff/monthly_los_level0_melted.csv"
# %%
import pandas as pd

df_old = pd.read_csv(old_path, index_col=0)
df_new = pd.read_csv(new_path, index_col=0)

df_old.equals(df_new)
# %%
