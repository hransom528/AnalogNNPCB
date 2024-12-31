import pandas as pd
import os

to_merge = os.listdir()
to_merge.remove("MergeCSV.py")

print("Importing CSVs into dataframes...")
dfs = []
for filename in to_merge:
    # read the csv, making sure the first two columns are str
    df = pd.read_csv(filename, header="infer")
    dfs.append(df)

for i in range(len(dfs)):
    uniqueVals = dfs[i]["event"].unique()
    print(f"{to_merge[i]}: {uniqueVals}")

'''
# concatenate them vertically
print("Merging dataframes...")
merged = dfs[0]
for i in range(1, len(dfs)):
    merged = pd.concat([merged, dfs[i]], axis=0, ignore_index=True)

# write it out
print(merged.shape)
merged.to_csv("merged.csv", header=None, index=None)
'''