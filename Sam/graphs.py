import pandas as pd
# df = pd.read_csv('./Experiments/csvvv')
df = pd.read_csv('./Experiments/rf')
df2 = pd.DataFrame()
# for i in range(9):
#     f = df.iloc[[i, i+9, i+18,i+27,i+36,i+45,i+54]].sum(axis = 0)
#     print(f)
#     df2 = df2.append(f, ignore_index=True)
i=0
while i < 63:
    f = df.iloc[[i,i+1,i+2,i+3,i+4,i+5, i+6,i+7,i+8 ]].sum(axis = 0)
    print(f)
    df2 = df2.append(f, ignore_index=True)
    i+=9

perturbed_df = pd.DataFrame()
i = 0
for x in df2.columns:
    tempColumn = df2.loc[:, x].apply(lambda item: (i,round(item / 9,2)))
    perturbed_df[x] = tempColumn
    i+=1
# f =df.sum(axis = 0)

print(df2)
print(perturbed_df)