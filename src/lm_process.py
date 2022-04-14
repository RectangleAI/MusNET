import pandas as pd
data = pd.read_excel('./data/guitar.xlsx')
print(data.head())
data.dropna(axis = 0, how = 'all', inplace = True)
data['S/N'] = data['S/N'].astype(int)
data = data.set_index('S/N')

data.columns = data.columns.astype(str).str.replace(' ', '')
print(data.columns)
for c in data.columns:
    data[c] = data[c].astype(str).str.replace(' ','')
print(data.head())
print(data.shape)

data.to_csv("./data/guitar.txt", header = False, sep = ",", index = False)

with open("./data/guitar.txt",encoding='utf-8') as f:
    text = f.read()
    text = text.replace(',nan', '')
print(text)
with open("./data/guitarp.txt",'w',encoding='utf-8') as f:
    f.write(text)