# Set name to the target protein name.
import pandas as pd
name = 'Aptamer'
def remove_duplicate_string_rows(data):
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    if df.shape[1] < 3:
        raise ValueError("The data must contain at least three columns")
    col2 = df.columns[1]
    col3 = df.columns[2]
    df[col2] = df[col2].astype(str)
    df[col3] = df[col3].astype(str)
    result_df = df.drop_duplicates(subset=[col2,col3], keep='first')
    return result_df

with open('best_sequence.txt','r')as F:
    seq = F.readlines()
data = []
for i in range(len(seq)):
    data.append([f'{name}{i+1}',seq[i].strip('\n').split(' ')[0],seq[i].strip('\n').split(' ')[1]])
result = remove_duplicate_string_rows(data)
print(f'Duplicate entries removed: {len(seq)-len(result)}')
with open("unique_best_sequence.txt",'w')as F:
    for i in range(len(seq)):
        try:
            F.writelines(f'{result[0][i]} {result[1][i]} {result[2][i]}\n')
        except:
            pass
