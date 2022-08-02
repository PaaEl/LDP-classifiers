import pandas as pd
# df = pd.read_csv('./Experiments/csvvv')
ldp_mechanism = ['de','olh', 'hr', 'he', 'oue', 'rap']
database_names=['adult','mushroom','iris','vote','car','nursery','spect','weightliftingexercises','htru']
epsilon_values=[0.01,0.1,0.5,1,2,3,5]
depth = [1,2,6]
size =[1,20,200]
metrics = ['time', 'balanced_accuracy', 'f1', 'precision', 'recall']
colour = ['blue','red', 'green', 'orange', 'cyan', 'violet', 'olive', 'yellow', 'teal']
leng=5
lis=[]
lis_lis=[]
search1= ['de','he']
search2= ['balanced_accuracy']
file ='./b.csv'
df = pd.read_csv(file, sep =';')


for x in search1:
    for y in search2:
        lis=[]
        contain_values = df[df['Unnamed: 0'].str.contains(y, na=False) & df['Unnamed: 0'].str.contains(x, na=False)]
        for i in contain_values.columns[1:]:
            a = contain_values.loc[:, i].sum()
            lis.append(a / len(contain_values.loc[:, i]))
        lis_lis.append(lis)

ag = [tuple([i,lis_lis[j][i]]) for j in range(len(lis_lis)) for i in range(len(lis_lis[0])) ]
ll= [ag[x:x+len(lis)] for x in range(0, len(ag), len(lis))]
p=0
leng = len(ll)
for u in range(len(ll)):
    ll.insert(p, colour[u])
    p+=2

bb= '''\\addplot[color={},mark=square,] 
    coordinates {{  
      {} 
    }};
    '''*leng
out = bb.format(*ll)
out = out.replace('[(','(')
out = out.replace(')]',')')
out = out.replace('), ',')')
print(out)
with open('readme.txt', 'w') as f:
    f.write(out)