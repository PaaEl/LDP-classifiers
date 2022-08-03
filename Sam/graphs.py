import pandas as pd
'''
Point file to a file in this format (it's what the testsuite produces) 
(make sure depth is written as dpth, because it messes with the substring search)

;0.010;0.100;0.500;1.000;2.000;3.000;5.000
de/adult/dpth1/score_time;0.008;0.008;0.009;0.009;0.008;0.008;0.011
de/adult/dpth1/test_accuracy;0.317;0.505;0.675;0.756;0.762;0.754;0.756
de/adult/dpth1/test_balanced_accuracy;0.500;0.433;0.544;0.500;0.538;0.500;0.500
de/adult/dpth1/test_f1_macro;0.273;0.536;0.668;0.651;0.689;0.648;0.650
de/adult/dpth1/test_precision_macro;0.629;0.586;0.662;0.572;0.732;0.568;0.571
de/adult/dpth1/test_recall_macro;0.317;0.505;0.675;0.756;0.762;0.754;0.756
;0.010;0.100;0.500;1.000;2.000;3.000;5.000
de/mushroom/dpth1/score_time;0.003;0.003;0.003;0.003;0.003;0.003;0.003
de/mushroom/dpth1/test_accuracy;0.485;0.559;0.726;0.980;0.985;0.981;0.986
de/mushroom/dpth1/test_balanced_accuracy;0.500;0.549;0.734;0.980;0.985;0.981;0.986
de/mushroom/dpth1/test_f1_macro;0.317;0.494;0.720;0.980;0.985;0.981;0.986
de/mushroom/dpth1/test_precision_macro;0.235;0.597;0.761;0.981;0.986;0.982;0.987
de/mushroom/dpth1/test_recall_macro;0.485;0.559;0.726;0.980;0.985;0.981;0.986

Then change search1 and search2 to what you want to graph.
search1= ldp_mechanism and search2= ['balanced_accuracy'] for example will return the accuracy 
for each mechanism on every dataset.

It will writ the results like:
\addplot[color=blue,mark=square,] 
    coordinates {  
      (0, 0.002)(1, 0.003)(2, 0.003)(3, 0.003)(4, 0.003)(5, 0.004)(6, 0.004) 
    };
    \addplot[color=red,mark=square,] 
    coordinates {  
      (0, 0.2)(1, 0.2)(2, 0.214)(3, 0.289)(4, 0.526)(5, 0.53)(6, 0.443) 
    };
    \addplot[color=green,mark=square,] 
    coordinates {  
      (0, 0.075)(1, 0.172)(2, 0.196)(3, 0.453)(4, 0.671)(5, 0.672)(6, 0.642) 
    };
    \addplot[color=orange,mark=square,] 
    coordinates {  
      (0, 0.046)(1, 0.115)(2, 0.245)(3, 0.558)(4, 0.692)(5, 0.69)(6, 0.641) 
    };
    \addplot[color=cyan,mark=square,] 
    coordinates {  
      (0, 0.214)(1, 0.339)(2, 0.178)(3, 0.474)(4, 0.683)(5, 0.685)(6, 0.691) 
    };
    

It can't compare different classifiers yet nor dataset properties.
'''
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
search1= ldp_mechanism
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