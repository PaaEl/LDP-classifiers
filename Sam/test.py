import pandas as pd

# making data frame
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://media.geeksforgeeks.org/wp-content/uploads/nba.csv")

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1:-1], test_size=0.2)
# print(data)
# print(data.loc[data['Name'] == 'Jeff Withey'])
# a = data.index[data['Team'] == 'Boston Celtics'].tolist()
# print(a)
# # for i in a:
# # b = data.loc[[a.all()]]
# b = data.take(a)
# print(b)
print(X_train)
# print(X_train.loc[X_train['Name'] == 'Jeff Withey'])
a = X_train.index[X_train.iloc[:, 1] == 'Boston Celtics'].tolist()
print(a)
# for i in a:
# b = X_train.loc[[a.all()]]
b = X_train.take(a)
print(b)
