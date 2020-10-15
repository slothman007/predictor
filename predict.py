import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
import warnings
import pickle
warnings.filterwarnings('ignore')

train = pd.read_csv("Training.csv")
test = pd.read_csv("Testing.csv")
X = train.drop(['prognosis'],axis=1)


def evaluate(train_data , kmax , algo):
    test_scores={}
    train_scores={}
    for i in range (2 , kmax , 2):
        kf = KFold(n_splits = i)
        sum_train = 0
        sum_test = 0
        data = train_data
        for train,test in kf.split(data):
            train_data = data.iloc[train,:]
            test_data = data.iloc[test,:]
            x_train = train_data.drop(['prognosis'] , axis = 1)
            y_train = train_data['prognosis']
            x_test = test_data.drop(['prognosis'] , axis = 1)
            y_test = test_data['prognosis']
            algo_model = algo.fit(x_train,y_train)
            sum_train += algo_model.score(x_train,y_train)
            y_pred = algo_model.predict(x_test)
            sum_test += accuracy_score(y_test,y_pred)
        average_test = sum_test/i
        average_train = sum_train/i
        test_scores[i] = average_test
        train_scores[i] = average_train
        print('kvalue: ',i)
    return (test_scores,train_scores)

dt = DecisionTreeClassifier(criterion='entropy',)
Test_score,Train_score = evaluate(train,4,dt)
print("Test :" + str(Test_score))
print("Train :"+ str(Train_score))
pickle.dump(dt,open('model.pkl','wb'))
a = list(range(2,134))
i_name  = (input('Enter your name :'))
i_age = (int(input('Enter your age:')))
for i in range(len(X.columns)):
    print(str(i+1+1) + ":", X.columns[i])
choices = input('Enter the Serial no.s which is your Symptoms are exist:  ')
b = [int(X) for X in choices.split()]
count = 0
while count < len(b):
    item_to_replace =  b[count]
    replacement_value = 1
    indices_to_replace = [i for i,X in enumerate(a) if X==item_to_replace]
    count += 1
    for i in indices_to_replace:
        a[i] = replacement_value
a = [0 if X !=1 else X for X in a]
y_diagnosis = dt.predict([a])
y_pred_2 = dt.predict_proba([a])
print(('Name of the infection = %s , confidence score of : = %s') %(y_diagnosis[0],y_pred_2.max()* 100),'%' )
print(('Name = %s , Age : = %s') %(i_name,i_age))