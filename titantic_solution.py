
#import dependencies
import pandas as pd
import scipy
import sklearn
import matplotlib as plt
import seaborn as sns  
color = sns.color_palette

#read test file
ti = pd.read_csv("D:/titanic/train.csv")

#drop these columns, not useful for analysis
ti_f = ti.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

#investigate column heads and data types
print(ti_f.head())
print(ti_f.info())
print(ti_f.describe())

# #Histograms of survival

# sns.countplot(x='Embarked', data=ti_f, ax=axis1)
# sns.countplot(x='Survived', hue='Embarked', data=ti_f, order=[1,0], ax=axis2)
# plt.show()

#Select variable for prediction - all numerical variables
y =  ti_f.Survived
ti_features = ['Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
X = ti_f[ti_features]

print(X.head())
print(X.describe())

#defective code, cound not convert string to float 
from sklearn.tree import DecisionTreeRegressor

ti_model = DecisionTreeRegressor(random_state=1)

ti_model.fit(X,y)