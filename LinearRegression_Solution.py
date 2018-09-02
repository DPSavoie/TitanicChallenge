#import dependencies
import pandas as pd
import scipy
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns  
color = sns.color_palette

#load data
train = pd.read_csv("D:/titanic/train.csv")
test = pd.read_csv("D:/titanic/test.csv")

##Create a new variable age categories

def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)

pivot = train.pivot_table(index="Age_categories",values='Survived')
pivot.plot.bar()

# plt.show()

def create_dummies (df, column_name): 
	dummies = pd.get_dummies(df[column_name], prefix=column_name)
	df = pd.concat([df, dummies], axis=1)
	return df

for column in ['Pclass', 'Sex', 'Age_categories']:
	train = create_dummies(train,column)
	test = create_dummies(test, column)

from sklearn.linear_model import LogisticRegression

columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']

lr = LogisticRegression()
lr.fit(train[columns], train['Survived'])
plt.show()


#This is where the data is split into training and testing

holdout = test # from now on we will refer to this
               # dataframe as the holdout data

from sklearn.model_selection import train_test_split

all_X = train[columns]
all_y = train['Survived']

train_X, test_X, train_y, test_y = train_test_split(
    all_X, all_y, test_size=0.20,random_state=0)


#Accuracy of prediction printed below
from sklearn.metrics import accuracy_score

lr = LogisticRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)
accuracy = accuracy_score(test_y, predictions)

print('Logistic Regression')
print(accuracy)

#Accuracy of 
from sklearn.model_selection import cross_val_score

lr = LogisticRegression()
scores = cross_val_score(lr, all_X, all_y, cv=10)
scores.sort()
accuracy = scores.mean()

print('Scores')
print(scores)

print('Accuracy')
print(accuracy)

#creating submission file
lr = LogisticRegression()
lr.fit(all_X,all_y)
holdout_predictions = lr.predict(holdout[columns])

holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 'Survived': holdout_predictions}

submission = pd.DataFrame(submission_df)

submission.to_csv("submission.csv",index=False)
