#import dependencies
import pandas as pd
import numpy as np
import scipy
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#load data
train = pd.read_csv("D:/titanic/train.csv")
test = pd.read_csv("D:/titanic/test.csv")


#Data cleaning for NA
train['Embarked'] = train['Embarked'].fillna('S')
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())


#Map categorical variables to numbers
embarked_dict = {'S':0, 'C':1, 'Q':2}
sex_dict = {'male':0, 'female':1}
train["Embarked_num"] = train["Embarked"].map(embarked_dict)
train["Sex_num"] = train["Sex"].map(sex_dict)
test["Embarked_num"] = test["Embarked"].map(embarked_dict)
test["Sex_num"] = test["Sex"].map(sex_dict)

#create new variable for family size
train['family_size'] = train['SibSp'] + train['Parch'] + 1
test['family_size'] = test['SibSp'] + test['Parch'] + 1

#create random forest features
features_forest = train[["Pclass", "Age", "Sex_num", "Fare", "SibSp", "Parch", "Embarked_num", "family_size"]].values
target = train['Survived'].values

#Print random forest and target score 
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)
print(my_forest.score(features_forest, target))

#Test features and length of prediction
test_features = test[["Pclass", "Age", "Sex_num", "Fare", "SibSp", "Parch", "Embarked_num", "family_size"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))

#print feature scores 
print(my_forest.feature_importances_)
print(my_forest.score(features_forest, target))

#Export to csv module
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])
print(my_solution.shape)
print(my_solution.describe())

my_solution.to_csv("predictions.csv", index_label="PassengerId")