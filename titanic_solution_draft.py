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

df = train.append(test)

#drop these columns, not useful for analysis
train = train.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

#investigate column heads and data types
print('-----------------------------------')
print(train.head())
print(train.info())

print('Categorical Variables')
print(train.describe())

#found on kaggle, need to understand it
print('-----------------------------------')
for i in train.columns:
	catdat = pd.Categorical(df[i])
	if len(catdat.categories)>9:
		continue
	print(i, " ", pd.Categorical(df[i]))

#check for na values

total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending = False)
missing_df_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


#na values are survived, age, embarked, fare and cabin
# need to fillna to continue

train.Survived.fillna(value=-1, inplace=True)
train.Age.fillna(value=df.Age.mean(), inplace=True)
train.Embarked.fillna(value=(df.Embarked.value_counts().idxmax()), inplace=True)
train.Fare.fillna(value=df.Fare.mean(), inplace=True)

#print missing values
print('-----------------------------------')
print('Missing Values after Cleaning')
print(missing_df_data.isnull().sum())


# #Histograms of survival

# sns.countplot(x='Embarked', data=train, ax=axis1)
# sns.countplot(x='Survived', hue='Embarked', data=train, order=[1,0], ax=axis2)
# plt.show()

#Select variable for prediction - all numerical variables

# y =  train.Survived
# train_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
# X = train[train_features]

# print(X.head())
# print(X.describe())

# #defective code, cound not convert string to float 
# from sklearn.tree import DecisionTreeRegressor

# train = DecisionTreeRegressor(random_state=1)

# train.fit(X,y)


#survival rate by passenger class
print('-----------------------------------')
print('Survival Rate by Class')
p_table = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(p_table)

#class and survival rate correlates

#Survival rate by gender
print('Survival Rate by Gender')
Survived_gender = train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(Survived_gender)

#Survival rate by SibSp
print('Survival Rate by Number of Siblings/Spouses')
Survived_Sib = train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='SibSp', ascending=False)
print(Survived_Sib)

#Survival rate by Partners onboard
print('survival rate by partners on board')
Survived_P = train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Parch', ascending=False)
print(Survived_P)

#histogram of survived by age
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()

#Graph of age vs Survived
facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.show()

#Graph of age vs Survived
facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Pclass', shade=True)
facet.set(xlim=(0, train['Pclass'].max()))
facet.add_legend()
plt.show()

#histogram of survived, pclass and age
grid = sns.FacetGrid(train, col='Survived')
grid.map(plt.hist, 'Pclass', bins=20)
grid.add_legend();
plt.show()

#histogram of survived and fare
grid = sns.FacetGrid(train, col='Survived', row='Fare', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
plt.show()

#see if destination of embarking impacted survival
grid=sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.lmplot, 'Pclass', 'Survived', 'Sex')
grid.add_legend()
plt.show()

#combine family size, create a new variable
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

facet = sns.FacetGrid(train, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'FamilySize', shade=True)
facet.set(xlim=(0, train['FamilySize'].max()))
facet.add_legend()
plt.show()