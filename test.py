#import dependencies
import pandas as pd
import matplotlib as plt
import scipy
import sklearn 
import seaborn as sns  
color = sns.color_palette

#read test file
train = pd.read_csv("D:/titanic/train.csv")
test = pd.read_csv("D:/titanic/test.csv")

df = train.append(test)

print(df.head())
print(df.info())
print(df.corr())

#found on kaggle, need to understand it

for i in df.columns:
	catdat = pd.Categorical(df[i])
	if len(catdat.categories)>9:
		continue
	print(i, " ", pd.Categorical(df[i]))

#check for na values

total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending = False)
missing_df_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_df_data)


#na values are survived, age, embarked, fare and cabin
# need to fillna to continue

df.Survived.fillna(value=-1, inplace=True)
df.Age.fillna(value=df.Age.mean(), inplace=True)
df.Embarked.fillna(value=(df.Embarked.value_counts().idxmax()), inplace=True)
df.Fare.fillna(value=df.Fare.mean(), inplace=True)

#print missing values
print(missing_df_data.isnull().sum())

