import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris(as_frame=True)

df = iris.frame


sns.set_style("whitegrid")
sns.set_palette("Set2")
sns.set_context("talk")

# Map numbers to species names
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})


# Plots

#sns.scatterplot(x='petal width (cm)', y='sepal width (cm)', data=df, hue='species' )

#sns.histplot(df["sepal length (cm)"], bins = 20)

#sns.boxplot(x='petal width (cm)', y='sepal width (cm)', data=df, hue='species')

#sns.barplot(x='petal width (cm)', y='sepal width (cm)', data=df, hue='species')

#sns.pairplot(df, hue = 'species')

#sns.relplot(x='petal width (cm)', y='sepal width (cm)', data=df, col='species', kind='scatter')



# DATA PREP


# only setosas
#setosa_df = df[df["species"] == 'setosa']

#sns.scatterplot(x='petal length (cm)', y='petal width (cm)', data=setosa_df)


#Select only 2 columns
#df_subset = df[['petal width (cm)', 'sepal width (cm)']]

# Filter by numeric range
df_filtered = df[df['petal width (cm)'] > 1.5]

# Create new columns

"""df['petal area'] = df['petal width (cm)'] * df['petal length (cm)']

sns.boxplot(x='species', y='petal area', data = df)"""

# Grouping and aggregating
"""df.groupby('species')['petal width (cm)'].mean()

df_grouped = df.groupby('species', as_index = False)['petal width (cm)'].mean()
sns.barplot(x='species', y='petal width (cm)', data=df_grouped)"""

# easy rename columns

"""df = df.rename(columns=lambda x: x.replace(" (cm)", "").replace(" ", "_"))
print(df.head())"""

#combine filters

"""df_filtered = df[(df['species'] == 'virginica') & (df['petal length (cm)'] > 5)] # only virginicas

print(df_filtered)"""

"""plt.title("Iris petal width vs. sepal width")"""
plt.xlabel("Species")
plt.ylabel("Petal area")

plt.show()
