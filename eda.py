import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_path = '/content/drive/MyDrive/magic04'
df = pd.read_csv(os.path.join(data_path, 'magic04.data'), header=None)
df.columns = [
    'fLength', 'fWidth', 'fSize', 'fConc', 'fConc1',
    'fAsym', 'fM3Long', 'f3Trans', 'fAlpha', 'fDist', 'class'
]

df.head()

#missing values in each column
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

#summary stats
df.info()
df.describe()

# class distribution
class_counts = df['class'].value_counts()
print("Class Distribution:\n", class_counts)

sns.countplot(x='class', data=df)
plt.title('Class Distribution')
plt.show()

# feature distribution plot
df.drop('class', axis=1).hist(bins=30, figsize=(14, 10), color='skyblue', edgecolor='black')
plt.suptitle('Feature Distribution')
plt.show()

# Correlation matrix
corr_matrix = df.drop('class', axis=1).corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()
