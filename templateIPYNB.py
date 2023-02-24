# Advanced Data Science Project Cheatsheet

## Step 1: Data Collection

### Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Load the data from a file
df = pd.read_csv('data.csv')

## Step 2: Data Cleaning

### Drop any duplicate rows
df = df.drop_duplicates()

### Handle missing data
# Fill in missing values with the median or mean
df['column_name'] = df['column_name'].fillna(df['column_name'].median())

# Drop rows with missing values
df = df.dropna()

### Handle outliers
# Remove any outliers based on domain knowledge or statistical analysis
df = df[df['column_name'] < 100]

### Convert data types
# Convert strings to datetime objects
df['date_column'] = pd.to_datetime(df['date_column'])

### Feature engineering
# Create new features based on existing features
df['new_column'] = df['column_1'] + df['column_2']

## Step 3: Exploratory Data Analysis

### Data visualization
# Create a histogram of a numerical feature
sns.histplot(data=df, x='column_name', bins=10)

# Create a scatter plot of two numerical features
sns.scatterplot(data=df, x='column_1', y='column_2')

# Create a box plot of a numerical feature by a categorical feature
sns.boxplot(data=df, x='categorical_column', y='numerical_column')

# Create a line plot of a time series
sns.lineplot(data=df, x='date_column', y='numerical_column')

### Statistical analysis
# Calculate summary statistics
df.describe()

# Calculate correlation coefficients
df.corr()

## Step 4: Feature Selection

### Correlation analysis
# Calculate correlation matrix
corr_matrix = df.corr()

# Create a heatmap of correlation matrix
sns.heatmap(corr_matrix)

### Feature importance
# Use decision tree or random forest to determine feature importance
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(df.drop('target_column', axis=1), df['target_column'])
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

## Step 5: Model Selection and Training

### Prepare the data for modeling
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target_column', axis=1), df['target_column'], test_size=0.2)

### Train the model
# Linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Decision tree
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=5)
model.fit(X_train, y_train)

# Random forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

## Step 6: Model Evaluation

### Evaluate the model on the testing set
# Calculate mean squared error
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print('R-squared:', r2)

# Visualize the results
plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title('Regression Analysis')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Use k-fold cross-validation to estimate model performance
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print('Cross-validation scores:', scores)
print('Average score:', scores.mean())


# Set number of folds for k-fold cross-validation
num_folds = 10

# Use the cross_val_score() function to compute the accuracy of the model for each fold
scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring='accuracy')

# Print the accuracy score for each fold
print("Accuracy scores for each fold:\n", scores)

# Compute the mean accuracy score and standard deviation of the accuracy scores
mean_accuracy = scores.mean()
std_accuracy = scores.std()
print("\nMean accuracy score:", mean_accuracy)
print("Standard deviation of accuracy scores:", std_accuracy)