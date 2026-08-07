# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

path_to_file = "student.csv"

# Load the dataset and explore it
df = pd.read_csv(path_to_file)


df.head()
df.plot.scatter(x='Hours', y='Scores', title='Hours vs Scores')
print(df.corr())
print(df.describe())

SEED = 42

# Split the data into training (80%) and testing (20%) sets
x_train, x_test, y_train, y_test = train_test_split(df['Hours'].values, df['Scores'].values, test_size=0.2, random_state=SEED)

# Create and train the linear regression model
regression_model = LinearRegression()
regression_model.fit(x_train.reshape(-1, 1), y_train)

# Print the learned parameters
print("Coefficients: ", regression_model.coef_)
print("Intercept: ", regression_model.intercept_)

def calculate_score(slope, intercept, hours):
    # Use the line equation y = mx + c to predict score
    return slope * hours + intercept

# Predict & print the score for 9.25 hours of study
print("Predicted score for 9.25 hours: ", regression_model.predict([[9.25]]))
# print("Predicted score for 9.25 hours: ", calculate_score(regression_model.coef_[0], regression_model.intercept_, 9.25))

y_pred = regression_model.predict(x_test.reshape(-1, 1))

df_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) 
print(df_pred)

mean_absolute_error = sum(abs(y_test - y_pred)) / len(y_test)
print("Mean Absolute Error: ", mean_absolute_error)

mean_squared_error = sum((y_test - y_pred) ** 2) / len(y_test)
print("Mean Squared Error: ", mean_squared_error)

mean_squared_log_error = sum((np.log1p(y_test) - np.log1p(y_pred)) ** 2) / len(y_test)
print("Mean Squared Log Error: ", mean_squared_log_error)