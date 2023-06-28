import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Read the data from a CSV file (assuming the data is in columns)
data = pd.read_csv('dataset_combine.csv')

# Split the data into features and labels
X = data[['Week', 'Temp_Max', 'Temp_Min', 'Humd_Max', 'Humd_Min', 'Nh3_Max', 'Nh3_Min', 'Feed']]
y = data['Weight']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

pickle.dump(model, open('model.pkl','wb'))

