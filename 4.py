import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Load the dataset
data_Set = pd.read_csv('big_mart_data.csv')

# Fill missing values in 'Item_Weight'
data_Set['Item_Weight'].fillna(data_Set['Item_Weight'].mean(), inplace=True)

# Replace missing 'Outlet_Size' based on 'Outlet_Type' mode
mode = data_Set.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
missing = data_Set['Outlet_Size'].isnull()
data_Set.loc[missing, 'Outlet_Size'] = data_Set.loc[missing, 'Outlet_Type'].apply(lambda x: mode[x])

# Replace 'Item_Fat_Content' inconsistencies
data_Set.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}}, inplace=True)

# Encode categorical variables
encoder = LabelEncoder()
categorical_columns = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type',
                       'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

# Encode categorical columns using LabelEncoder
for col in categorical_columns:
    data_Set[col] = encoder.fit_transform(data_Set[col])

# Define features and target
X = data_Set.drop('Item_Outlet_Sales', axis=1)
Y = data_Set['Item_Outlet_Sales']

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the model
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)

# Function to request input from the user and predict item sales
def get_user_input():
    input_data = {}

    # Request numeric input
    input_data['Item_Weight'] = float(input("Enter Item Weight: "))
    input_data['Item_Visibility'] = float(input("Enter Item Visibility: "))
    input_data['Item_MRP'] = float(input("Enter Item MRP: "))
    input_data['Outlet_Establishment_Year'] = int(input("Enter Outlet Establishment Year: "))

    # Request categorical input
    input_data['Item_Identifier'] = input("Enter Item Identifier (e.g., FDA15): ")
    input_data['Item_Fat_Content'] = input("Enter Item Fat Content (Low Fat, Regular): ")
    input_data['Item_Type'] = input("Enter Item Type (e.g., Dairy, Soft Drinks): ")
    input_data['Outlet_Identifier'] = input("Enter Outlet Identifier (e.g., OUT049): ")
    input_data['Outlet_Size'] = input("Enter Outlet Size (Small, Medium, High): ")
    input_data['Outlet_Location_Type'] = input("Enter Outlet Location Type (Tier 1, Tier 2, Tier 3): ")
    input_data['Outlet_Type'] = input("Enter Outlet Type (Supermarket Type1, Grocery Store): ")

    # Encode categorical features using the same LabelEncoder
    for col in categorical_columns:
        input_data[col] = encoder.transform([input_data[col]])[0]

    # Convert the input into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Reorder the columns to match the training data
    input_df = input_df[X_train.columns]

    return input_df

# Get user input
user_input_df = get_user_input()

# Make a prediction based on user input
predicted_sales = regressor.predict(user_input_df)

# Display the prediction
print(f"Predicted Item Outlet Sales: {predicted_sales[0]}nah