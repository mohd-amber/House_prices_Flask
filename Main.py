# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from flask import Flask, render_template, request
import joblib
from sklearn.impute import SimpleImputer
data = pd.read_csv('new_data.csv')
data.dropna(axis=1, inplace=True)
# Identify columns with missing values
missing_values = data.isnull().sum()
print("Columns with missing values:\n", missing_values)

# Identify columns with non-numeric values
non_numeric_columns = data.select_dtypes(exclude=['float64', 'int64']).columns
print("Columns with non-numeric values:\n", non_numeric_columns)

# Inspect specific rows or columns with issues
# For example, you can print out rows containing NaN values
nan_rows = data[data.isnull().any(axis=1)]
print("Rows with NaN values:\n", nan_rows)

# Print out unique values for non-numeric columns
for column in non_numeric_columns:
    unique_values = data[column].unique()
    print(f"Unique values for column '{column}':\n", unique_values)
	
# Load the dataset
data = pd.read_csv('new_data.csv')

# Identify numeric and non-numeric columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
non_numeric_columns = data.select_dtypes(exclude=['float64', 'int64']).columns

# Apply mean imputation to numeric columns and most frequent imputation to non-numeric columns
imputer_numeric = SimpleImputer(strategy='mean')
imputer_non_numeric = SimpleImputer(strategy='most_frequent')

# Impute missing values separately for numeric and non-numeric columns
data_imputed_numeric = imputer_numeric.fit_transform(data[numeric_columns])
data_imputed_non_numeric = imputer_non_numeric.fit_transform(data[non_numeric_columns])

# Concatenate imputed data back together
data_imputed = pd.concat([pd.DataFrame(data_imputed_numeric, columns=numeric_columns),
                          pd.DataFrame(data_imputed_non_numeric, columns=non_numeric_columns)],
                         axis=1)

# Check if there are any remaining missing values
missing_values = data_imputed.isnull().sum()
print("Missing values after imputation:\n", missing_values)
data_imputed.head()
from sklearn.impute import SimpleImputer
data_imputed.to_csv('new_data.csv',index=False)
# Identify categorical columns
categorical_columns = ['Location']

# Initialize SimpleImputer to fill missing values with the most frequent value
imputer_categorical = SimpleImputer(strategy='most_frequent')

# Fit and transform the imputer on the dataset for categorical columns
data[categorical_columns] = imputer_categorical.fit_transform(data[categorical_columns])
# Data preprocessing
# Handling missing values
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(include=['object']).columns

numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = SimpleImputer(strategy='most_frequent')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

X = preprocessor.fit_transform(data)
X = pd.DataFrame(X, columns=numeric_features.append(categorical_features))


# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# One-hot encode categorical columns
X = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Split data into features and target variable
y = X['Price']
X = X.drop('Price', axis=1)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)	
from sklearn.impute import SimpleImputer

# Initialize SimpleImputer to fill missing values with the mean
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on the training features
X_train_imputed = imputer.fit_transform(X_train)

# Transform the testing features using the same imputer
X_test_imputed = imputer.transform(X_test)


# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'linear_regression_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')


# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    if request.method == 'POST':
        input_data = request.form.to_dict()
        input_values = [float(input_data[feature]) for feature in X.columns]
        input_values = [input_values]  # Convert to 2D array

        # Load the model and make prediction
        model = joblib.load('linear_regression_model.pkl')
        prediction = model.predict(input_values)

        return render_template('result.html', prediction=prediction)
		
if __name__ == '__main__':
    # Run the app on port 8080
    app.run(debug=True, port=8080)
		
