import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load training data
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
train_data = pd.read_csv(train_url)

# Load test data
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
test_data = pd.read_csv(test_url)

# Assuming 'meal' is already binary, but check and encode if necessary
train_data['meal'] = train_data['meal'].astype(int)

# --- Changes start here ---
# Convert 'DateTime' to numerical features
# Extract features from DateTime
for df in [train_data, test_data]:
    df['DateTime'] = pd.to_datetime(df['DateTime'])  # Convert to datetime objects
    df['Year'] = df['DateTime'].dt.year
    df['Month'] = df['DateTime'].dt.month
    df['Day'] = df['DateTime'].dt.day
    df['Hour'] = df['DateTime'].dt.hour
    df['Minute'] = df['DateTime'].dt.minute
# --- Changes end here ---

# Drop irrelevant columns or features, especially 'id' which is a string
X_train = train_data.drop(['meal', 'id', 'DateTime'], axis=1)  # Dropping 'id' and 'DateTime' columns
y_train = train_data['meal']

# The test data should have the same structure, minus the 'meal' column
# --- The fix: Explicitly drop 'meal' from X_test ---
X_test = test_data.drop(['id', 'DateTime', 'meal'], axis=1)  # Dropping 'id', 'DateTime', and 'meal' columns from test data 
# --- End of fix ---


# Initialize the model
model = DecisionTreeClassifier()

# Fit the model
modelFit = model.fit(X_train, y_train)

# Generate predictions
pred = modelFit.predict(X_test)

# Convert predictions to binary if needed
pred = [1 if p > 0.5 else 0 for p in pred]  # If you want hard classification
import joblib

# Save the model
joblib.dump(modelFit, 'modelFit.pkl')

