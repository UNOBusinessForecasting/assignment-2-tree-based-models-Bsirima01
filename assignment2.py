from xgboost import XGBClassifier
import joblib

# Step 1: Load the data
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"

train_data = pd.read_csv(train_url)
test_data = pd.read_csv(test_url)

# Step 2: Prepare the data
train_data['meal'] = train_data['meal'].astype(int)
X_train = train_data.drop('meal', axis=1)
y_train = train_data['meal']
X_test = test_data

# Step 3: Initialize and fit XGBoost model
model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
modelFit = model.fit(X_train, y_train)

# Step 4: Generate predictions
pred = modelFit.predict(X_test)

# Step 5: Save the model for later use
joblib.dump(modelFit, 'xgboost_model.pkl')

# Step 6: Output predictions (ensure they are binary)
pred = [1 if p > 0.5 else 0 for p in pred]
print(pred)  # Or save predictions to a file
