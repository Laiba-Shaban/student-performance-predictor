import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load data and separate features (X) and target (y)
data = {
    'Hours_Studied': [3, 5, 7, 9, 10, 4, 6, 11],
    'Attendance': [75, 85, 92, 88, 96, 65, 78, 94],
    'Sleep_Hours': [7, 6, 8, 5, 7, 6, 8, 9],
    'Marks': [58, 70, 83, 86, 92, 60, 74, 96]
}

df = pd.DataFrame(data)

X = df[['Hours_Studied', 'Attendance', 'Sleep_Hours']]
y = df['Marks']

# Step 2: Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Step 3: Predict for a new student
new_student = [[7, 88, 7]]
predicted_marks = model.predict(new_student)
print(f"Predicted Marks for the new student: {predicted_marks[0]:.2f}")

# Step 4: Evaluate performance
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")