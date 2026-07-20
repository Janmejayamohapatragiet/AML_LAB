import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.DataFrame({
    "Study_Hours": [2, 4, 6, 8],
    "Attendance": [70, 80, 85, 90],
    "Assignment_Marks": [12, 15, 18, 20],
    "Final_Marks": [45, 60, 75, 90]
})

X = df[["Study_Hours", "Attendance", "Assignment_Marks"]]
y = df["Final_Marks"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("\n----- Model Information -----")
print("Intercept :", model.intercept_)
print("Coefficients :", model.coef_)

print("\n----- Enter Student Details -----")

hours = float(input("Enter Study Hours : "))
attendance = float(input("Enter Attendance (%) : "))
assignment = float(input("Enter Assignment Marks : "))

new_student = pd.DataFrame({
    "Study_Hours": [hours],
    "Attendance": [attendance],
    "Assignment_Marks": [assignment]
})

prediction = model.predict(new_student)[0]

print("\nPredicted Final Marks = {:.2f}".format(prediction))

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\n----- Evaluation Metrics -----")
print("MAE  =", round(mae,2))
print("MSE  =", round(mse,2))
print("RMSE =", round(rmse,2))
print("R²   =", round(r2,2))

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.scatter(
    y,
    y_pred,
    color="limegreen",
    edgecolors="black",
    s=120,
    label="Predicted Points"
)

plt.plot(
    [min(y), max(y)],
    [min(y), max(y)],
    color="darkred",
    linestyle="--",
    linewidth=2,
    label="Ideal Line"
)

plt.title("Actual vs Predicted")
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)

metrics = ["MAE","RMSE","MSE","R²"]
values = [mae, rmse, mse, r2]

colors = ["red","royalblue","orange","green"]

plt.scatter(
    metrics,
    values,
    color=colors,
    s=220
)

for i, value in enumerate(values):
    plt.text(
        i,
        value + 0.02,
        f"{value:.2f}",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

plt.title("Evaluation Metrics")
plt.xlabel("Metrics")
plt.ylabel("Values")
plt.grid(True)

plt.tight_layout()
plt.show()