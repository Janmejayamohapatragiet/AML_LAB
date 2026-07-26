import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

url = "https://raw.githubusercontent.com/RajeshRanaGiet/AML-LAB/main/Experiment%203/studentGradeDataSet.csv"

df = pd.read_csv(url)

print("\n========= DATASET INFORMATION =========")
print(df.info())

print("\n========= FIRST 5 RECORDS =========")
print(df.head())

print("\n========= STATISTICAL SUMMARY =========")
print(df.describe())

print("\n========= MISSING VALUES =========")
print(df.isnull().sum())

plt.figure(figsize=(7,6))
sns.heatmap(df.corr(), annot=True, cmap="viridis")
plt.title("Correlation Matrix")
plt.show()

for column in df.columns:
    plt.figure(figsize=(4,4))
    sns.boxplot(y=df[column], color="skyblue")
    plt.title(f"Boxplot of {column}")
    plt.show()

X = df[['SEM 1','SEM 2','SEM 3','SEM 4']]
y = df['SEM 5']

corr = df.corr()['SEM 5'].drop('SEM 5')
best_feature = corr.abs().idxmax()

print("\nBest Feature:", best_feature)

X_simple = df[[best_feature]]

X_train, X_test, y_train, y_test = train_test_split(
    X_simple,
    y,
    test_size=0.20,
    random_state=10
)

slr = LinearRegression()
slr.fit(X_train, y_train)

y_pred = slr.predict(X_test)

print("\n===== SIMPLE LINEAR REGRESSION =====")
print("Intercept :", slr.intercept_)
print("Coefficient :", slr.coef_[0])

print("MAE :", mean_absolute_error(y_test,y_pred))
print("MSE :", mean_squared_error(y_test,y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test,y_pred)))
print("R2 Score :", r2_score(y_test,y_pred))

plt.figure(figsize=(6,5))
plt.scatter(X_test, y_test, color="blue")
plt.plot(X_test, y_pred, color="red")
plt.xlabel(best_feature)
plt.ylabel("SEM 5")
plt.title("Simple Linear Regression")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=10
)

mlr = LinearRegression()
mlr.fit(X_train, y_train)

prediction = mlr.predict(X_test)

print("\n===== MULTIPLE LINEAR REGRESSION =====")

print("Intercept:")
print(mlr.intercept_)

print("\nCoefficients:")
for name, value in zip(X.columns, mlr.coef_):
    print(name, ":", value)

print("\nPerformance")

mae = mean_absolute_error(y_test,prediction)
mse = mean_squared_error(y_test,prediction)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,prediction)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2)

plt.figure(figsize=(6,5))
plt.scatter(y_test, prediction, color="green")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual SEM 5")
plt.ylabel("Predicted SEM 5")
plt.title("Actual vs Predicted")
plt.show()

metrics = ["MAE","MSE","RMSE","R2"]
values = [mae,mse,rmse,r2]

plt.figure(figsize=(6,4))
plt.bar(metrics, values, color=["orange","purple","green","blue"])
plt.title("Model Performance")
plt.ylabel("Value")
plt.show()

print("\nPredict Semester 5 Marks")

sem1 = float(input("Enter Semester 1 Marks: "))
sem2 = float(input("Enter Semester 2 Marks: "))
sem3 = float(input("Enter Semester 3 Marks: "))
sem4 = float(input("Enter Semester 4 Marks: "))

new_student = pd.DataFrame({
    "SEM 1":[sem1],
    "SEM 2":[sem2],
    "SEM 3":[sem3],
    "SEM 4":[sem4]
})

result = mlr.predict(new_student)

print("\nPredicted Semester 5 Marks =", round(result[0],2))