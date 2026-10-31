import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. IMPORTING THE DATASET
df = pd.read_csv('car.csv')

# Printing basic dataset info
print("--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATASET INFO ---")
df.info()

print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

print("\n--- DATA TYPES ---")
print(df.dtypes)

# 2. QUANTITATIVE & QUALITATIVE DATA SEPARATION
quantitative = df.select_dtypes(include=['int64', 'float64'])
print("\nQuantitative Column Labels:\n", quantitative.columns.tolist())

qualitative = df.select_dtypes(include=['object', 'category'])
print("\nQualitative Column Labels:\n", qualitative.columns.tolist())

# Skewness and Kurtosis
print("\nSkewness:\n", quantitative.skew())
print("\nKurtosis:\n", quantitative.kurt())


# 3. EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS
# Histogram for quantitative data (e.g., Selling_Price)
plt.figure(figsize=(8, 5))
plt.hist(df["Selling_Price"], bins=15, color='skyblue', edgecolor='black')
plt.xlabel("Selling Price")
plt.ylabel("Frequency")
plt.title("Selling Price Distribution")
plt.show()

# Boxplot for quantitative data
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Selling_Price"], color='lightgreen')
plt.title("Selling Price Box Plot")
plt.show()

# pie chart for qualitative data (e.g., Fuel_Type)  
plt.figure(figsize=(7, 7))
df["Fuel_Type"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=140,
)
plt.title("Fuel Type Distribution (Pie Chart)")
plt.ylabel("")  # Hide the y-label
plt.show()


# Bar chart for qualitative data (e.g., Brand)
plt.figure(figsize=(10, 5))
df["Brand"].value_counts().plot(kind="bar", color='coral')
plt.xlabel("Brand")
plt.ylabel("Count")
plt.title("Brand-wise Car Count")
plt.xticks(rotation=45)
plt.show()

# Correlation Matrix heatmap
plt.figure(figsize=(8, 6))
corr = df.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# 4. AUTOMATED EDA (Sweetviz)
import sweetviz as sv
try:
    report = sv.analyze(df)
    report.show_html("Sweetviz_Report.html")
    print("\nSweetviz Report generated successfully as 'Sweetviz_Report.html'")
except Exception as e:
    print("Sweetviz generation note:", e)


# 5. DATA PREPARATION & TRAIN-TEST SPLIT
# Feature Selection: Let's use Kilometers_Driven and Engine_CC as features to predict Selling_Price
X = df[['Kilometers_Driven', 'Engine_CC', 'Mileage_KMPL', 'Year', 'Seats']]
y = df['Selling_Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\nTraining Data Shape: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Testing Data Shape: X_test {X_test.shape}, y_test {y_test.shape}")


# 6. MODEL TRAINING & EVALUATION (Linear Regression, Ridge, Lasso)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

linear_r2 = r2_score(y_test, linear_pred)
linear_mse = mean_squared_error(y_test, linear_pred)
linear_rmse = np.sqrt(linear_mse)
linear_cv = cross_val_score(linear_model, X, y, cv=5, scoring="r2")

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)

ridge_r2 = r2_score(y_test, ridge_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_rmse = np.sqrt(ridge_mse)
ridge_cv = cross_val_score(ridge_model, X, y, cv=5, scoring="r2")

# Lasso Regression
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)

lasso_r2 = r2_score(y_test, lasso_pred)
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_rmse = np.sqrt(lasso_mse)
lasso_cv = cross_val_score(lasso_model, X, y, cv=5, scoring="r2")


# 7. MODEL COMPARISON RESULTS
results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression"
    ],
    "R² Score": [
        linear_r2,
        ridge_r2,
        lasso_r2
    ],
    "MSE": [
        linear_mse,
        ridge_mse,
        lasso_mse
    ],
    "RMSE": [
        linear_rmse,
        ridge_rmse,
        lasso_rmse
    ],
    "Cross Validation R² (Mean)": [
        linear_cv.mean(),
        ridge_cv.mean(),
        lasso_cv.mean()
    ]
})

print("\n================ MODEL COMPARISON ================")
print(results.to_string(index=False))

print("\n================ CROSS VALIDATION SCORES ================")
print("Linear Regression CV:", linear_cv)
print("Ridge Regression CV :", ridge_cv)
print("Lasso Regression CV :", lasso_cv)

# Visualizing Actual vs Predicted for Linear Regression
plt.figure(figsize=(8, 5))
plt.scatter(y_test, linear_pred, color='purple', alpha=0.6)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price (Linear Regression)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.show()