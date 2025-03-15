import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans

# Load the dataset from a CSV file
X_train = pd.read_csv('Train_x.csv', header=None)  # Replace 'your_file.csv' with your actual file path
y_train = pd.read_csv('Train_y.csv', header=None)
X_test = pd.read_csv('Test_x.csv', header=None)
#X_test = pd.read_csv('Test_x_2.csv', header=None)
y_test = pd.read_csv('Test_y.csv', header=None)
X_train = np.array(X_train)
y_train = np.array(y_train).ravel()
X_test = np.array(X_test)
y_test = np.array(y_test) .ravel()
#X_train = np.delete(X_train, 0, axis=1)
#X_test = np.delete(X_test, 0, axis=1)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
model = LinearRegression()
model_2 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

model.fit(X_train, y_train)
model_2.fit(X_train, y_train)
#print("Weights (Coefficients):", model.coef_)  # Model slope(s)
#print("Weights_2 (Coefficients):", model_2.feature_importances_) 
y_pred = model.predict(X_test)
probabilities = np.array(y_pred)
y_pred_2 = model_2.predict(X_test)
probabilities_2 = np.array(y_pred_2)
probabilities = probabilities.flatten()
first_half = probabilities[:16]
second_half = probabilities[16:]
top3_indices = np.argsort(first_half)[-3:]
top3_second = np.argsort(second_half)[-3:]
binary_output = np.zeros_like(first_half)
binary_output_2 = np.zeros_like(second_half)

binary_output[top3_indices] = 1
binary_output_2[top3_second] = 1
merged_output = np.concatenate([binary_output, binary_output_2])

probabilities_2 = probabilities_2.flatten()
first_half_2 = probabilities_2[:16]
second_half_2 = probabilities_2[16:]
top3_indices_2 = np.argsort(first_half_2)[-3:]
top3_second_2 = np.argsort(second_half_2)[-3:]
binary_output_3 = np.zeros_like(first_half_2)
binary_output_4 = np.zeros_like(second_half_2)

binary_output_3[top3_indices_2] = 1
binary_output_4[top3_second_2] = 1
merged_output_2 = np.concatenate([binary_output_3, binary_output_4])
mse = mean_squared_error(y_test, merged_output)
r2 = r2_score(y_test, merged_output)
precision = precision_score(y_test, merged_output)
recall = recall_score(y_test, merged_output)
f1 = f1_score(y_test, merged_output)
auroc = roc_auc_score(y_test, probabilities)
print(binary_output)
print(binary_output_2)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUROC: {auroc:.4f}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
print("----------------------------------------------------")
print("Cross_validation below")
r2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print("R² scores:", r2_scores)
print("Average R²:", r2_scores.mean())

mse_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=make_scorer(mean_squared_error, greater_is_better=False))
print("MSE scores:", mse_scores)
print("Average MSE:", mse_scores.mean())
print("----------------------------------------------------")
mse_2 = mean_squared_error(y_test, merged_output_2)
r2_2 = r2_score(y_test, merged_output_2)
precision_2 = precision_score(y_test, merged_output_2)
recall_2 = recall_score(y_test, merged_output_2)
f1_2 = f1_score(y_test, merged_output_2)
auroc_2 = roc_auc_score(y_test, probabilities_2)
print(binary_output_3)
print(binary_output_4)
print(f"Precision: {precision_2:.4f}")
print(f"Recall: {recall_2:.4f}")
print(f"F1-score: {f1_2:.4f}")
print(f"AUROC: {auroc_2:.4f}")
print(f"Mean Squared Error_2: {mse_2}")
print(f"R^2 Score_2: {r2_2}")
print("----------------------------------------------------")
print("Cross_validation_2 below")
r2_scores_2 = cross_val_score(model_2, X_train, y_train, cv=5, scoring='r2')
print("R² scores:", r2_scores_2)
print("Average R²:", r2_scores_2.mean())

mse_scores_2 = cross_val_score(model_2, X_train, y_train, cv=5, scoring=make_scorer(mean_squared_error, greater_is_better=False))
print("MSE scores:", mse_scores_2)
print("Average MSE:", mse_scores_2.mean())

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_train)

# Print cluster labels
print("Cluster Assignments:", clusters)
'''
# Split data into two clusters
first_half_3 = X_train[clusters == 0]
second_half_3 = X_train[clusters == 1]

# Find top 3 based on distance from cluster centroid
distances_first = np.linalg.norm(first_half_3 - kmeans.cluster_centers_[0], axis=1)
distances_second = np.linalg.norm(second_half_3 - kmeans.cluster_centers_[1], axis=1)

top3_first_indices = np.argsort(distances_first)[:3]
top3_second_indices = np.argsort(distances_second)[:3]

# Create binary output
binary_output_first = np.zeros(len(first_half_3))
binary_output_second = np.zeros(len(second_half_3))
binary_output_first[top3_first_indices] = 1
binary_output_second[top3_second_indices] = 1

# Merge back into original order
merged_output_3 = np.zeros(len(X_train))
merged_output_3[clusters == 0] = binary_output_first
merged_output_3[clusters == 1] = binary_output_second

# Print results
print("Binary Output (Merged):", merged_output_3)'''