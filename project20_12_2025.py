import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
df = pd.read_csv("C:/Users/munwa/Downloads/predictive_maintenance.csv")



le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])
df['Failure Type'] = le.fit_transform(df['Failure Type'])

X = df.drop(['Target', 'Product ID', 'UDI'], axis=1)
y = df['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



plt.figure()
sns.countplot(x='Target', data=df)
plt.title("Target Distribution")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.drop(['Product ID'], axis=1).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df['Torque [Nm]'],
    y=df['Tool wear [min]'],
    hue=df['Target'],
    s=20,
    alpha=0.6
)
plt.title("Classification View: Torque vs Tool Wear")
plt.show()



X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)



lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred_lr),
            annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Logistic Regression")
plt.show()

print("Logistic Regression Report:\n")
print(classification_report(y_test, y_pred_lr))



importance = np.abs(lr_model.coef_[0])
imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values(by='Importance')

plt.figure()
plt.barh(imp_df['Feature'], imp_df['Importance'])
plt.title("Feature Importance (Logistic Regression)")
plt.show()



df['ToolWear_Bin'] = pd.cut(df['Tool wear [min]'], bins=15)
avg_torque = df.groupby('ToolWear_Bin')['Torque [Nm]'].mean()

plt.figure()
avg_torque.plot(marker='o')
plt.xlabel("Tool Wear (Binned)")
plt.ylabel("Average Torque (Nm)")
plt.title("Average Torque vs Tool Wear")
plt.show()

plt.figure()
sns.kdeplot(df['Tool wear [min]'], fill=True)
plt.title("Tool Wear Distribution")
plt.show()



kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df['Torque [Nm]'],
    y=df['Tool wear [min]'],
    hue=df['Cluster'],
    palette='Set1',
    s=20,
    alpha=0.6
)
plt.title("K-Means Clustering (Clear View)")
plt.show()





cv_lr = cross_val_score(lr_model, X_scaled, y, cv=5, scoring='accuracy')
print("Logistic Regression CV Scores:", cv_lr)
print("LR Mean CV Accuracy:", cv_lr.mean())
print("LR Std Dev:", cv_lr.std())


# Decision Tree CV
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train,y_train)
dt_pred=dt_model.predict(X_test)
cv_dt = cross_val_score(dt_model, X_scaled, y, cv=5, scoring='accuracy')
print("\nDecision Tree CV Scores:", cv_dt)
print("DT Mean CV Accuracy:", cv_dt.mean())
print("DT Std Dev:", cv_dt.std())
plt.figure()
sns.heatmap(confusion_matrix(y_test,dt_pred),
            annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Decision Tree")
plt.show()


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train,y_train)
knn_pred=knn_model.predict(X_test)
cv_knn = cross_val_score(knn_model, X_scaled, y, cv=5, scoring='accuracy')
print("\nKNN CV Scores:", cv_knn)
print("KNN Mean CV Accuracy:", cv_knn.mean())
print("KNN Std Dev:", cv_knn.std())
plt.figure()
sns.heatmap(confusion_matrix(y_test,knn_pred),
            annot=True, fmt='d', cmap="Oranges")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – KNN")
plt.show()
