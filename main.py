import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------
# 1. Load Data
# -------------------------------
df = pd.read_csv("churn.csv")

print("Dataset Loaded ✅")

# -------------------------------
# 2. Encode Categorical Data
# -------------------------------
le = LabelEncoder()
df["contract_type"] = le.fit_transform(df["contract_type"])

# -------------------------------
# 3. Features & Target
# -------------------------------
X = df.drop("churn", axis=1)
y = df["churn"]

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. Logistic Regression
# -------------------------------
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

# -------------------------------
# 6. Random Forest
# -------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# -------------------------------
# 7. Evaluation Function
# -------------------------------
def evaluate(y_true, y_pred, name):
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

evaluate(y_test, y_pred_lr, "Logistic Regression")
evaluate(y_test, y_pred_rf, "Random Forest")

# -------------------------------
# 8. Confusion Matrix Graph
# -------------------------------
plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# 9. Feature Importance
# -------------------------------
importance = rf_model.feature_importances_
features = X.columns

plt.figure()
sns.barplot(x=importance, y=features)
plt.title("Feature Importance")
plt.show()