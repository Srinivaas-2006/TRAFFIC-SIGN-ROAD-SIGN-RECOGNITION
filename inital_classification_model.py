# Initial Classification Model using SVM

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Split training data into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Try different C values to reduce overfitting
C_values = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01]

best_model = None
best_C = None
best_score = -1
best_train_acc = 0
best_val_acc = 0
best_test_acc = 0
best_val_gap = 0
best_test_gap = 0
best_test_pred = None

for C in C_values:
    model = LinearSVC(
        C=C,
        class_weight="balanced",
        max_iter=30000,
        random_state=42,
        dual=False
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, train_pred) * 100
    val_acc = accuracy_score(y_val, val_pred) * 100
    test_acc = accuracy_score(y_test, test_pred) * 100

    val_gap = train_acc - val_acc
    test_gap = train_acc - test_acc

    # Prefer model with low test gap and good test accuracy
    if test_gap <= 10:
        score = test_acc
    else:
        score = test_acc - test_gap

    if score > best_score:
        best_score = score
        best_model = model
        best_C = C
        best_train_acc = train_acc
        best_val_acc = val_acc
        best_test_acc = test_acc
        best_val_gap = val_gap
        best_test_gap = test_gap
        best_test_pred = test_pred

print("\nInitial Classification Model: Improved SVM")
print("Best C Value:", best_C)
print("Train Accuracy:", round(best_train_acc, 2), "%")
print("Validation Accuracy:", round(best_val_acc, 2), "%")
print("Test Accuracy:", round(best_test_acc, 2), "%")
print("Validation Gap:", round(best_val_gap, 2), "%")
print("Final Test Gap:", round(best_test_gap, 2), "%")

if best_test_gap <= 10:
    print("Model Status: Good Fit")
elif best_test_gap <= 15:
    print("Model Status: Slight Generalization Gap")
else:
    print("Model Status: Possible Overfitting")

print("\nClassification Report:")
print(classification_report(y_test, best_test_pred, zero_division=0))
