from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report


# Function that fits and predicts a given column and a model
def fit_and_predict(column, X, y, validation_accuracies, test_accuracies, scaler, model, feature_importances):
    # Split the data into training, validation, and testing sets
    # (0.25 x 0.8 = 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # Scale the features in the training the datasets X_train, X_val and X_test
    x_train_scaled = scaler.fit_transform(X_train)
    x_val_scaled = scaler.transform(X_val)
    x_test_scaled = scaler.transform(X_test)

    # Train the model
    model.fit(x_train_scaled, y_train)

    # Make predictions on validation set
    y_val_pred = model.predict(x_val_scaled)

    # Evaluate the model on validation set
    val_accuracy = accuracy_score(y_val, y_val_pred)
    validation_accuracies.append(val_accuracy)
    print(f"Validation Accuracy for {column}: {val_accuracy}")
    # print(f"{classification_report(y_val_pred, y_val)}\n")

    # Make predictions on test set
    y_test_pred = model.predict(x_test_scaled)

    # Evaluate the model on test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_accuracies.append(test_accuracy)
    print(f"Test Accuracy for {column}: {test_accuracy}\n")
    # print(f"{classification_report(y_test_pred, y_test)}\n")

    # Store feature importances
    feature_importances[column] = model.feature_importances_
    
    return validation_accuracies, test_accuracies, model, feature_importances
