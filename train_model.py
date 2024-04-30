import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report


# Function that prints accuracy, classification report
def print_scores(set_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    cr = classification_report(y_true, y_pred, output_dict=True)

    # Printing all scores
    print(f"{set_name} scores:")
    print(f"Accuracy: {accuracy:.5f}, Precision: {cr['macro avg']['precision']:.5f}, Recall: "
          f"{cr['macro avg']['recall']:.5f}, F1-score: {cr['macro avg']['f1-score']:.5f}\n")


# Function that returns a dictionary containing accuracy and classification report
def get_scores(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    cr = classification_report(y_true, y_pred, output_dict=True)
    return {'accuracy': accuracy, 'cr': cr}


# Function that calculates the average value of accuracy, precision, recall and f1-score
def calculate_average_score(scores):
    accuracies, precisions, recalls, f1_scores = [], [], [], []

    for score in scores:
        accuracies.append(score['accuracy'])
        cr = score.get('cr', None)
        if cr:
            precisions.append(cr.get('precision', 0))
            recalls.append(cr.get('recall', 0))
            f1_scores.append(cr.get('f1-score', 0))

    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1_score = np.mean(f1_scores)
    return avg_accuracy, avg_precision, avg_recall, avg_f1_score


# Function that fits and predicts a given column and a model
def fit_and_predict(column, X, y, validation_scores, test_scores, scaler, model, feature_importances):
    # Split the data into training, validation, and testing sets
    # (0.25 x 0.8 = 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42)

    # Scale the features in the training the datasets X_train, X_val and X_test
    x_train_scaled = scaler.fit_transform(X_train)
    x_val_scaled = scaler.transform(X_val)
    x_test_scaled = scaler.transform(X_test)

    # Train the model
    model.fit(x_train_scaled, y_train)

    # Make predictions on validation set
    y_val_pred = model.predict(x_val_scaled)

    print(f"\033[1mColumn '{column}':\033[0m")

    # Evaluate the model on validation set
    val_score = get_scores(y_val, y_val_pred)
    validation_scores.append(val_score)
    print_scores('Validation', y_val, y_val_pred)

    # Make predictions on test set
    y_test_pred = model.predict(x_test_scaled)

    # Evaluate the model on test set
    test_score = get_scores(y_test, y_test_pred)
    test_scores.append(test_score)
    print_scores('Test', y_test, y_test_pred)

    # Store feature importances
    feature_importances[column] = model.feature_importances_

    return validation_scores, test_scores, model, feature_importances
