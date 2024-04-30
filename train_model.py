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


# Function that returns a dictionary containing accuracy, precision, recall and f1-score
def get_scores(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    # Classification report
    cr = classification_report(y_true, y_pred, output_dict=True)
    precision = cr['macro avg']['precision']
    recall = cr['macro avg']['recall']
    f1_score = cr['macro avg']['f1-score']

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score}


# Function that calculates the average values of accuracy, precision, recall and f1-score
def calculate_average_score(scores):
    accuracies, precisions, recalls, f1_scores = [], [], [], []

    for score in scores:
        accuracies.append(score['accuracy'])
        precisions.append(score['precision'])
        recalls.append(score['recall'])
        f1_scores.append(score['f1_score'])

    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1_score = np.mean(f1_scores)

    return {'accuracy': avg_accuracy,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1_score}


# Function that prints the average values of accuracy, precision, recall and f1-score
def print_average_score(set_name, scores):
    print(f"Average {set_name} Scores:")
    print(f"Accuracy: {scores['accuracy']:.5f}")
    print(f"Precision: {scores['precision']:.5f}")
    print(f"Recall: {scores['recall']:.5f}")
    print(f"F1-score: {scores['f1_score']:.5f}\n")


# Function that returns features after scaling in the datasets X_train, X_val and X_test
def scale_sets(scaler, X_train, X_val, X_test):
    if scaler != None:
        return scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test)
    else:
        return X_train, X_val, X_test


# Function that fits and predicts a given column and a model
def fit_and_predict(column, X, y, validation_scores, test_scores, model, feature_importances, scaler=None):
    # Split the data into training, validation, and testing sets
    # (0.25 x 0.8 = 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # Scale the features in the training the datasets X_train, X_val and X_test
    x_train_scaled, x_val_scaled, x_test_scaled = scale_sets(
        scaler, X_train, X_val, X_test)

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
