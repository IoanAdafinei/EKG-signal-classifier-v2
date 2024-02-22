import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve
from prettytable import PrettyTable
import numpy as np
import time

start = time.time()

# Load the EKG signals from the CSV file
df = pd.read_csv('./dataset/merge-csv.csv')

# Separate the features and labels
features = df.iloc[:, 0:186].values
labels = df.iloc[:, -1].values


# Apply feature selection to identify the most relevant features
selector = SelectKBest(f_classif, k='all')

features = selector.fit_transform(features, labels)

# Split the data into training, validation, and testing sets
X_train, X_val_test, y_train, y_val_test = train_test_split(features, labels, test_size=0.3, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Choose a suitable machine learning algorithm for binary classification
knn_classifier = KNeighborsClassifier(n_neighbors=1)

# Training the  model on the training set
knn_classifier.fit(X_train, y_train)

# Validation zone
# Evaluate the trained model's accuracy, precision, recall, and F1-score on the testing set
y_pred_knn = knn_classifier.predict(X_validation)
accuracy_knn = accuracy_score(y_validation, y_pred_knn)
precision_knn, recall_knn, f1_knn, _ = precision_recall_fscore_support(y_validation, y_pred_knn, average='macro')

precision_knn1 = precision_knn
recall_knn1 = recall_knn
# print('KNN Predictions:', y_pred_knn.tolist())
# print('KNN Actual Labels:', y_validation.tolist())

# Confusion Matrix
conf_matrix_knn = confusion_matrix(y_validation, y_pred_knn)
plt.figure(figsize=(16, 12))
sns.heatmap(conf_matrix_knn, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix (validation step) KNN ')
plt.show()

# Predict probabilities for the positive class
y_probs_knn = knn_classifier.predict_proba(X_validation)[:, 1]

# Compute ROC curve and AUC score
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_validation, y_probs_knn)
auc_knn = roc_auc_score(y_validation, y_probs_knn)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, label=f'AUC = {auc_knn:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel('Specificity')
plt.ylabel('Sensibility')
plt.title('Receiver Operating Characteristic (ROC) Curve  (validation step) KNN ')
plt.legend()
plt.show()

# Learning curve
train_sizes_knn, train_scores_knn, test_scores_knn = learning_curve(
    knn_classifier, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

train_mean_knn = np.mean(train_scores_knn, axis=1)
test_mean_knn = np.mean(test_scores_knn, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes_knn, train_mean_knn, label='Training score')
plt.plot(train_sizes_knn, test_mean_knn, label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curve  (validation step) KNN ')
plt.legend()
plt.show()


# Recall curve
precision_knn, recall_knn, _ = precision_recall_curve(y_validation, y_probs_knn)

plt.figure(figsize=(8, 6))
plt.plot(recall_knn, precision_knn)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve  (validation step) KNN ')
plt.show()


# Plot to visualize predictions compared to actual labels
plt.figure(figsize=(20, 10))
plt.scatter(np.arange(len(y_validation)), y_validation, color='blue', label='Actual Labels')
plt.scatter(np.arange(len(y_validation)), y_pred_knn, color='red', label='Predictions')
plt.title('Model Predictions vs Actual Labels (validation step) KNN ')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.legend()
plt.show()

# Test zone
y_pred_test_knn = knn_classifier.predict(X_test)
test_accuracy_knn = accuracy_score(y_test, y_pred_test_knn)

# print('KNN Predictions:', y_pred_test_knn.tolist())
# print('KNN Actual Labels:', y_test.tolist())

# Confusion Matrix
test_conf_matrix_knn = confusion_matrix(y_test, y_pred_test_knn)
plt.figure(figsize=(16, 12))
sns.heatmap(test_conf_matrix_knn, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix (testing step) KNN ')
plt.show()

# Plot to visualize predictions compared to actual labels
plt.figure(figsize=(20, 10))
plt.scatter(np.arange(len(y_test)), y_test, color='blue', label='Actual Labels')
plt.scatter(np.arange(len(y_test)), y_pred_test_knn, color='red', label='Predictions')
plt.title('Model Predictions vs Actual Labels (test step) KNN ')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.legend()
plt.show()


# Naive Bayes from here

nb_classifier = GaussianNB()

# Training the  model on the training set
nb_classifier.fit(X_train, y_train)

# Validation zone
# Evaluate the trained model's accuracy, precision, recall, and F1-score on the testing set
y_pred = nb_classifier.predict(X_validation)
accuracy = accuracy_score(y_validation, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_validation, y_pred, average='macro')
precision1 = precision
recall1 = recall
# print('NB Predictions:', y_pred.tolist())
# print('NB Actual Labels:', y_validation.tolist())

# Confusion Matrix
conf_matrix = confusion_matrix(y_validation, y_pred)
plt.figure(figsize=(16, 12))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix (validation step) NB ')
plt.show()

# Predict probabilities for the positive class
y_probs = nb_classifier.predict_proba(X_validation)[:, 1]

# Compute ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_validation, y_probs)
auc = roc_auc_score(y_validation, y_probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel('Specificity')
plt.ylabel('Sensibility')
plt.title('Receiver Operating Characteristic (ROC) Curve  (validation step) NB ')
plt.legend()
plt.show()

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(
    nb_classifier, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, test_mean, label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curve  (validation step) NB ')
plt.legend()
plt.show()


# Recall curve
precision, recall, _ = precision_recall_curve(y_validation, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve  (validation step) NB ')
plt.show()


# Plot to visualize predictions compared to actual labels
plt.figure(figsize=(20, 10))
plt.scatter(np.arange(len(y_validation)), y_validation, color='blue', label='Actual Labels')
plt.scatter(np.arange(len(y_validation)), y_pred, color='red', label='Predictions')
plt.title('Model Predictions vs Actual Labels (validation step) NB ')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.legend()
plt.show()

# Test zone
y_pred_test = nb_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

# print('NB Predictions:', y_pred_test.tolist())
# print('NB Actual Labels:', y_test.tolist())

# Confusion Matrix
test_conf_matrix = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(16, 12))
sns.heatmap(test_conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix (testing step) NB ')
plt.show()

# Plot to visualize predictions compared to actual labels
plt.figure(figsize=(20, 10))
plt.scatter(np.arange(len(y_test)), y_test, color='blue', label='Actual Labels')
plt.scatter(np.arange(len(y_test)), y_pred_test, color='red', label='Predictions')
plt.title('Model Predictions vs Actual Labels (test step)NB ')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.legend()
plt.show()

table = PrettyTable()

table.field_names = ["", "KNN", "Naive Bayes"]
table.add_row(["Accuracy", accuracy_knn, accuracy])
table.add_row(["Precision", precision_knn1, precision1])
table.add_row(["Recall", recall_knn1, recall1])
table.add_row(["F1 score", f1_knn, f1])
table.add_row(["Test Accuracy", test_accuracy_knn, test_accuracy])

print(table)

end = time.time()
print('\nProgram ran for:', end - start, 'seconds\n')

print('Do you want to test with new dataset?(y/n)')
continue_value = input()
if continue_value == 'y':
    print('Do you want to use KNN or Naive Bayes?(knn/nb)')
    alg = input()


if continue_value == "y" and alg == 'knn':
    # Test with new data from a separate CSV file
    new_data = pd.read_csv(
        './dataset/new_file.csv')  # Replace 'new_file.csv' with the name of the file in the dataset folder
    # Separate the feature data and label data from the new dataset
    new_features = new_data.iloc[:, :186].values
    new_labels = new_data.iloc[:, -1].values

    # Apply the same feature selection method used in training
    new_features_selected = selector.transform(new_features)

    # Use the trained model to predict the labels for the new data
    new_predictions = knn_classifier.predict(new_features_selected)

    new_accuracy = accuracy_score(new_labels, new_predictions)
    print(f'Accuracy for KNN with new data is: {new_accuracy}')

    # Create a plot to visualize the model's predictions on the new data
    plt.figure(figsize=(20, 10))
    plt.scatter(np.arange(len(new_labels)), new_labels, color='blue', label='Actual Labels')
    plt.scatter(np.arange(len(new_labels)), new_predictions, color='red', label='Predictions')
    plt.title('Model Predictions vs Actual Labels (test from external file) with KNN')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.legend()
    plt.show()

if continue_value == "y" and alg == 'nb':
    # Test with new data from a separate CSV file
    new_data = pd.read_csv(
        './dataset/new_file.csv')  # Replace 'new_file.csv' with the name of the file in the dataset folder
    # Separate the feature data and label data from the new dataset
    new_features = new_data.iloc[:, :186].values
    new_labels = new_data.iloc[:, -1].values

    # Apply the same feature selection method used in training
    new_features_selected = selector.transform(new_features)

    # Use the trained model to predict the labels for the new data
    new_predictions = nb_classifier.predict(new_features_selected)

    new_accuracy = accuracy_score(new_labels, new_predictions)
    print(f'Accuracy for Naive Bayes with new data is: {new_accuracy}')

    # Create a plot to visualize the model's predictions on the new data
    plt.figure(figsize=(20, 10))
    plt.scatter(np.arange(len(new_labels)), new_labels, color='blue', label='Actual Labels')
    plt.scatter(np.arange(len(new_labels)), new_predictions, color='red', label='Predictions')
    plt.title('Model Predictions vs Actual Labels (test from external file) with NB')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.legend()
    plt.show()

# Show confidence scores for samples
# scores = model.decision_function(X_test)
# print(scores.tolist())
