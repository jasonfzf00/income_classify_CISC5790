import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt

# Prediction Files
knn_predictions_file = 'KNNPredictionTest.csv'
nb_predictions_file = 'NBPredictionTest.csv'
logistic_predictions_file = 'LogisticPredictionTest.csv'
rf_predictions_file = 'RandomForestPredictionTest.csv'
actual_labels_file = 'income_actual.csv'

# Read prediction files
knn_preds = pd.read_csv(knn_predictions_file)['KNN_predictions']
nb_preds = pd.read_csv(nb_predictions_file)['NB_predictions']
logistic_preds = pd.read_csv(logistic_predictions_file)['logistic_predictions']
rf_preds = pd.read_csv(rf_predictions_file)['RandomForest_predictions']

# Read actual labels
actual_labels = pd.read_csv(actual_labels_file)['income_actual']

# Create a list of actual labels and predictions for each model
true_labels_list = [actual_labels, actual_labels, actual_labels, actual_labels]
pred_probs_list = [knn_preds, nb_preds, logistic_preds, rf_preds]
models = ['KNN', 'Naive Bayes', 'Logistic Regression', 'Random Forest']
colors = ['darkorange', 'green', 'blue', 'purple']

plt.figure(figsize=(8, 8))

for i in range(len(models)):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=true_labels_list[i], y_score=pred_probs_list[i],
                                                     pos_label=1)
    auroc = sklearn.metrics.auc(fpr, tpr)

    # Plot ROC curve for each model
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{models[i]} (AUC = {auroc:.2f})')

# Plot the diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
