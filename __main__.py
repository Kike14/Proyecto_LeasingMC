import sys
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, ConfusionMatrixDisplay, accuracy_score, precision_score
)
from modules.logistic_regression import Lg
from modules.lg_powered import Lg_powered

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

def plot_metrics_grid(log_powered_model: Lg_powered):
    # Fix: Change the figure's subplot to (1, 2) for two plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Logistic Regression (Lg_powered)
    lgp_model, lgp_y_pred, lgp_y_prob = log_powered_model.model()
    lgp_fpr, lgp_tpr, _ = roc_curve(log_powered_model.y_test, lgp_y_prob)
    lgp_roc_auc = auc(lgp_fpr, lgp_tpr)

    # Plot ROC Curve
    axes[0].plot(lgp_fpr, lgp_tpr, color='green', lw=2, label=f'AUC = {lgp_roc_auc:.2f}')
    axes[0].plot([0, 1], [0, 1], color='red', linestyle='--')
    axes[0].set_title('ROC Curve - Lg_powered')
    axes[0].legend(loc='lower right')
    axes[0].grid()

    # Plot Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(log_powered_model.y_test, lgp_y_pred, ax=axes[1])
    axes[1].set_title('Confusion Matrix - Lg_powered')

    plt.tight_layout()
    plt.show()

def print_metrics(log_powered_model: Lg_powered):
    # Get model predictions
    _, lgp_y_pred, _ = log_powered_model.model()

    # Calculate metrics
    lgp_accuracy = accuracy_score(log_powered_model.y_test, lgp_y_pred)
    lgp_precision = precision_score(log_powered_model.y_test, lgp_y_pred)

    # Print the results
    print(f"Lg_powered - Accuracy: {lgp_accuracy:.2f}, Precision: {lgp_precision:.2f}")

if __name__ == "__main__":
    route = "./data/data.csv"

    # Initialize the Lg_powered model
    log_powered_model = Lg_powered(route)

    # Print metrics
    print_metrics(log_powered_model)

    # Plot ROC and Confusion Matrix
    plot_metrics_grid(log_powered_model)
