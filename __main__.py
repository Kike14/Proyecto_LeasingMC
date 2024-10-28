import sys
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, ConfusionMatrixDisplay, RocCurveDisplay
)
from modules.logistic_regression import Lg
from modules.random_forest import Forest
from sklearn.ensemble import RandomForestClassifier
from modules.modelo_xgb import Xgb

# Asegura que el directorio raíz del proyecto esté en sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

def plot_metrics_grid(logistic_model: Lg, forest_model: Forest, xgb: Xgb):
    # Modelos
    lg_model, lg_y_pred = logistic_model.model()
    rf_model, rf_y_pred = forest_model.model()
    xgb_model, xgb_y_pred = xgb.model()

    # Probabilidades
    lg_y_pred_prob = lg_model.predict_proba(logistic_model.X_test)[:, 1]
    rf_y_pred_prob = rf_model.predict_proba(forest_model.X_test)[:, 1]
    xgb_y_pred_prob = xgb_model.predict_proba(xgb.X_test)[:, 1]  # Corregido aquí

    # ROC/AUC
    lg_fpr, lg_tpr, _ = roc_curve(logistic_model.y_test, lg_y_pred_prob)
    lg_roc_auc = auc(lg_fpr, lg_tpr)

    rf_fpr, rf_tpr, _ = roc_curve(forest_model.y_test, rf_y_pred_prob)
    rf_roc_auc = auc(rf_fpr, rf_tpr)

    xgb_fpr, xgb_tpr, _ = roc_curve(xgb.y_test, xgb_y_pred_prob)
    xgb_roc_auc = auc(xgb_fpr, xgb_tpr)


    fig, axes = plt.subplots(3, 2, figsize=(15, 18))

    #LR Graphics
    axes[0, 0].plot(lg_fpr, lg_tpr, color='blue', lw=2, label=f'AUC = {lg_roc_auc:.2f}')
    axes[0, 0].plot([0, 1], [0, 1], color='red', linestyle='--')
    axes[0, 0].set_title('ROC Curve for Logistic Regression')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid()
    ConfusionMatrixDisplay.from_predictions(logistic_model.y_test, lg_y_pred, ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix - Logistic Regression')

    # RF Graphics
    axes[1, 0].plot(rf_fpr, rf_tpr, color='green', lw=2, label=f'AUC = {rf_roc_auc:.2f}')
    axes[1, 0].plot([0, 1], [0, 1], color='red', linestyle='--')
    axes[1, 0].set_title('ROC Curve for Random Forest')
    axes[1, 0].legend(loc='lower right')
    axes[1, 0].grid()
    ConfusionMatrixDisplay.from_predictions(forest_model.y_test, rf_y_pred, ax=axes[1, 1])
    axes[1, 1].set_title('Confusion Matrix - Random Forest')
    
    #XGBOOST Graphics
    axes[2, 0].plot(xgb_fpr, xgb_tpr, color='orange', lw=2, label=f'AUC = {xgb_roc_auc:.2f}')
    axes[2, 0].plot([0, 1], [0, 1], color='red', linestyle='--')
    axes[2, 0].set_title('ROC Curve for XGBoost')
    axes[2, 0].legend(loc='lower right')
    axes[2, 0].grid()
    ConfusionMatrixDisplay.from_predictions(xgb.y_test, xgb_y_pred, ax=axes[2, 1])
    axes[2, 1].set_title('Confusion Matrix - XGBoost')

    # Ajustar la separación entre las gráficas
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    route = "./data/data.csv"


    log_model = Lg(route)
    forest_model = Forest(route)
    xgb_model = Xgb(route)

    log_model.metrics()
    forest_model.metrics()
    xgb_model.metrics()

    plot_metrics_grid(log_model, forest_model, xgb_model)
