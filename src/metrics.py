# ==========================================
# FILE: metrics.py
# ROLE: Multi-label Evaluation Metrics & Error Analysis
# ==========================================
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from typing import List, Tuple

def evaluate_goemotions(
    y_true: np.ndarray, 
    y_pred_probs: np.ndarray, 
    label_names: List[str], 
    threshold: float = 0.5
) -> Tuple[float, float, pd.DataFrame]:
    """
    Evaluates a Multi-label classification model for the GoEmotions dataset.
    Computes Macro/Micro F1-scores and generates a False Negative report for Error Analysis.
    
    Args:
        y_true (np.ndarray): Ground truth multi-hot encoded matrix (Shape: [n_samples, n_classes]).
        y_pred_probs (np.ndarray): Predicted probabilities from Sigmoid (Shape: [n_samples, n_classes]).
        label_names (List[str]): List of the emotion class names.
        threshold (float): Probability threshold to convert to binary predictions.
        
    Returns:
        Tuple containing:
            - macro_f1 (float): Macro-averaged F1 score.
            - micro_f1 (float): Micro-averaged F1 score.
            - top_5_hardest (pd.DataFrame): DataFrame containing the top 5 labels with the highest FN rates.
    """
    # 1. Convert probabilities to binary predictions based on the threshold
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    # 2. Calculate CORE METRICS: Macro and Micro F1-scores
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    print("==================================================")
    print("OVERALL PERFORMANCE REPORT (CORE METRICS)") 
    print("==================================================")
    print(f"Macro F1-score : {macro_f1:.4f} (Primary Metric)")
    print(f"Micro F1-score : {micro_f1:.4f}\n")
    
    # 3. Calculate Multi-label Confusion Matrix for ERROR ANALYSIS
    # Returns an array of 2x2 confusion matrices for each class: [[TN, FP], [FN, TP]]
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    
    fn_analysis = []
    
    for i, label in enumerate(label_names):
        # Flatten the 2x2 matrix to easily extract values
        tn, fp, fn, tp = mcm[i].ravel()
        
        # Calculate actual positives (Total times the emotion actually appeared)
        actual_positives = fn + tp
        
        # Calculate False Negative Rate (Avoid division by zero for completely missing classes)
        fn_rate = (fn / actual_positives) if actual_positives > 0 else 0.0
        
        fn_analysis.append({
            'Emotion': label,
            'FN Rate': f"{fn_rate * 100:.2f}%",
            'False Negatives (FN)': int(fn),
            'Actual Positives': int(actual_positives)
        })
        
    # 4. Process the Error DataFrame
    df_errors = pd.DataFrame(fn_analysis)
    
    # Create a hidden sorting key based on the float value of FN Rate
    df_errors['Sort_Key'] = df_errors['FN Rate'].str.rstrip('%').astype(float)
    
    # Sort descending by FN Rate, then by absolute FN count
    df_errors = df_errors.sort_values(by=['Sort_Key', 'False Negatives (FN)'], ascending=[False, False])
    df_errors = df_errors.drop(columns=['Sort_Key'])
    
    # Extract the top 5 most difficult labels
    top_5_hardest = df_errors.head(5)
    
    print("==================================================")
    print("ERROR ANALYSIS REPORT - FALSE NEGATIVES (FOR PHUONG)")
    print("==================================================")
    print("Description: Top 5 emotions the model is most likely to miss (Predicted 0, Actual 1).")
    print("Action item: Consider reviewing class_weights or augmenting data for these labels.\n")
    
    # Print the DataFrame without index for a cleaner copy-paste format
    print(top_5_hardest.to_string(index=False))
    print("\n==================================================")
    
    return macro_f1, micro_f1, top_5_hardest  