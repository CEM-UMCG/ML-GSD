#!/usr/bin/env python
# coding: utf-8§
# Initial imports
import glob
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import imblearn
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostClassifier, CatBoostError, Pool
from pdf2image import convert_from_path
from pyhtml2pdf import converter
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, auc, fbeta_score
)
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold)
from sklearn.utils import resample


def split_dataset(dataset, group_column, event_column, test_size=0.20, random_state=42):
    X = dataset.drop(event_column, axis=1)  # drop the target variable
    y = dataset[event_column]
    groups = dataset[group_column]

    sgkf = StratifiedGroupKFold(n_splits=int(np.round(1 / test_size)), shuffle=True, random_state=random_state)

    for train_idx, test_idx in sgkf.split(X, y, groups):
        train_set = dataset.iloc[train_idx]
        test_set = dataset.iloc[test_idx]
        break  # we only want the first split
    return train_set, test_set


def clean_data(data_frame):
    for column in data_frame.select_dtypes(include=['number']).columns:
        data_frame[column] = pd.to_numeric(data_frame[column], errors='coerce').round(2)
    threshold = len(data_frame.columns) * 0.6
    return data_frame.dropna(thresh=threshold)


def process_columns(data_frame):
    data_frame['GSD I'].value_counts()
    data_frame['AfnameDatum'] = pd.to_datetime(data_frame['AfnameDatum'])
    data_frame['PAID'] = pd.to_numeric(data_frame['PAID'], errors='coerce').astype(int)
    return data_frame


def deduplicate_data(data_frame):
    gsd_zero_subset = data_frame[data_frame['GSD I'] == 0].sort_values(by='AfnameDatum',
                                                                       ascending=True).drop_duplicates(subset='PAID',
                                                                                                       keep='first')
    gsd_non_zero_subset = data_frame[data_frame['GSD I'] != 0]
    return pd.concat([gsd_zero_subset, gsd_non_zero_subset])


def knn_impute(train_set, test_set):
    # Identify numeric and non-numeric columns
    numeric_cols = train_set.select_dtypes(include=[np.number]).columns
    non_numeric_cols = train_set.select_dtypes(exclude=[np.number]).columns

    # Separate the dataframes into numeric and non-numeric data
    train_set_numeric = train_set[numeric_cols]
    train_set_non_numeric = train_set[non_numeric_cols]

    test_set_numeric = test_set[numeric_cols]
    test_set_non_numeric = test_set[non_numeric_cols]

    knn_imputer = KNNImputer()

    # Replace the numeric data with the imputed data
    train_set_numeric_imputed = pd.DataFrame(knn_imputer.fit_transform(train_set_numeric),
                                             columns=numeric_cols)
    test_set_numeric_imputed = pd.DataFrame(knn_imputer.transform(test_set_numeric),
                                            columns=numeric_cols)

    # Concatenate the non-numeric data to the imputed numeric data
    train_set_imputed = pd.concat([train_set_numeric_imputed, train_set_non_numeric.reset_index(drop=True)], axis=1)
    test_set_imputed = pd.concat([test_set_numeric_imputed, test_set_non_numeric.reset_index(drop=True)], axis=1)

    return train_set_imputed, test_set_imputed


def feature_creation(train_set, test_set, ratios):
    for ratio_name in ratios:
        num_col, denom_col = ratio_name.split('/')[0], ratio_name.split('/')[1].split(' ')[0]
        train_set.loc[:, ratio_name] = train_set[num_col] / train_set[denom_col]
        test_set.loc[:, ratio_name] = test_set[num_col] / test_set[denom_col]

    numeric_cols = train_set.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        train_set.loc[:, col] = train_set.loc[:, col].replace([np.inf, -np.inf], np.nan)
        test_set.loc[:, col] = test_set.loc[:, col].replace([np.inf, -np.inf], np.nan)

    train_set.fillna(0, inplace=True)
    test_set.fillna(0, inplace=True)

    return train_set, test_set


# Function to perform hyperparameter tuning and feature selection
def tune_and_select_features(X, y, n_splits=10, trial_runs=200):
    def objective(trial):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # Hyperparameter space
        param_grid = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 4, 7),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'border_count': trial.suggest_int('border_count', 1, 255),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 30, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-9, 20, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'eval_metric': 'PRAUC',
            'auto_class_weights': 'Balanced',
            'random_state': 42,
        }

        inner_cv = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
        pr_auc_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(inner_cv.split(X, y)):
            X_train_hp, y_train_hp = X.iloc[train_idx], y.iloc[train_idx]
            X_val_hp, y_val_hp = X.iloc[val_idx], y.iloc[val_idx]

            # Drop samples of patients in val set from train set and deduplicate val set
            val_patients_hp = X_val_hp['PAID'].unique()
            X_train_hp = X_train_hp[~X_train_hp['PAID'].isin(val_patients_hp)]
            y_train_hp = y_train_hp[X_train_hp.index]
            X_val_hp = X_val_hp.sort_values('AfnameDatum').drop_duplicates('PAID', keep='first')
            y_val_hp = y_val_hp[X_val_hp.index]

            # Consistently drop columns right after defining subsets
            X_train_hp = X_train_hp.drop(columns=['PAID', 'AfnameDatum'])
            X_val_hp = X_val_hp.drop(columns=['PAID', 'AfnameDatum'])

            # After preprocessing, ensure identical columns in training and test sets
            assert set(X_train_hp.columns) == set(X_val_hp.columns), "Mismatch in train and test features"

            try:
                X_train_hp, y_train_hp = smote.fit_resample(X_train_hp, y_train_hp)
            except ValueError as e:
                print(f"Skipping SMOTE due to error: {e}")
                X_train_hp, y_train_hp = X_train_hp, y_train_hp

            train_pool_hp = Pool(X_train_hp, y_train_hp)
            val_pool_hp = Pool(X_val_hp, y_val_hp)

            model = CatBoostClassifier(**param_grid, verbose=False, early_stopping_rounds=100)

            try:
                model.fit(train_pool_hp, eval_set=val_pool_hp, verbose=False)
                pred_proba = model.predict_proba(val_pool_hp)[:, 1]
                pr_auc = average_precision_score(y_val_hp, pred_proba)
                pr_auc_scores.append(pr_auc)

                # Report intermediate objective value.
                trial.report(np.mean(pr_auc_scores), fold_idx)

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            except CatBoostError as e:
                print(f"Training failed for trial due to an error: {e}")
                return -1  # Return a default low score to indicate failure

        return np.mean(pr_auc_scores)

    param_grid_start = {
        'iterations': 500,
        'depth': 5,
        'bagging_temperature': 0.5,
        'border_count': 128,
        'learning_rate': 0.1,
        'l2_leaf_reg': 3,
        'random_strength': 10,
        'subsample': 0.8,
        'eval_metric': 'PRAUC',
        'auto_class_weights': 'Balanced',
        'random_state': 42}

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.enqueue_trial(param_grid_start)
    study.optimize(objective, n_trials=trial_runs)

    best_params = study.best_trial.params

    # Final model training using the entire dataset
    X = X.drop(columns=['PAID', 'AfnameDatum'])
    try:
        X, y = smote.fit_resample(X, y)
    except ValueError as e:
        print(f"Skipping SMOTE due to error: {e}")
        X, y = X, y
    full_pool_hp = Pool(X, y)
    best_model = CatBoostClassifier(**best_params, verbose=False)
    best_model.fit(full_pool_hp)

    # Extracting feature importances
    feature_importances = best_model.get_feature_importance()
    selected_features = X.columns[feature_importances > np.percentile(feature_importances, 30)]

    return best_params, feature_importances, selected_features


# Main nested cross-validation function
def nested_cv(dataset, n_splits=10, trial_runs=200):
    outer_cv = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    metrics = defaultdict(list)
    all_true_labels = []
    all_predicted_probs = []

    for train_idx, val_idx in outer_cv.split(X=dataset, y=dataset['GSD I'], groups=dataset['PAID']):
        train_set_cv, val_set_cv = dataset.iloc[train_idx], dataset.iloc[val_idx]

        # Deduplicate the validation set
        val_set_cv = val_set_cv.sort_values(by='AfnameDatum')
        val_set_cv = val_set_cv.drop_duplicates(subset='PAID', keep='first')

        # Separate features and labels, make sure to drop 'PAID' here
        X_train_cv = train_set_cv.drop(['GSD I'], axis=1)
        y_train_cv = train_set_cv['GSD I']
        X_val_cv = val_set_cv.drop(['GSD I'], axis=1)
        y_val_cv = val_set_cv['GSD I']

        # Drop samples of patients in validation set from train set
        val_patients_cv = X_val_cv['PAID'].unique()
        X_train_cv = X_train_cv[~X_train_cv['PAID'].isin(val_patients_cv)]
        y_train_cv = y_train_cv[X_train_cv.index]

        # Get best params and feature importances
        best_params, feature_importances, selected_features = tune_and_select_features(X_train_cv, y_train_cv,
                                                                                       trial_runs=trial_runs)
        aggregate_params.append(best_params)
        aggregate_feature_importances.append(feature_importances)

        X_train_cv = X_train_cv.drop(columns=['PAID', 'AfnameDatum'])
        X_val_cv = X_val_cv.drop(columns=['PAID', 'AfnameDatum'])

        # Use the selected features for the outer training and testing
        X_train_cv = X_train_cv[selected_features]
        X_val_cv = X_val_cv[selected_features]

        # Ensure no leakage (since 'PAID' is not in X_train_cv or X_test_cv, this check can be simplified)
        assert 'PAID' not in X_train_cv.columns, "PAID column should not be present in training features"

        try:
            X_train_cv, y_train_cv = smote.fit_resample(X_train_cv, y_train_cv)
        except ValueError as e:
            print(f"Skipping SMOTE due to error: {e}")
            X_train_cv, y_train_cv = X_train_cv, y_train_cv

        train_pool_cv = Pool(X_train_cv, y_train_cv)
        val_pool_cv = Pool(X_val_cv, y_val_cv)

        model = CatBoostClassifier(**best_params, verbose=False)
        model.fit(train_pool_cv)

        threshold = 0.5
        beta = 4
        predicted_probs = model.predict_proba(val_pool_cv)[:, 1]
        predictions = (predicted_probs >= threshold).astype(int)
        metrics['roc_auc'].append(roc_auc_score(y_val_cv, predicted_probs))
        print(roc_auc_score(y_val_cv, predicted_probs))
        metrics['pr_auc'].append(average_precision_score(y_val_cv, predicted_probs))
        print(average_precision_score(y_val_cv, predicted_probs))
        f_score = fbeta_score(y_val_cv.astype(int), predictions, beta=beta)
        metrics['F4_score'].append(f_score)
        print(f_score)
        all_true_labels.extend(y_val_cv.tolist())
        all_predicted_probs.extend(predicted_probs.tolist())

    return metrics, aggregate_params, aggregate_feature_importances, all_true_labels, all_predicted_probs


def calculate_stats(metrics):
    calculated_stats = {}
    for metric, scores in metrics.items():
        scores_array = np.array(scores)
        mean_score = np.mean(scores_array)
        std_dev = np.std(scores_array, ddof=1)
        sem = std_dev / np.sqrt(len(scores_array))
        alpha = 0.05
        df = len(scores_array) - 1
        t_critical = stats.t.ppf(1 - alpha / 2, df)
        margin_of_error = t_critical * sem
        ci_lower, ci_upper = mean_score - margin_of_error, mean_score + margin_of_error
        ci_lower, ci_upper = max(0, ci_lower), min(1, ci_upper)
        calculated_stats[metric] = {
            'mean': mean_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
        print(f"{metric}: Mean = {mean_score:.3f}, Std = {std_dev:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    return calculated_stats


def calculate_best_params(aggregate_params):
    best_params = {}
    for key in aggregate_params[0]:
        if key in ['iterations', 'depth', 'border_count']:
            best_params[key] = int(round(np.mean([d[key] for d in aggregate_params])))
        else:
            best_params[key] = np.mean([d[key] for d in aggregate_params])
    return best_params


def identify_important_features(aggregate_feature_importances, train_set):
    aggregate_feature_importances_array = np.array(aggregate_feature_importances)
    mean_feature_importances = np.mean(aggregate_feature_importances_array, axis=0)
    feature_threshold = np.percentile(aggregate_feature_importances, 30)

    selected_features = [
        feature for feature, importance in
        zip(train_set.drop(columns=['PAID', 'AfnameDatum', 'GSD I']).columns, mean_feature_importances)
        if importance > feature_threshold
    ]
    return selected_features


def train_final_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                      best_params: dict, smote) -> Tuple[
    CatBoostClassifier, np.array, np.array]:

    # SMOTE oversampling
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Build pools
    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)

    # Model building
    best_cb = CatBoostClassifier(**best_params, verbose=False)
    best_cb.fit(train_pool)

    # Model prediction
    y_pred_proba = best_cb.predict_proba(test_pool)[:, 1]
    y_pred = best_cb.predict(test_pool)

    return best_cb, y_pred_proba, y_pred


def stratified_bootstrap_metrics(true_labels, predicted_probs, n_bootstraps=10000, alpha=0.05, beta=4.0):
    true_labels = np.array(true_labels).astype(int)
    predicted_probs = np.array(predicted_probs)
    roc_aucs, pr_aucs, f4_scores = [], [], []
    pos_mask = true_labels == 1
    neg_mask = true_labels == 0
    for _ in range(n_bootstraps):
        pos_indices = resample(np.where(pos_mask)[0], replace=True)
        neg_indices = resample(np.where(neg_mask)[0], replace=True)
        indices = np.concatenate([pos_indices, neg_indices])
        bootstrap_true_labels = true_labels[indices]
        bootstrap_predicted_probs = predicted_probs[indices]
        # ROC curve
        fpr, tpr, _ = roc_curve(bootstrap_true_labels, bootstrap_predicted_probs)
        roc_aucs.append(auc(fpr, tpr))
        # PR curve
        precision, recall, _ = precision_recall_curve(bootstrap_true_labels, bootstrap_predicted_probs)
        # Sort by recall
        sort_idx = np.argsort(recall)
        pr_aucs.append(auc(recall[sort_idx], precision[sort_idx]))
        predicted_labels = (bootstrap_predicted_probs > 0.5).astype(int)
        f4_scores.append(fbeta_score(bootstrap_true_labels, predicted_labels, beta=4))

    # Calculate the AUC mean and Confidence intervals, F4 mean
    lower = max(0.0, 100 * (alpha / 2))
    upper = min(100.0, 100 * (1 - alpha / 2))
    if len(roc_aucs) > 0 and len(pr_aucs) > 0 and len(f4_scores) > 0:
        return {
            "roc_auc": {
                "mean": np.mean(roc_aucs),
                "lower": np.percentile(roc_aucs, lower),
                "upper": np.percentile(roc_aucs, upper)
            },
            "pr_auc": {
                "mean": np.mean(pr_aucs),
                "lower": np.percentile(pr_aucs, lower),
                "upper": np.percentile(pr_aucs, upper)
            },
            "f4_score": {
                "mean": np.mean(f4_scores),
                "lower": np.percentile(f4_scores, lower),
                "upper": np.percentile(f4_scores, upper)
            }
        }
    else:
        return "Not enough data to calculate metrics"


def calculate_precision_recall(all_true_labels, all_predicted_probs):
    precision, recall, thresholds = precision_recall_curve(all_true_labels, all_predicted_probs)
    thresholds = np.append(thresholds, 1)
    return precision, recall, thresholds


def calculate_f_beta_scores(precision, recall, beta):
    f_beta_scores = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
    return np.nan_to_num(f_beta_scores)


def find_optimal_threshold(f_beta_scores, thresholds):
    optimal_threshold_index = np.argmax(f_beta_scores)
    return thresholds[optimal_threshold_index], optimal_threshold_index


def calculate_weighted_fbeta_scores(all_true_labels, all_predicted_probs, beta, thresholds):
    weighted_fbeta_scores = []
    for thresh in thresholds:
        y_pred_thresh = (all_predicted_probs >= thresh).astype(int)
        weighted_fbeta = fbeta_score(all_true_labels, y_pred_thresh, beta=beta)
        weighted_fbeta_scores.append(weighted_fbeta)
    return weighted_fbeta_scores


def plot_f_beta_scores(thresholds, weighted_fbeta_scores, optimal_threshold, optimal_f_beta, title):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, weighted_fbeta_scores, label='F4 Score')
    plt.scatter(optimal_threshold, optimal_f_beta, color='red', label='Optimal Threshold')
    plt.title(f'F4 Score by Threshold ({title})')
    plt.xlabel('Threshold')
    plt.ylabel('F4 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'optimal_threshold_F4_{title}.pdf', format='pdf')
    plt.savefig(f'optimal_threshold_F4_{title}.tiff', format='tiff', dpi=300)
    plt.savefig(f'optimal_threshold_F4_{title}.eps', format='eps')
    plt.show()


def create_confusion_matrix(all_true_labels, all_predicted_probs, optimal_threshold):
    y_pred_optimal = (all_predicted_probs >= optimal_threshold).astype(int)
    return confusion_matrix(all_true_labels, y_pred_optimal)


def plot_confusion_matrix(conf_matrix, title):
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    total_samples = np.sum(conf_matrix)
    percentages_row = conf_matrix / row_sums * 100
    labels_row = (np.asarray(["{0}\n({1:.2f}%)".format(count, percentage)
                              for count, percentage in zip(conf_matrix.flatten(), percentages_row.flatten())])
                  .reshape(2, 2))

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.4)
    sns.heatmap(percentages_row, annot=labels_row, fmt='', cmap="Blues",
                cbar_kws={'label': 'Percentage of Total per Class'}, annot_kws={"size": 16})
    plt.title(f"Confusion Matrix at optimal F4 score ({title})", fontsize=20)
    plt.ylabel('Actual GSD Ia', fontsize=16)
    plt.xlabel('Predicted GSD Ia', fontsize=16)
    plt.tick_params(labelsize=14)
    plt.savefig(f'confusion_matrix_optimal_threshold_F4_{title}.pdf', format='pdf')
    plt.savefig(f'confusion_matrix_optimal_threshold_F4_{title}.tiff', format='tiff', dpi=300)
    plt.savefig(f'confusion_matrix_optimal_threshold_F4_{title}.eps', format='eps')
    plt.show()


def calculate_optimal_threshold(all_true_labels, all_predicted_probs, beta, title):
    precision, recall, thresholds = calculate_precision_recall(all_true_labels, all_predicted_probs)
    f_beta_scores = calculate_f_beta_scores(precision, recall, beta)
    optimal_threshold, optimal_threshold_index = find_optimal_threshold(f_beta_scores, thresholds)

    optimal_precision = precision[optimal_threshold_index]
    optimal_recall = recall[optimal_threshold_index]
    optimal_f_beta = f_beta_scores[optimal_threshold_index]

    thresholds = np.linspace(0, 1, 101)
    weighted_fbeta_scores = calculate_weighted_fbeta_scores(all_true_labels, all_predicted_probs, beta, thresholds)

    print(f"Optimal Threshold {title}:", optimal_threshold)
    print(f"Optimal F{beta} Score {title}:", optimal_f_beta)
    print(f"Corresponding Precision {title}:", optimal_precision)
    print(f"Corresponding Recall {title}:", optimal_recall)

    plot_f_beta_scores(thresholds, weighted_fbeta_scores, optimal_threshold, optimal_f_beta, title)

    conf_matrix = create_confusion_matrix(all_true_labels, all_predicted_probs, optimal_threshold)
    plot_confusion_matrix(conf_matrix, title)

    y_pred_optimal = (all_predicted_probs >= optimal_threshold).astype(int)

    return optimal_threshold, y_pred_optimal


def plot_roc_pr_curves(y_test, y_pred_proba, roc_auc_mean, roc_auc_ci, pr_auc_mean, pr_auc_ci, positive_rate, title):
    plt.figure(figsize=(14, 7))

    # Calculate ROC and Precision-Recall values
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    sns.set(font_scale=1.4)

    # Plotting ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_mean:.3f}, 95% CI: {roc_auc_ci})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title(f'Receiver Operating Characteristic Curve {title}', fontsize=20)
    plt.legend(loc="lower right", fontsize=14)
    plt.tick_params(labelsize=14)

    # Plotting Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='steelblue', lw=2,
             label=f'PR curve (area = {pr_auc_mean:.3f}, 95% CI: {pr_auc_ci})')
    plt.plot([0, 1], [positive_rate, positive_rate], color='red', linestyle='--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.title(f'Precision-Recall Curve {title}', fontsize=20)
    plt.legend(loc="lower left", fontsize=14)
    plt.tick_params(labelsize=14)

    plt.tight_layout()
    plt.savefig(f'ROC_PR_curve_{title}.pdf', format='pdf')
    plt.savefig(f'ROC_PR_curve_{title}.tiff', format='tiff', dpi=300)
    plt.savefig(f'ROC_PR_curve_{title}.eps', format='eps')
    plt.show()


def plot_feature_importance(model, X_train, num_features=20):
    # Get feature importances
    feature_importances = model.feature_importances_

    # Match feature names with their importances
    features = X_train.columns

    if len(features) != len(feature_importances):
        print("Warning: Length of 'features' and 'feature_importance' do not match.")
        print('Number of features: ', len(features))
        print('Number of importances: ', len(feature_importances))
        return

    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

    # Sort by importance and select top n features
    top_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(num_features)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {num_features} Features')
    plt.gca().invert_yaxis()
    plt.show()


def visualize_instance(explainer, shap_values, X_test, sample_index, show_plot=False):
    shap.initjs()

    if np.isscalar(explainer.expected_value):
        expected_value = explainer.expected_value
    else:
        # In case of multi-class adjust this index
        expected_value = explainer.expected_value[1]

    # Generate a plot
    plot = shap.force_plot(
        expected_value,
        shap_values[sample_index],
        X_test.iloc[sample_index, :].round(2),
        link="logit",
        show=show_plot
    )

    # If the plot should not be displayed, it will be saved
    if not show_plot:
        filename = 'force_plot_{}.html'.format(sample_index)  # Unique filename for each sample index
        with open(filename, 'w') as f:
            shap.save_html(f, plot)


# Using seaborn style
sns.set(style="whitegrid")

# Data loading and preprocessing
df = pd.read_excel(
    "/path/to/data/file")
df = clean_data(df)
df = process_columns(df)
df = deduplicate_data(df)

train_set, test_set = split_dataset(df, group_column='PAID', event_column='GSD I', test_size=0.20, random_state=42)
train_set = train_set.drop(['LBM_Diagnose', 'Age', 'Monster', 'C6DC'], axis=1)
test_set = test_set.drop(['LBM_Diagnose', 'Age', 'Monster', 'C6DC'], axis=1)

train_set, test_set = knn_impute(train_set, test_set)

train_set.loc[:, 'C0/(C16+C18) ratio'] = train_set['C0'] / (train_set['C16'] + train_set['C18'])
test_set.loc[:, 'C0/(C16+C18) ratio'] = test_set['C0'] / (test_set['C16'] + test_set['C18'])

train_set.loc[:, '(C16+C18:1)/C2 ratio'] = (train_set['C16'] + train_set['C18:1']) / train_set['C2']
test_set.loc[:, '(C16+C18:1)/C2 ratio'] = (test_set['C16'] + test_set['C18:1']) / test_set['C2']

train_set.loc[:, 'C16/(C10+C12) ratio'] = train_set['C16'] / (train_set['C10'] + train_set['C12'])
test_set.loc[:, 'C16/(C10+C12) ratio'] = test_set['C16'] / (test_set['C10'] + test_set['C12'])

train_set.loc[:, 'C16/(C14+C14:1) ratio'] = train_set['C16'] / (train_set['C14'] + train_set['C14:1'])
test_set.loc[:, 'C16/(C14+C14:1) ratio'] = test_set['C16'] / (test_set['C14'] + test_set['C14:1'])

ratios = ['C3/C2 ratio',
          'C3/C16 ratio',
          'C5/C0 ratio',
          'C8/C2 ratio',
          'C4/C2',
          'C4/C0',
          'C8/C10 ratio',
          'C5DC/C8 ratio',
          'C5DC/C16 ratio',
          'C14:1/C12:1 ratio',
          'C16OH/C16 ratio',
          'C14/C3 ratio',
          'C16/C2 ratio',
          'C16/C3 ratio',
          'C18/C3 ratio',
          'C16:1/C3 ratio',
          'C2/C0 ratio',
          'C14:1/C2 ratio',
          'C16/C14:1 ratio',
          'C16:1/C16 ratio',
          'C16/C10 ratio',
          'C16:1/C14:1 ratio',
          'C14/C12 ratio',
          'C16:1/C14 ratio',
          'C16/C2 ratio',
          'C3/C0 ratio',
          'C16/C0 ratio',
          'C3DC+C4OH/C0 ratio',
          'C3DC+C4OH/C10 ratio',
          'C4DC+C5OH/C0 ratio',
          'C4DC+C5OH/C8 ratio']

train_set, test_set = feature_creation(train_set, test_set, ratios)

# Drop duplicates based on 'Pat. nr', keep the first (oldest) entry
test_set['AfnameDatum'] = pd.to_datetime(test_set['AfnameDatum'])
test_set = test_set.sort_values(by='AfnameDatum')
test_set = test_set.drop_duplicates(subset='PAID', keep='first')
test_set = test_set.drop(['AfnameDatum'], axis=1)

# Remove the duplicates of non-GSD Ia patients from train_set, keeping oldest
train_set.sort_values(by='AfnameDatum', inplace=True)
condition = train_set['GSD I'] == 0
train_set = train_set.loc[~condition]._append(train_set.loc[condition].drop_duplicates(subset='PAID'))

# Separate features and targets
X_train = train_set.drop('GSD I', axis=1)
y_train = train_set['GSD I']
X_test = test_set.drop('GSD I', axis=1)
y_test = test_set['GSD I']

# Extract unique PAID values from both training and test sets
unique_paid_train = set(X_train['PAID'].unique())
unique_paid_test = set(X_test['PAID'].unique())

# Check if there is any intersection between the two sets of PAID values
common_paid = unique_paid_train.intersection(unique_paid_test)

# Print the result
if len(common_paid) == 0:
    print("No matching PAID numbers between the training and test sets.")
else:
    print(f"Matching PAID numbers found in both sets: {common_paid}")

# Initialize SMOTE
smote = imblearn.over_sampling.SVMSMOTE(sampling_strategy='minority', random_state=42)

# Initialize Containers for Storing Aggregate Parameters and Feature Importances
aggregate_params = []
aggregate_feature_importances = []

# Run nested cross-validation
metrics, aggregate_params, aggregate_feature_importances, all_true_labels, all_predicted_probs = nested_cv(train_set, trial_runs=200)

# Calculate performance of nested cross-validation
stats = calculate_stats(metrics)
print(stats)
roc_auc_mean = stats['roc_auc']['mean']
roc_auc_ci = f"[{stats['roc_auc']['ci_lower']:.3f}, {stats['roc_auc']['ci_upper']:.3f}]"
pr_auc_mean = stats['pr_auc']['mean']
pr_auc_ci = f"[{stats['pr_auc']['ci_lower']:.3f}, {stats['pr_auc']['ci_upper']:.3f}]"
positive_rate = np.mean(all_true_labels)
title = "Nested CV"
plot_roc_pr_curves(all_true_labels, all_predicted_probs, roc_auc_mean, roc_auc_ci, pr_auc_mean, pr_auc_ci,
                   positive_rate, title="Nested CV")
calculate_optimal_threshold(all_true_labels, all_predicted_probs, beta=4, title='Nested CV')


# Selecting best hyperparameters
best_params = calculate_best_params(aggregate_params)
print("Best Parameters:", best_params)

# Selecting best features
selected_features = identify_important_features(aggregate_feature_importances, train_set)
print("Selected Features:", selected_features)

# Drop unused columns
PAID_test_df = X_test[['PAID']].copy()
X_train = X_train.drop(columns=['PAID', 'AfnameDatum'])
X_test = X_test.drop(columns=['PAID'])

# Use selected features from nested cross-validation
X_train = X_train[selected_features]
X_test = X_test[selected_features]

# Train final model and test on held-out test set
best_cb, y_pred_proba, y_pred = train_final_model(X_train, X_test, y_train, y_test,
                                                  best_params, smote)

# Bootstrapping to calculate performance on test set with confidence intervals
results = stratified_bootstrap_metrics(y_test, y_pred_proba, n_bootstraps=10000, beta=4)

roc_auc_mean = results['roc_auc']['mean']
roc_auc_ci_lower = results['roc_auc']['lower']
roc_auc_ci_upper = results['roc_auc']['upper']
pr_auc_mean = results['pr_auc']['mean']
pr_auc_ci_lower = results['pr_auc']['lower']
pr_auc_ci_upper = results['pr_auc']['upper']
f4_mean = results['f4_score']['mean']
f4_lower = results['f4_score']['lower']
f4_upper = results['f4_score']['upper']

roc_auc_ci = f'[{roc_auc_ci_lower:.3f}, {roc_auc_ci_upper:.3f}]'
pr_auc_ci = f'[{pr_auc_ci_lower:.3f}, {pr_auc_ci_upper:.3f}]'
f4_ci = f'[{f4_lower:.3f}, {f4_upper:.3f}]'
positive_rate = np.mean(y_test)
title = "Test Set"

# Print the results
print(f"ROC AUC Mean of final model on test set: {roc_auc_mean:.3f}, 95% CI: {roc_auc_ci}")
print(f"PR AUC Mean of final model on test set: {pr_auc_mean:.3f}, 95% CI: {pr_auc_ci}")
print(f"F4 score of final model on test set: {f4_mean:.3f}, 95% CI: {f4_ci}")

# Plot ROC and PR curves
plot_roc_pr_curves(y_test, y_pred_proba, roc_auc_mean, roc_auc_ci, pr_auc_mean, pr_auc_ci, positive_rate, title="Test set")

# Plot feature importances
plot_feature_importance(best_cb, X_train)

# Determine optimal threshold where F4 score is highest
optimal_threshold, y_pred_optimal = calculate_optimal_threshold(y_test, y_pred_proba, beta=4, title=title)

# Determine which samples were FN and FP
y_test = y_test.reindex(X_test.index)
FP_rows = np.where((y_test.values == 0) & (y_pred_optimal == 1))
FN_rows = np.where((y_test.values == 1) & (y_pred_optimal == 0))
TP_rows = np.where((y_test.values == 1) & (y_pred_optimal == 1))

FP_test = X_test.iloc[FP_rows]
FN_test = X_test.iloc[FN_rows]
TP_test = X_test.iloc[TP_rows]

FP_PAIDs = FP_test.index.tolist()
FN_PAIDs = FN_test.index.tolist()
TP_PAIDs = TP_test.index.tolist()
FP_FN_PAIDs = FP_PAIDs + FN_PAIDs + TP_PAIDs

# Print PAID values
print("False Positives PAIDs: ", FP_test)
print("False Negatives PAIDs: ", FN_test)
print("True Positives PAIDs: ", TP_test)

# Calculate SHAP values and make plots
explainer = shap.Explainer(best_cb)
shap_values = explainer.shap_values(X_test)

shap_values_for_positive_class = shap_values[1]

# Initialize JavaScript visualization
shap.initjs()

# SHAP force plot for FN and FP instances
for paid in FP_FN_PAIDs:
    position_index = X_test.index.get_loc(paid)
    visualize_instance(explainer, shap_values, X_test, position_index, show_plot=False)

html_files = glob.glob("/Users/joostgroen/Documents/GSD machine learning/PAPER/CATBOOST/CATBOOST SMOTE 70% FS/*.html")

for html_file in html_files:
    html_file_url = Path(html_file).absolute().as_uri()

    pdf_file = html_file.replace('.html', '.pdf')

    try:
        converter.convert(html_file_url, pdf_file)
        images = convert_from_path(pdf_file)
        tiff_file = pdf_file.replace('.pdf', '.tiff')
        eps_file = pdf_file.replace('.pdf', '.eps')
        images[0].save(tiff_file, 'TIFF')
        images[1].save(eps_file, 'EPS')
    except Exception as e:
        print(f"Failed to convert {html_file} to PDF/TIFF: {e}")

# Create SHAP summary plot
shap.summary_plot(shap_values, X_test, show=False, max_display=15)
fig = plt.gcf()
fig.set_size_inches(20, 10)
plt.savefig('SHAP.pdf', format='pdf')
plt.savefig('SHAP.tiff', format='tiff', dpi=300)
plt.savefig('SHAP.eps', format='eps')
