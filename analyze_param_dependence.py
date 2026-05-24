#!/usr/bin/env python3
"""
analyze_param_dependence.py – Анализ зависимости ошибки от гиперпараметров.

Строит:
- Корреляционную матрицу числовых параметров и best_test_loss.
- Pairplot для визуального анализа парных зависимостей.
- Оценку важности признаков с помощью Random Forest.
- Графики частичной зависимости (partial dependence) для топ-параметров.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.preprocessing import LabelEncoder
from experiment_manager import manager

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_experiments_data(filter_model_type=None, filter_exercise=None):
    """Загружает данные экспериментов с best_test_loss."""
    experiments = manager.list_experiments(status='completed')
    data = []
    for exp in experiments:
        if filter_model_type and exp['model_type'] != filter_model_type:
            continue
        if filter_exercise and exp['exercise'] != filter_exercise:
            continue
        if 'best_test_loss' not in exp.get('result', {}):
            continue

        entry = {
            'exercise': exp['exercise'],
            'model_type': exp['model_type'],
            'best_test_loss': exp['result']['best_test_loss'],
            'exp_id': exp['id'],
        }
        params = exp.get('params', {})
        for k, v in params.items():
            try:
                entry[k] = float(v)
            except (TypeError, ValueError):
                entry[k] = v
        data.append(entry)
    return pd.DataFrame(data)


def plot_correlation_matrix(df, numeric_cols, output_dir='analysis_plots'):
    """Строит корреляционную матрицу для числовых параметров и best_test_loss."""
    os.makedirs(output_dir, exist_ok=True)
    corr_df = df[numeric_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Parameters and Best Test Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=150)
    plt.close()
    print(f"Сохранена корреляционная матрица.")


def plot_pairplot(df, params_subset, hue='model_type', output_dir='analysis_plots'):
    """Pairplot для выбранных параметров и best_test_loss."""
    os.makedirs(output_dir, exist_ok=True)
    plot_cols = params_subset + ['best_test_loss']
    # Отбираем только числовые колонки
    plot_cols = [c for c in plot_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(plot_cols) < 2:
        print("Недостаточно числовых колонок для pairplot.")
        return

    g = sns.pairplot(df[plot_cols + [hue]], hue=hue, diag_kind='kde',
                     plot_kws={'alpha': 0.6, 's': 50})
    g.fig.suptitle('Pairplot of Parameters vs Best Test Loss', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pairplot.png'), dpi=150)
    plt.close()
    print("Сохранён pairplot.")


def feature_importance_analysis(df, feature_cols, target='best_test_loss', output_dir='analysis_plots'):
    """
    Оценивает важность признаков с помощью Random Forest.
    Строит barplot важности.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Подготовка данных: кодируем категориальные признаки
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    y = df[target]

    # Убираем строки с NaN
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    if len(X) == 0:
        print("Нет данных для анализа важности признаков.")
        return

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150)
    plt.close()
    print("Сохранён график важности признаков.")

    # Выводим топ-5
    print("\nТоп важных признаков:")
    print(importance.head(10).to_string(index=False))
    return model, X.columns.tolist()


def plot_partial_dependence(model, X, features, output_dir='analysis_plots', n_cols=2):
    """Строит графики частичной зависимости для указанных признаков."""
    os.makedirs(output_dir, exist_ok=True)
    # Выбираем топ-6 признаков для отображения
    top_features = features[:6]
    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(model, X, top_features, ax=ax, n_cols=n_cols)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'partial_dependence.png'), dpi=150)
    plt.close()
    print("Сохранены графики частичной зависимости.")


def main():
    parser = argparse.ArgumentParser(description='Анализ зависимости ошибки от гиперпараметров')
    parser.add_argument('--model_type', choices=['rae', 'transformer', 'vae', 'analyzer'],
                        help='Фильтр по типу модели')
    parser.add_argument('--exercise', help='Фильтр по упражнению')
    parser.add_argument('--output_dir', default='analysis_plots', help='Папка для сохранения')
    parser.add_argument('--params', nargs='+',
                        default=['hidden_dim', 'latent_dim', 'seq_len', 'dropout', 'lr', 'num_layers', 'batch_size'],
                        help='Список параметров для анализа')
    args = parser.parse_args()

    df = load_experiments_data(args.model_type, args.exercise)
    if df.empty:
        print("Нет данных для анализа.")
        return

    print(f"Загружено {len(df)} экспериментов.")

    # Определяем числовые колонки из параметров
    numeric_param_cols = [p for p in args.params if p in df.columns and pd.api.types.is_numeric_dtype(df[p])]
    if not numeric_param_cols:
        print("Нет числовых параметров для анализа.")
        return

    # Корреляционная матрица
    plot_correlation_matrix(df, numeric_param_cols + ['best_test_loss'], args.output_dir)

    # Pairplot (если параметров немного)
    if len(numeric_param_cols) <= 5:
        plot_pairplot(df, numeric_param_cols, output_dir=args.output_dir)
    else:
        # Выбираем первые 5 параметров для pairplot
        plot_pairplot(df, numeric_param_cols[:5], output_dir=args.output_dir)

    # Анализ важности признаков
    # Включаем также model_type как категориальный признак
    feature_cols = numeric_param_cols.copy()
    if 'model_type' in df.columns:
        feature_cols.append('model_type')
    model, top_features = feature_importance_analysis(df, feature_cols, output_dir=args.output_dir)

    # Partial dependence plots для топ-признаков
    if model is not None:
        # Подготовим X для PDP
        X = df[feature_cols].copy()
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        X = X.dropna()
        # Получаем список признаков в порядке убывания важности
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        top_feature_names = [feature_cols[i] for i in sorted_idx[:6]]
        plot_partial_dependence(model, X, top_feature_names, output_dir=args.output_dir)

    print("Анализ завершён.")


if __name__ == '__main__':
    main()