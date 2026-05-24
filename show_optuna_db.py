#!/usr/bin/env python3
"""
show_optuna_db.py – Просмотр содержимого баз данных Optuna.
Выводит информацию о всех исследованиях, найденных в SQLite файлах optuna_*.db.
"""

import os
import glob
import sqlite3
import json
import argparse
from typing import Dict, List, Any, Optional


def get_table_columns(cursor, table_name: str) -> List[str]:
    """Возвращает список имён столбцов таблицы."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [col[1] for col in cursor.fetchall()]


def get_best_trial(cursor, study_id: int, direction: str) -> Optional[tuple]:
    """
    Возвращает кортеж (trial_id, best_value) для лучшего trial в исследовании.
    Если таблица trials не содержит столбца value, пытается использовать
    intermediate_values (последнее значение) или возвращает None.
    """
    trials_cols = get_table_columns(cursor, "trials")
    # Определяем, какой столбец использовать для значения метрики
    value_col = None
    if 'value' in trials_cols:
        value_col = 'value'
    elif 'intermediate_values' in trials_cols:
        value_col = 'intermediate_values'
    else:
        return None  # не можем получить значение

    # Сортируем в зависимости от направления
    order = 'ASC' if direction.upper() == 'MINIMIZE' else 'DESC'

    # Для столбца intermediate_values нужно взять последнее записанное значение
    if value_col == 'intermediate_values':
        # Получаем все trial_id и intermediate_values, затем в Python обрабатываем
        cursor.execute(f"""
            SELECT trial_id, intermediate_values
            FROM trials
            WHERE study_id = ? AND state = 'COMPLETE'
        """, (study_id,))
        rows = cursor.fetchall()
        best_trial = None
        best_val = None
        for trial_id, json_str in rows:
            if json_str:
                try:
                    values = json.loads(json_str)
                    if values:
                        # Берем последнее intermediate значение
                        last_val = values[-1].get('value') if isinstance(values[-1], dict) else values[-1]
                        if best_val is None or (direction.upper() == 'MINIMIZE' and last_val < best_val) \
                           or (direction.upper() != 'MINIMIZE' and last_val > best_val):
                            best_val = last_val
                            best_trial = trial_id
                except:
                    continue
        return (best_trial, best_val) if best_trial is not None else None
    else:
        cursor.execute(f"""
            SELECT trial_id, {value_col}
            FROM trials
            WHERE study_id = ? AND state = 'COMPLETE'
            ORDER BY {value_col} {order}
            LIMIT 1
        """, (study_id,))
        return cursor.fetchone()


def get_studies_from_db(db_path: str) -> List[Dict[str, Any]]:
    """
    Извлекает информацию о всех исследованиях из базы данных Optuna.
    Возвращает список словарей.
    """
    studies = []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Проверяем наличие таблицы studies
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='studies'")
    if not cursor.fetchone():
        conn.close()
        return studies

    # Получаем все исследования
    cursor.execute("SELECT study_id, study_name FROM studies")
    study_rows = cursor.fetchall()

    for study_id, study_name in study_rows:
        # Определяем направление оптимизации
        direction = 'MINIMIZE'  # по умолчанию
        # Пробуем получить direction из study_system_attributes
        cursor.execute("""
            SELECT value_json 
            FROM study_system_attributes 
            WHERE study_id = ? AND key = 'study:direction'
        """, (study_id,))
        row = cursor.fetchone()
        if row:
            try:
                direction = json.loads(row[0])
            except:
                pass
        else:
            # Старые версии - колонка direction в таблице studies
            cols = get_table_columns(cursor, "studies")
            if 'direction' in cols:
                cursor.execute("SELECT direction FROM studies WHERE study_id = ?", (study_id,))
                dir_row = cursor.fetchone()
                if dir_row:
                    direction = dir_row[0]

        # Лучший trial
        best_info = get_best_trial(cursor, study_id, direction)
        if best_info:
            best_trial_id, best_value = best_info
        else:
            best_trial_id, best_value = None, None

        # Параметры лучшего trial
        best_params = {}
        if best_trial_id is not None:
            cursor.execute("""
                SELECT param_name, param_value
                FROM trial_params
                WHERE trial_id = ?
            """, (best_trial_id,))
            for param_name, param_value in cursor.fetchall():
                try:
                    best_params[param_name] = json.loads(param_value)
                except json.JSONDecodeError:
                    best_params[param_name] = param_value

        # Количество завершённых trial
        cursor.execute("""
            SELECT COUNT(*)
            FROM trials
            WHERE study_id = ? AND state = 'COMPLETE'
        """, (study_id,))
        n_trials = cursor.fetchone()[0]

        studies.append({
            'study_name': study_name,
            'study_id': study_id,
            'direction': direction,
            'best_trial_id': best_trial_id,
            'best_value': best_value,
            'best_params': best_params,
            'n_trials': n_trials,
        })

    conn.close()
    return studies


def print_studies_table(studies: List[Dict[str, Any]]):
    """Выводит таблицу исследований в читаемом виде."""
    if not studies:
        print("No studies found.")
        return

    print(f"{'Study Name':<40} {'Direction':<10} {'Trials':<8} {'Best Value':<15} {'Best Params'}")
    print("-" * 100)
    for s in studies:
        params_str = ', '.join(f"{k}={v}" for k, v in s['best_params'].items())
        best_val_str = f"{s['best_value']:.6f}" if s['best_value'] is not None else "N/A"
        print(f"{s['study_name']:<40} {s['direction']:<10} {s['n_trials']:<8} {best_val_str:<15} {params_str}")


def export_to_csv(studies: List[Dict[str, Any]], output_file: str):
    """Экспортирует сводку в CSV."""
    import csv
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['study_name', 'direction', 'n_trials', 'best_value', 'best_params'])
        for s in studies:
            writer.writerow([
                s['study_name'],
                s['direction'],
                s['n_trials'],
                s['best_value'],
                json.dumps(s['best_params'])
            ])
    print(f"Exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Show Optuna database contents')
    parser.add_argument('--db_pattern', default='optuna_*.db',
                        help='Glob pattern for Optuna SQLite files (default: optuna_*.db)')
    parser.add_argument('--export_csv', help='Export summary to CSV file')
    parser.add_argument('--verbose', action='store_true', help='Show all parameters for each study')
    args = parser.parse_args()

    db_files = glob.glob(args.db_pattern)
    if not db_files:
        print(f"No database files found matching pattern: {args.db_pattern}")
        return

    all_studies = []
    for db_file in db_files:
        print(f"\n--- Database: {db_file} ---")
        studies = get_studies_from_db(db_file)
        all_studies.extend(studies)
        print_studies_table(studies)

        if args.verbose:
            for s in studies:
                print(f"\nStudy: {s['study_name']}")
                print(f"  Direction: {s['direction']}")
                print(f"  Completed trials: {s['n_trials']}")
                print(f"  Best trial ID: {s['best_trial_id']}")
                print(f"  Best value: {s['best_value']}")
                print("  Best parameters:")
                for k, v in s['best_params'].items():
                    print(f"    {k}: {v}")

    if args.export_csv and all_studies:
        export_to_csv(all_studies, args.export_csv)


if __name__ == '__main__':
    main()