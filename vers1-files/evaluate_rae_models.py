import os
import json
import pandas as pd
import argparse

def parse_training_log(log_path):
    if not os.path.exists(log_path):
        return None, None, {}
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.split('\n')
    best_epoch = None
    best_loss = None
    params = {}
    in_params = False
    for line in lines:
        line = line.strip()
        if line.startswith('Best epoch:'):
            parts = line.split('|')
            best_epoch = int(parts[0].split(':')[1].strip())
            best_loss = float(parts[1].split(':')[1].strip())
        elif line.startswith('Parameters:'):
            in_params = True
            continue
        elif in_params:
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()
                try:
                    if '.' in val:
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    pass
                params[key] = val
            elif line == '':
                in_params = False
    return best_epoch, best_loss, params

def find_all_models(models_dir):
    records = []
    for exercise in sorted(os.listdir(models_dir)):
        ex_path = os.path.join(models_dir, exercise)
        if not os.path.isdir(ex_path):
            continue
        runs = [d for d in os.listdir(ex_path) if d.startswith('run_')]
        for run in runs:
            run_path = os.path.join(ex_path, run)
            log_file = os.path.join(run_path, 'training_log.txt')
            model_file = os.path.join(run_path, 'best_model.pth')
            if not os.path.exists(model_file):
                continue
            best_epoch, best_loss, params = parse_training_log(log_file)
            if best_loss is None:
                continue
            record = {
                'exercise': exercise,
                'run': run,
                'best_test_loss': best_loss,
                'best_epoch': best_epoch,
                **params
            }
            records.append(record)
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str, default='models')
    parser.add_argument('--output_csv', type=str, default='rae_ranking.csv')
    parser.add_argument('--top_n', type=int, default=10)
    args = parser.parse_args()

    records = find_all_models(args.models_dir)
    if not records:
        print("Не найдено моделей с логами.")
        return

    df = pd.DataFrame(records)
    df_sorted = df.sort_values('best_test_loss').reset_index(drop=True)
    df_sorted.index = df_sorted.index + 1

    print("\n=== ТОП лучших моделей ===\n")
    print(df_sorted.head(args.top_n).to_string())

    df_sorted.to_csv(args.output_csv, index_label='rank')
    print(f"\nПолный рейтинг сохранён в {args.output_csv}")

    # Лучшая модель для каждого упражнения
    print("\n=== Лучшая модель для каждого упражнения ===")
    best_per_ex = df_sorted.groupby('exercise').first().reset_index()
    print(best_per_ex[['exercise', 'run', 'best_test_loss', 'best_epoch']].to_string(index=False))

if __name__ == '__main__':
    main()