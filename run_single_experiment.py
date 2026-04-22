#!/usr/bin/env python3
import subprocess
import sys
import os
import json
import argparse
import csv
from datetime import datetime

def update_csv_status(csv_path, exp_id, status, **kwargs):
    if not os.path.exists(csv_path):
        return
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row['experiment_id'] == exp_id:
                row['status'] = status
                for k, v in kwargs.items():
                    row[k] = str(v) if v is not None else ''
                rows.append(row)
            else:
                rows.append(row)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_id', required=True)
    parser.add_argument('--csv_log', default='experiments.csv')
    parser.add_argument('--json_file', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--python_cmd', default='python')
    args, unknown = parser.parse_known_args()

    start_time = datetime.now().isoformat()
    update_csv_status(args.csv_log, args.experiment_id, 'running', start_time=start_time)

    result_file = f"result_{args.experiment_id}.json"

    # Собираем аргументы для train_rae_per_json.py
    cmd = [
        args.python_cmd, 'train_rae_per_json.py',
        '--json_file', args.json_file,
        '--output_base', args.output_dir,
        '--experiment_id', args.experiment_id,
        '--result_file', result_file,
    ] + unknown

    print(f"[{args.experiment_id}] Запуск: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                res_data = json.load(f)
            best_loss = res_data.get('best_test_loss')
            best_epoch = res_data.get('best_epoch')
            run_dir = res_data.get('run_dir')
            end_time = datetime.now().isoformat()
            update_csv_status(args.csv_log, args.experiment_id, 'completed',
                              end_time=end_time,
                              best_test_loss=best_loss,
                              best_epoch=best_epoch,
                              run_dir=run_dir)
            os.remove(result_file)
            print(f"[{args.experiment_id}] Успешно завершён. Test Loss: {best_loss}")
        else:
            raise RuntimeError("Не создан файл результата")
    except subprocess.CalledProcessError as e:
        end_time = datetime.now().isoformat()
        error_msg = e.stderr[:500] if e.stderr else "Unknown error"
        update_csv_status(args.csv_log, args.experiment_id, 'failed',
                          end_time=end_time,
                          error_message=error_msg)
        print(f"[{args.experiment_id}] Ошибка:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()