import json
import os
import shutil
import sys
import Data_maker
import yolo_main

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.m4v', '.mpg', '.mpeg'}
# Путь к датасетам: подкаталог datasets в папке проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_PATH = os.path.join(BASE_DIR, "datasets")

def get_unique_path(path):
    """Генерирует уникальное имя файла, добавляя _1, _2 и т.д."""
    if not os.path.exists(path):
        return path

    base, extension = os.path.splitext(path)
    counter = 1
    new_path = f"{base}_{counter}{extension}"

    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base}_{counter}{extension}"

    return new_path
def otf(dir_name, name, pack_name):
    """Генерация JSON-датасета из папки с видео одного типа упражнения."""
    video_dir = os.path.join(DATASETS_PATH, dir_name)
    if not os.path.isdir(video_dir):
        print(f"Ошибка: папка не найдена: {video_dir}")
        sys.exit(1)

    all_video = os.listdir(video_dir)
    json_name = name.strip().replace(' ', '_') + '.json'
    json_path = os.path.join(DATASETS_PATH, json_name)

    data = {}
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    all_vids = data.get("vids_list", [])

    for vid in all_video:
        vid_key = vid
        if vid_key in all_vids:
            print(f"Пропуск (уже обработано): {vid}")
            continue

        print(f"Обработка: {vid}")
        data[vid] = {}
        video_data = yolo_main.main_video(os.path.join(video_dir, vid))
        frame = 0
        for i in video_data:
            try:
                d = Data_maker.SplittedData(video_data[i])
                data[vid][frame] = d.return_data()
                frame += 1
            except KeyError as e:
                print(f"  Кадр {i}: пропущен (нет ключевой точки {e})")
            except Exception as e:
                print(f"  Кадр {i}: ошибка — {e}")

        all_vids.append(vid_key)
        data["vids_list"] = all_vids

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"  Сохранено {frame} кадров для {vid}")

def move_files(target_root, source_root):
    if not os.path.exists(source_root):
        print(f"Ошибка: Путь {source_root} не найден.")
        return

    for exercise_name in os.listdir(source_root):
        source_dir = os.path.join(source_root, exercise_name)

        if os.path.isdir(source_dir):
            target_dir = os.path.join(target_root, exercise_name)
            os.makedirs(target_dir, exist_ok=True)

            for file_name in os.listdir(source_dir):
                source_file = os.path.join(source_dir, file_name)

                if os.path.isfile(source_file):
                    # Формируем путь и проверяем на дубликаты
                    target_file = os.path.join(target_dir, file_name)
                    unique_target = get_unique_path(target_file)

                    # Копируем файл
                    shutil.copy2(source_file, unique_target)

                    # Выводим лог, если имя изменилось
                    final_name = os.path.basename(unique_target)
                    if final_name != file_name:
                        print(f"Переименовано: {file_name} -> {final_name}")
                        print(f"Скопировано: {file_name}")
                        pass
                    else:
                        pass


def install_pack():
    os.environ["KAGGLEHUB_CACHE"] = BASE_DIR

    import kagglehub
    path = kagglehub.dataset_download("hasyimabdillah/workoutfitness-video")
    path = kagglehub.dataset_download('riccardoriccio/real-time-exercise-recognition-dataset')
    move_files(source_root=r'datasets/hasyimabdillah/workoutfitness-video/versions/5',
               target_root=r'./datasets')
    move_files(source_root=r'datasets/riccardoriccio/real-time-exercise-recognition-dataset/versions/3/my_test_video_1',
               target_root=r'./datasets')
    move_files(source_root=r'datasets/riccardoriccio/real-time-exercise-recognition-dataset/versions/3/similar_dataset',
               target_root=r'./datasets')
    move_files(
        source_root=r'datasets/riccardoriccio/real-time-exercise-recognition-dataset/versions/3/synthetic_dataset/synthetic_dataset',
        target_root=r'./datasets')





def clean_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Проверяем, что это файл, а не папка
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()

            if ext not in VIDEO_EXTENSIONS:
                try:
                    os.remove(file_path)
                    print(f"Удалено: {filename}")
                except Exception as e:
                    print(f"Ошибка при удалении {filename}: {e}")
if __name__ == "__main__":

    for i in os.listdir(DATASETS_PATH):
        if os.path.isdir(DATASETS_PATH+'/'+i):
            clean_folder(DATASETS_PATH+'/'+i)
            otf(i, i, "5")
