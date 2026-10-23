"""
prepare_dataset.py — підготовка набору Dogs vs Cats для навчання нейромережі в Keras.

1) знаходить усі зображення котів і собак (підтримує обидва формати архіву:
   PetImages/Cat, PetImages/Dog  — дзеркало Microsoft,  або  train/cat.N.jpg, train/dog.N.jpg — Kaggle);
2) відкидає пошкоджені файли (у наборі є порожні та «биті» зображення);
3) перемішує і ділить на навчальний (70%), оціночний (15%) і тестовий (15%) набори;
4) копіює у структуру каталогів, яку розуміє ImageDataGenerator.flow_from_directory:
      dataset/train/cats, dataset/train/dogs, dataset/val/..., dataset/test/...
"""
import os, sys, shutil, random
from PIL import Image

dataset_dir = 'dataset'                         # куди складаємо результат
train_dir = os.path.join(dataset_dir, 'train')  # каталог з даними для навчання
val_dir = os.path.join(dataset_dir, 'val')      # каталог з даними для перевірки
test_dir = os.path.join(dataset_dir, 'test')    # каталог з даними для тестування
val_data_portion = 0.15                         # частка зображень для перевірки
test_data_portion = 0.15                        # частка зображень для тестування
random.seed(42)
# необов'язковий аргумент: максимум зображень кожного класу (для навчання без GPU), напр. python prepare_dataset.py 3000
max_per_class = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None


def find_images():
    """Повертає словник {'cats': [шляхи], 'dogs': [шляхи]}."""
    if os.path.isdir('PetImages'):              # дзеркало Microsoft (той самий набір Kaggle)
        lst = lambda d: [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith('.jpg')]
        return {'cats': lst('PetImages/Cat'), 'dogs': lst('PetImages/Dog')}
    if os.path.isdir('train'):                  # оригінальний train.zip з Kaggle: cat.0.jpg, dog.0.jpg ...
        files = os.listdir('train')
        return {'cats': [os.path.join('train', f) for f in files if f.startswith('cat.')],
                'dogs': [os.path.join('train', f) for f in files if f.startswith('dog.')]}
    raise FileNotFoundError('Не знайдено ні PetImages/, ні train/ — розпакуйте архів з набором даних')


def is_valid(path):
    """Перевіряє, що файл — справжнє зображення, яке відкривається."""
    try:
        if os.path.getsize(path) == 0:
            return False
        with Image.open(path) as img:
            img.verify()                        # перевірка цілісності файлу
        with Image.open(path) as img:
            img.convert('RGB').load()           # повне декодування
        return True
    except Exception:
        return False


def create_directory(dir_name):
    """Створює каталог з підкаталогами cats і dogs (старий видаляє)."""
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(os.path.join(dir_name, 'cats'))
    os.makedirs(os.path.join(dir_name, 'dogs'))


def copy_images(files, dest_dir, class_name):
    for i, src in enumerate(files):
        shutil.copy2(src, os.path.join(dest_dir, class_name, f'{class_name[:-1]}.{i}.jpg'))


images = find_images()
for cls in images:
    total = len(images[cls])
    images[cls] = [p for p in images[cls] if is_valid(p)]
    print(f'{cls}: знайдено {total}, пошкоджених відкинуто {total - len(images[cls])}')

# однакова кількість зображень кожного класу
nb_images = min(len(images['cats']), len(images['dogs']))
if max_per_class:
    nb_images = min(nb_images, max_per_class)
    print(f'Обмеження: {nb_images} зображень кожного класу')
start_val_data_idx = int(nb_images * (1 - val_data_portion - test_data_portion))
start_test_data_idx = int(nb_images * (1 - test_data_portion))

for d in (train_dir, val_dir, test_dir):
    create_directory(d)

for cls, files in images.items():
    random.shuffle(files)
    files = files[:nb_images]
    copy_images(files[:start_val_data_idx], train_dir, cls)
    copy_images(files[start_val_data_idx:start_test_data_idx], val_dir, cls)
    copy_images(files[start_test_data_idx:], test_dir, cls)

for d in (train_dir, val_dir, test_dir):
    print(f'{d}: cats = {len(os.listdir(os.path.join(d, "cats")))}, dogs = {len(os.listdir(os.path.join(d, "dogs")))}')
