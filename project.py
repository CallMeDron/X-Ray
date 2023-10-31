import trimesh
import numpy as np
from math import floor

# Для понимания задачи, геометрии, осей и параметров рекомендуется ознакомиться с файлами scheme.jpg

# Общие параметры программы
MODEL_PATH: str = 'D:/X-Ray/models/cylinders.stl'  # Путь к файлу с .stl моделью
SAVE_SLICES: bool = False  # Сохранять изображения слайсов?
SLICES_PATH: str = 'D:/X-Ray/slices/'  # Путь к папке хранения слайсов
X_RAY_DETECTION_STEP: float = 0.5  # Шаг сетки (в пикселях) для расчёта интенсивности излучения
SHOW_PROGRESS: bool = True  # Выводить информацию о прогрессе вычислений в консоль?

# Параметры конструкции. Все значения в мм
l: int = 1000  # Длина горизонтальной области с датчиками
L: int = 2000  # Расстояние от источника излучения до вертикали рамки с датчиками
h: int = 1000  # Расстояние от источника излучения до земли
H: int = 2000  # Высота вертикальной области с датчиками
W: int = 2000  # Длина результирующего изображения

# Параметры преобразования модели
SCALE: float = 0.25  # Коэффициент скалирования модели
# Углы поворота модели вокруг осей, радианы. Повороты применяются последовательно в этом порядке
PHI_X: float = 0.0
PHI_Y: float = 0.0
PHI_Z: float = 0.0

# Расположение центра модели в пространстве, мм
X: int = L - l // 2  # Нельзя ставить меньше L - l, смотри scheme1.jpg
Y: int = W // 2
Z: int = 5 * H // 8

MU: float = 0.001  # Коэффициент рентгеновского затухания лучей

# Разрешение результирующего изображения, пиксели
W_PIX: int = 1000  # Ширина
D_PIX: int = 1500  # Высота

# Мм/пиксель по осям Y и (X, Z) соответственно
DELTA_Y: float = W / W_PIX
DELTA_J: float = (l + H) / D_PIX

# Перевод параметров из мм в пиксели
l_pix = int(l / DELTA_J)
H_pix = D_PIX - l_pix
L_pix = int(L / DELTA_J)
h_pix = int(h / DELTA_J)


def get_stl_size(model_path: str) -> list[float]:
    """
    Расчёт размеров .stl модели из файла (миллиметры)
    :param model_path: путь к модели
    :return: размеры модели по осям X, Y, Z
    """
    mesh: trimesh.Trimesh = trimesh.load_mesh(model_path)
    vert_coords: list[list[float]] = [sorted([v[i] for v in mesh.vertices]) for i in range(3)]
    print([round(vert_coords[i][-1] - vert_coords[i][0], 2) for i in range(3)])
    return [vert_coords[i][-1] - vert_coords[i][0] for i in range(3)]


def get_mesh_pozition(mesh: trimesh.Trimesh) -> list[list[float]]:
    """
        Расчёт границ координатной коробки модели mesh в миллиметрах
        :param mesh: модель в формате trimesh.Trimesh
        :return: мин. и макс. координаты по осям X, Y, Z
    """
    vert_coords: list[list[float]] = [sorted([v[i] for v in mesh.vertices]) for i in range(3)]
    return [[vert_coords[i][0], vert_coords[i][-1]] for i in range(3)]


def slicer() -> [list[float], np.ndarray[bool]]:
    """
    Переносит модель в начало координат, скалирует, поворачивает и нарезает плоскостями y = const.
    Одна из плоскостей проходит через центр координатной коробки модели, остальные идут с шагом, заданным DELTA_Y.
    :return: массив с размерами модели и массив, где первая ось - номер слайса, а двумерный массив внутри - сам слайс
    """
    mesh: trimesh.Trimesh = trimesh.load_mesh(MODEL_PATH)  # Загрузка модели

    # Скалирование (умножение всех размеров на коэффициент scale)
    if SCALE != 1.0:
        mesh.apply_scale(SCALE)

    # Изначальное расположение модели в пространстве (границы координатной коробки)
    mesh_position: list[list[float]] = get_mesh_pozition(mesh)

    # Центрирование, перенос центра координатной коробки в 0
    mesh.vertices -= np.array([sum(mesh_position[i]) / 2 for i in range(3)])

    # Применяем повороты модели на заданные углы
    phi_vect: list[list[float, list[int]]] = [pair for pair in
                                              [[PHI_X, [1, 0, 0]], [PHI_Y, [0, 1, 0]], [PHI_Z, [0, 0, 1]]]
                                              if pair[0]]
    for pair in phi_vect:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(pair[0], pair[1], [0, 0, 0]))

    mesh_position: list[list[float]] = get_mesh_pozition(mesh)
    mesh.vertices -= np.array([sum(mesh_position[i]) / 2 for i in range(3)])

    # Пересчёт новой коробки
    mesh_position: list[tuple[float]] = get_mesh_pozition(mesh)
    mesh_size: list[float] = [mesh_position[i][1] - mesh_position[i][0] for i in range(3)]  # Размеры модели

    # Далее проводится мультислайсинг модели плоскостями y = const, с шагом DELTA_Y и проходя через центр модели.
    # Необходимо использовать section_multiplane(), а не цикл из section() для того, чтобы все слайсы имели
    # одинаковую ориентацию.

    num: int = floor(mesh_size[1] / DELTA_Y)  # Число слайсов, чтобы точно не выйти за коробку
    if not num % 2:  # Приведение к нечётности, чтобы проходить через центр
        num -= 1
    border: float = (num - 1) // 2 * DELTA_Y  # Вычисленные границы разбиения
    # Получаем массив слайсов в формате Path2D, часть из которых None, поскольку пересечения с моделью не было
    slices_path: list[trimesh.path.Path2D] = mesh.section_multiplane([0, 0, 0],
                                                                     [0, 1, 0],
                                                                     np.linspace(- border, border, num=num))

    # Поиск максимальных границ всех разрезов для установки вручную
    max_bounds: list[list[float]] = [[min([slice_.bounds[0][i] for slice_ in slices_path if slice_]) for i in range(2)],
                                     [max([slice_.bounds[1][i] for slice_ in slices_path if slice_]) for i in range(2)]]

    # Массив слайсов в формате двумерных bool массивов. Тип в дальнейшем можно будет поменять для неоднородностей.
    # На первом месте ось Y, затем X и Z (или Z и X)
    slices_np: list[np.ndarray[bool]] = []

    for i, slice_ in enumerate(slices_path):
        file = None
        if slice_:
            # Ручное задание слайсу размера в модельном пространстве.
            # Делается в обход отсутствия сеттера для slice_.bounds, к счастью, работает.
            for j in range(2):
                slice_.bounds[j] = max_bounds[j]

            # Превращение 2D контура в изображение с заполненными(!) пустотами.
            # Очень важный этап, если бы эта функция не была реализована, задача бы сильно усложнялась.
            # Разрешение итогового файла зависит от размеров модели в мм. и коэффициента перевода мм. в пиксели
            # для оси J. Внутри производится округление вверх. Итоговый размер чуть больше ожидаемого,
            # т.к. добавляются границы, но это абсолютно не критично
            file = slice_.rasterize(pitch=DELTA_J)

        if file:
            slices_np.append(np.asarray(file))
            if SAVE_SLICES:  # Сохранение полученных изображений для просмотра
                file.save(fp=f'{SLICES_PATH}slice_{i}.png')
        else:
            slices_np.append(np.zeros(slices_np[0].shape))

    # Возврат размеров модели в мм и массив слайсов
    return [round(x) for x in mesh_size], np.array(slices_np)


def insert(sliced_model: np.ndarray[bool], space: np.ndarray[bool]) -> list[list[int]]:
    """
    Помещает трёхмерный массив sliced_model в трёхмерный массив space, так, что центр sliced_model
    находится в координатах X, Y, Z.
    :param sliced_model: помещаемая модель
    :param space: пространство
    :return: координаты результирующих границ модели в пикселях
    """
    x, y, z = int((X - (L - l)) / DELTA_J), int(Y / DELTA_Y), int(Z / DELTA_J)  # Перевод заданных координат в пиксели

    mins: list[int] = [pair[0] - sliced_model.shape[pair[1]] // 2 for pair in [[x, 1], [y, 0], [z, 2]]]
    maxs: list[int] = [mins[pair[0]] + sliced_model.shape[pair[1]] for pair in [[0, 1], [1, 0], [2, 2]]]

    if mins[0] < 0 or maxs[0] > l or mins[1] < 0 or maxs[1] > W or mins[2] < 0 or maxs[2] > H:
        print(f'Центр модели нельзя поместить в {X=} {Y=} {Z=}!', end='\n\n')
        return None

    for j in range(sliced_model.shape[0]):
        for i in range(sliced_model.shape[1]):
            space[mins[1] + j][mins[0] + i][mins[2]:maxs[2]] = sliced_model[j][i]
    return [[mins[1], maxs[1]], [mins[0], maxs[0]], [mins[2], maxs[2]]]


def x_ray(pixel_size: list[list[int]], space: np.ndarray[bool]) -> np.ndarray[float]:
    """
    Создаём пустой экран и заполняем его проекцией облучённой модели
    :param pixel_size: положение модели в пространстве
    :param space: пространство
    :return: результируюзее изображение
    """
    assert pixel_size is not None

    # Создаём пустое изображение
    display: np.ndarray[float] = np.ones([W_PIX, D_PIX], dtype=float)

    # Вычисляем пиксельные размеры проекции модели на изображение, чтобы многократно сократить дальнейшие вычисления.
    # a - угловые коэффициенты прямых, ограничивающих тень
    a_down: float = min([(pixel_size[2][0] - h_pix) / (L_pix - l_pix + pixel_size[1][i]) for i in range(2)])
    down_proection: int = int(a_down * L_pix + h_pix - 1)

    a_up: float = max([(pixel_size[2][1] - h_pix) / (L_pix - l_pix + pixel_size[1][i]) for i in range(2)])
    upper_proection: int = int(a_up * L_pix + h_pix + 1)

    border_1: int = max(0, down_proection)
    border_2: int = min(H_pix, upper_proection)
    border_3: int = H_pix
    border_4: int = H_pix + min(l_pix + 1, int((upper_proection - H_pix) / a_up))

    if SHOW_PROGRESS:
        print('Прогресс вычислений:')

    # Проходим по всем пикселям, которые попали в тень модели
    for j in range(pixel_size[0][0], pixel_size[0][1]):

        # Для вертикальных датчиков
        for i in range(border_1, border_2):
            intensity: float = 1.0  # Начальная интенсивность
            a: float = (i - h_pix) / L_pix  # Угловой коэффициент прямой от источника до текущего пикселя

            # Сетки для детекции того, что луч прошёл через модель. Сначала float, затем дискретезируются
            grid_len: float = l_pix * ((1.0 + a * a) ** 0.5)  # Длина соответствующего отрезка
            n: int = int(grid_len / X_RAY_DETECTION_STEP)  # Число точек разбиения
            x_grid: np.ndarray[float] = np.linspace(0, l_pix - 1, n)
            z_grid: np.ndarray[float] = x_grid * a + h_pix + int((L_pix - l_pix) * a)
            x_grid, z_grid = np.array(x_grid, dtype=int), np.array(z_grid, dtype=int)

            # За каждую точку сетки, попавшую на модель, интенсивность луча уменьшается по экспоненциальному закону
            for ind in range(n):
                if space[j][x_grid[ind]][z_grid[ind]]:
                    intensity -= intensity * MU * X_RAY_DETECTION_STEP

            # На экране отображается финальная интенсивность
            display[j][i] = intensity

        # Для горизонтальных датчиков
        for i in range(border_3, border_4):
            intensity: float = 1.0  # Начальная интенсивность
            # Угловой коэффициент прямой от источника до текущего пикселя
            a: float = (H_pix - h_pix) / (L_pix - (i - H_pix))

            # Сетки для детекции того, что луч прошёл через модель. Сначала float, затем дискретезируются
            # Длина соответствующего отрезка
            grid_len: float = ((H_pix - (L_pix - l_pix) * a) ** 2 + (l_pix - (i - H_pix)) ** 2) ** 0.5
            n: int = int(grid_len / X_RAY_DETECTION_STEP)  # Число точек разбиения
            x_grid: np.ndarray[float] = np.linspace(0, l_pix - (i - H_pix) - 1, n)
            z_grid: np.ndarray[float] = x_grid * a + h_pix + int((L_pix - l_pix) * a)
            x_grid, z_grid = np.array(x_grid, dtype=int), np.array(z_grid, dtype=int)

            # За каждую точку сетки, попавшую на модель, интенсивность луча уменьшается по экспоненциальному закону
            for ind in range(n):
                if space[j][x_grid[ind]][z_grid[ind]]:
                    intensity -= intensity * MU * X_RAY_DETECTION_STEP

            # На экране отображается финальная интенсивность
            display[j][i] = intensity

        # Печать прогресса вычислений
        if SHOW_PROGRESS:
            tmp = (pixel_size[0][1] - pixel_size[0][0]) / 100
            print(f'{round((j - pixel_size[0][0]) / tmp, 1)}%')

    return np.rot90(display)


def workflow() -> np.ndarray[float]:
    # Скалируем, поворачиваем и нарезаем модель
    model_size, sliced_model = slicer()
    print('Реальные размеры модели, мм XYZ:', model_size)
    print('Пиксельные размеры модели XYZ:', [sliced_model.shape[i] for i in [1, 0, 2]], end='\n\n')

    # Создаём пустое пространство, куда потом поместим модель
    space: np.ndarray[bool] = np.zeros([W_PIX, l_pix, H_pix], dtype=bool)
    print('Реальные размеры пространства, мм XYZ:', [l, W, H])
    print('Пиксельные размеры пространства XYZ:', [space.shape[i] for i in [1, 0, 2]], end='\n\n')

    # Помещаем модель в пространство
    pixel_size = insert(sliced_model, space)
    print('Помещаем в точку, мм XYZ:', [X, Y, Z])
    print('Модель расположена между пикселями XYZ:', [pixel_size[i] for i in [1, 0, 2]], end='\n\n')

    # Создаём изображение
    display: np.ndarray[float] = x_ray(pixel_size, space)
    print('Пиксельные размеры изображения YJ:', display.shape)

    return display
