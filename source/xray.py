import trimesh
import numpy as np
from math import floor
from matplotlib.pyplot import imsave
from time import time
from tqdm import tqdm
from typing import Optional
from os.path import isfile, isdir


def xray(model_path: str = 'D:/X-Ray/models/dreadnought.stl',
         result_path: str = 'D:/X-Ray/results/dreadnought.png',
         save_slices: bool = False,
         slices_dir: str = 'D:/X-Ray/slices/',
         x_ray_detection_step: float = 0.5,
         show_content: bool = True,
         l: int = 500,
         L: int = 700,
         h: int = 100,
         H: int = 500,
         W: int = 1000,
         scale: float = 5.0,
         X: int = 450,
         Y: int = 500,
         Z: int = 330,
         phi_x: float = 0.0,
         phi_y: float = 0.0,
         phi_z: float = 1.56,
         mu: float = 0.005,
         w_pix: int = 600,
         d_pix: int = 600) -> None:
    """
    Создаёт двумерный рентгеновский снимок трёхмерного объекта, заданного .stl моделью
    :param model_path: Путь к файлу с .stl моделью
    :param result_path: Путь к файлу с результирующим изображением
    :param save_slices: Сохранять изображения слайсов?
    :param slices_dir: Путь к папке хранения слайсов
    :param x_ray_detection_step: Шаг сетки (в пикселях) для расчёта интенсивности излучения
    :param show_content: Выводить workflow информацию?
    :param l: Длина горизонтальной области с датчиками, мм
    :param L: Расстояние от источника излучения до вертикали рамки с датчиками, мм
    :param h: Расстояние от источника излучения до земли, мм
    :param H: Высота вертикальной области с датчиками, мм
    :param W: Длина результирующего изображения, мм
    :param scale: Коэффициент скалирования модели
    :param X: Расположение центра модели в пространстве, мм
    :param Y: Расположение центра модели в пространстве, мм
    :param Z: Расположение центра модели в пространстве, мм
    :param phi_x: Угол поворота модели относительно оси X, рад.
    :param phi_y: Угол поворота модели относительно оси Y, рад.
    :param phi_z: Угол поворота модели относительно оси Z, рад.
    :param mu: Коэффициент рентгеновского затухания лучей
    :param w_pix: Ширина результирующего изображения, пиксели
    :param d_pix: Высота результирующего изображения, пиксели
    :return:
    """

    # Корректность типов
    assert all([isinstance(var, str) for var in (model_path, result_path, slices_dir)])
    assert all([isinstance(var, bool) for var in (save_slices, show_content)])
    assert all([isinstance(var, int) for var in (l, L, h, H, W, X, Y, Z, w_pix, d_pix)])
    assert all([isinstance(var, float) for var in (x_ray_detection_step, scale, phi_x, phi_y, phi_z, mu)])

    # Корректность значений
    assert isfile(model_path)
    assert isdir(slices_dir)
    assert 0 < x_ray_detection_step
    assert 0 < l <= L
    assert 0 <= h < H
    assert 0 < scale
    assert L - l < X < L
    assert 0 < Y < W
    assert 0 < Z < H
    assert mu >= 0
    assert 0 < w_pix
    assert 0 < d_pix

    start: Optional[float] = time() if show_content else None

    # Мм/пиксель по осям Y и (X, Z) соответственно
    delta_y: float = W / w_pix
    delta_j: float = (l + H) / d_pix

    # Перевод параметров из мм в пиксели
    l_pix: int = int(l / delta_j)
    L_pix: int = int(L / delta_j)
    h_pix: int = int(h / delta_j)
    H_pix: int = d_pix - l_pix

    def get_mesh_pozition(mesh: trimesh.Trimesh) -> list[list[float]]:
        """
        Расчёт границ координатной коробки модели mesh в миллиметрах
        :param mesh: модель в формате trimesh.Trimesh
        :return: мин. и макс. координаты по осям X, Y, Z
        """
        assert isinstance(mesh, trimesh.Trimesh)
        pozition: list[list[float]] = []
        for i in range(3):
            vert_coords: np.ndarray[float] = np.array([v[i] for v in mesh.vertices])
            pozition.append([np.min(vert_coords), np.max(vert_coords)])
        return pozition

    def load_transform_and_slice() -> tuple[tuple[float], np.ndarray[bool]]:
        """
        Загружает модель, переносит в начало координат, скалирует, поворачивает и нарезает плоскостями y = const.
        Одна из плоскостей проходит через центр координатной коробки модели, остальные идут с шагом, заданным delta_y.
        :return: массив с размерами модели и массив двумерных массивов - слайсов
        """
        if show_content:
            print(f'{round(time() - start, 2)} > Происходит загрузка модели')

        # Загрузка модели
        mesh: trimesh.Trimesh = trimesh.load_mesh(model_path)

        if show_content:
            print(f'{round(time() - start, 2)} > Происходит трансформация модели')

        # Скалирование (умножение всех размеров на коэффициент scale)
        if scale != 1.0:
            mesh.apply_scale(scale)

        # Изначальное расположение модели в пространстве (границы координатной коробки)
        mesh_position: list[list[float]] = get_mesh_pozition(mesh)

        # Центрирование, перенос центра координатной коробки в 0
        mesh.vertices -= np.array([sum(mesh_position[i]) / 2 for i in range(3)])

        # Применяем повороты модели на заданные углы
        phi_vect: list[list[float, list[int]]] = [pair for pair in
                                                  [[phi_x, [1, 0, 0]], [phi_y, [0, 1, 0]], [phi_z, [0, 0, 1]]]
                                                  if pair[0]]
        for pair in phi_vect:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(pair[0], pair[1], [0, 0, 0]))

        mesh_position: list[list[float]] = get_mesh_pozition(mesh)
        mesh.vertices -= np.array([sum(mesh_position[i]) / 2 for i in range(3)])

        # Пересчёт новой коробки
        mesh_position: list[tuple[float]] = get_mesh_pozition(mesh)
        mesh_size: list[float] = [mesh_position[i][1] - mesh_position[i][0] for i in range(3)]  # Размеры модели

        num: int = floor(mesh_size[1] / delta_y)  # Число слайсов, чтобы точно не выйти за коробку
        if not num % 2:  # Приведение к нечётности, чтобы проходить через центр
            num -= 1
        border: float = (num - 1) // 2 * delta_y  # Вычисленные границы разбиения

        if show_content:
            print(f'{round(time() - start, 2)} > Происходит нарезка модели')

        # Далее проводится мультислайсинг модели плоскостями y = const, с шагом delta_y и проходя через центр модели.
        # Необходимо использовать section_multiplane(), а не цикл из section() для того, чтобы все слайсы имели
        # одинаковую ориентацию.

        slices_trimesh: list[Optional[trimesh.path.Path2D]] = \
            mesh.section_multiplane([0, 0, 0], [0, 1, 0], np.linspace(- border, border, num=num))

        # Получаем массив слайсов в формате Path2D, часть из которых None, поскольку пересечения с моделью не было

        # Поиск максимальных границ всех разрезов для установки вручную
        max_bounds: list[list[float]] = [[min([slc.bounds[0][i] for slc in slices_trimesh if slc]) for i in range(2)],
                                         [max([slc.bounds[1][i] for slc in slices_trimesh if slc]) for i in range(2)]]

        # Массив слайсов в формате двумерных bool массивов. Тип в дальнейшем можно будет поменять для неоднородностей.
        # На первом месте ось Y, затем X и Z (или Z и X)
        slices: list[np.ndarray[bool]] = []

        if show_content:
            print(f'{round(time() - start, 2)} > Происходит обработка и сохранение слайсов')

        for i, slc in enumerate(slices_trimesh):
            file = None
            if slc:
                # Ручное задание слайсу размера в модельном пространстве.
                # Делается в обход отсутствия сеттера для slc.bounds, к счастью, работает.
                for j in range(2):
                    slc.bounds[j] = max_bounds[j]

                # Превращение 2D контура в изображение с заполненными(!) пустотами.
                # Очень важный этап, если бы эта функция не была реализована, задача бы сильно усложнялась.
                # Разрешение итогового файла зависит от размеров модели в мм. и коэффициента перевода мм. в пиксели
                # для оси J. Итоговый размер чуть больше ожидаемого, т.к. внутри rasterize происходит округление вверх
                # и добавление пары дополнительных пикселей
                file = slc.rasterize(pitch=delta_j)

            if file:
                slices.append(np.asarray(file))
                if save_slices:  # Сохранение полученных изображений для просмотра
                    file.save(fp=f'{slices_dir}slice_{i}.png')
            else:
                slices.append(np.zeros(slices[0].shape))

        # Возврат размеров модели в мм и массив слайсов
        return tuple(round(x) for x in mesh_size), np.array(slices)

    def insert(sliced_model: np.ndarray[bool], space: np.ndarray[bool]) -> list[list[int]]:
        """
        Помещает трёхмерный массив sliced_model в трёхмерный массив space, так, что центр sliced_model
        находится в координатах X, Y, Z.
        :param sliced_model: помещаемая модель
        :param space: пространство
        :return: координаты результирующих границ модели в пикселях
        """
        if show_content:
            print(f'{round(time() - start, 2)} > Происходит копирование слайсов в пространство')

        # Перевод заданных координат в пиксели
        x, y, z = int((X - (L - l)) / delta_j), int(Y / delta_y), int(Z / delta_j)

        mins: list[int] = [pair[0] - sliced_model.shape[pair[1]] // 2 for pair in [[y, 0], [x, 1], [z, 2]]]
        maxs: list[int] = [mins[i] + sliced_model.shape[i] for i in range(3)]

        if (not 0 <= mins[0] < maxs[0] < space.shape[0] or
                not 0 <= mins[1] < maxs[1] < space.shape[1] or
                not 0 <= mins[2] < maxs[2] < space.shape[2]):
            print(f'Центр модели нельзя поместить в {X=} {Y=} {Z=}!', end='\n\n')
            exit()

        for j in range(sliced_model.shape[0]):
            for i in range(sliced_model.shape[1]):
                space[mins[0] + j][mins[1] + i][mins[2]:maxs[2]] = sliced_model[j][i]

        return [[mins[i], maxs[i]] for i in range(3)]

    def create_picture(pixel_size: list[list[int]], space: np.ndarray[bool]) -> np.ndarray[float]:
        """
        Создаём пустой экран и заполняем его проекцией облучённой модели
        :param pixel_size: положение модели в пространстве
        :param space: пространство
        :return: результируюзее изображение
        """
        if show_content:
            print(f'{round(time() - start, 2)} > Происходит создание изображения')

        # Создаём пустое изображение
        display: np.ndarray[float] = np.ones([w_pix, d_pix], dtype=float)

        # Вычисляем пиксельные размеры проекции модели на изображение, чтобы сократить дальнейшие вычисления
        # a - угловые коэффициенты прямых, ограничивающих тень
        a_down: float = min([(pixel_size[2][0] - h_pix) / (L_pix - l_pix + pixel_size[1][i]) for i in range(2)])
        down_proection: int = int(a_down * L_pix + h_pix - 1)

        a_up: float = max([(pixel_size[2][1] - h_pix) / (L_pix - l_pix + pixel_size[1][i]) for i in range(2)])
        upper_proection: int = int(a_up * L_pix + h_pix + 1)

        border_1: int = max(0, down_proection)
        border_2: int = min(H_pix, upper_proection)
        border_3: int = H_pix
        border_4: int = H_pix + min(l_pix + 1, int((upper_proection - H_pix) / a_up))

        # Проходим по всем пикселям, которые попали в тень модели
        for j in tqdm(range(pixel_size[0][0], pixel_size[0][1])):

            # Для вертикальных датчиков
            for i in range(border_1, border_2):
                intensity: float = 1.0  # Начальная интенсивность
                a: float = (i - h_pix) / L_pix  # Угловой коэффициент прямой от источника до текущего пикселя

                # Сетки для детекции того, что луч прошёл через модель. Сначала float, затем дискретезируются
                grid_len: float = l_pix * ((1.0 + a * a) ** 0.5)  # Длина соответствующего отрезка
                n: int = int(grid_len / x_ray_detection_step)  # Число точек разбиения
                x_grid: np.ndarray[float] = np.linspace(0, l_pix - 1, n)
                z_grid: np.ndarray[float] = x_grid * a + h_pix + int((L_pix - l_pix) * a)
                x_grid, z_grid = np.array(x_grid, dtype=int), np.array(z_grid, dtype=int)

                # За каждую точку сетки, попавшую на модель, интенсивность луча уменьшается по экспоненциальному закону
                for ind in range(n):
                    if space[j][x_grid[ind]][z_grid[ind]]:
                        intensity -= intensity * mu * x_ray_detection_step

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
                n: int = int(grid_len / x_ray_detection_step)  # Число точек разбиения
                x_grid: np.ndarray[float] = np.linspace(0, l_pix - (i - H_pix) - 1, n)
                z_grid: np.ndarray[float] = x_grid * a + h_pix + int((L_pix - l_pix) * a)
                x_grid, z_grid = np.array(x_grid, dtype=int), np.array(z_grid, dtype=int)

                # За каждую точку сетки, попавшую на модель, интенсивность луча уменьшается по экспоненциальному закону
                for ind in range(n):
                    if space[j][x_grid[ind]][z_grid[ind]]:
                        intensity -= intensity * mu * x_ray_detection_step

                # На экране отображается финальная интенсивность
                display[j][i] = intensity

        return display.T

    def workflow() -> None:
        """
        Запуск основного процесса программы
        """
        # Загружаем, скалируем, поворачиваем и нарезаем модель
        try:
            model_size, sliced_model = load_transform_and_slice()
        except Exception:
            print("Ошибка в функции load_transform_and_slice")
            exit()

        # Создаём пустое пространство, куда потом поместим модель
        space: np.ndarray[bool] = np.zeros([w_pix, l_pix, H_pix], dtype=bool)

        # Помещаем модель в пространство
        try:
            pixel_size = insert(sliced_model, space)
        except Exception:
            print("Ошибка в функции insert")
            exit()

        # Создаём изображение
        try:
            result: np.ndarray[float] = create_picture(pixel_size, space)
        except Exception:
            print("Ошибка в функции create_picture")
            exit()

        # Сохраняем изображение в файл
        try:
            imsave(result_path, result, cmap='grey')
        except Exception:
            print("Ошибка при сохранении результата")
            exit()

        if show_content:
            print(f'{round(time() - start, 2)} > Программа успешно завершила работу\n\n'
                  f'Размеры модели: {model_size} мм, {[sliced_model.shape[i] for i in [1, 0, 2]]} пикселей\n'
                  f'Размеры пространства: {l, W, H} мм, {[space.shape[i] for i in [1, 0, 2]]} пикселей\n'
                  f'Центр модели помещён в точку: {X, Y, Z} мм, между пикселями {[pixel_size[i] for i in [1, 0, 2]]}\n'
                  f'Пиксельные размеры изображения: {result.shape}')

    workflow()


def get_stl_size(model_path: str) -> list[float]:
    """
    Расчёт размеров .stl модели из файла (миллиметры) для предварительного изучения модели пользователем
    :param model_path: путь к модели
    :return: размеры модели по осям X, Y, Z
    """
    mesh: trimesh.Trimesh = trimesh.load_mesh(model_path)
    sizes: list[float] = []
    for i in range(3):
        vert_coords: np.ndarray[float] = np.array([v[i] for v in mesh.vertices])
        sizes.append(np.max(vert_coords) - np.min(vert_coords))
    print('Базовые размеры модели, мм XYZ:', [round(sizes[i], 2) for i in range(3)], end='\n\n')
    return sizes
