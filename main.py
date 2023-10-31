from matplotlib.pyplot import imsave
from time import time

import project


def settings(model_path: str = 'D:/X-Ray/models/cylinders.stl', save_slices: bool = False,
             slices_path: str = 'D:/X-Ray/slices/', x_ray_detection_step: float = 0.5, show_progress: bool = True,
             l: int = 1000, L: int = 2000, h: int = 1000, H: int = 2000, W: int = 2000, scale: float = 0.25,
             phi_x: float = 0.0, phi_y: float = 0.0, phi_z: float = 0.0, x: int = project.L - project.l // 2,
             y: int = project.W // 2, z: int = 5 * project.H // 8, mu: float = 0.001, w_pix: int = 1000,
             d_pix: int = 1500) -> None:
    """
    :param model_path: Путь к файлу с .stl моделью
    :param save_slices: Сохранять изображения слайсов?
    :param slices_path: Путь к папке хранения слайсов
    :param x_ray_detection_step: Шаг сетки (в пикселях) для расчёта интенсивности излучения
    :param show_progress: Выводить в консоль прогресс вычислений?
    :param l: Длина горизонтальной области с датчиками, мм
    :param L: Расстояние от источника излучения до вертикали рамки с датчиками, мм
    :param h: Расстояние от источника излучения до земли, мм
    :param H: Высота вертикальной области с датчиками, мм
    :param W: Длина результирующего изображения, мм
    :param scale: Коэффициент скалирования модели
    :param phi_x: Угол поворота модели вокруг оси X, радианы
    :param phi_y: Угол поворота модели вокруг оси Y, радианы
    :param phi_z: Угол поворота модели вокруг оси Z, радианы
    :param x: Расположение центра модели по оси X, мм. НЕЛЬЗЯ ставить меньше L - l, смотри scheme1.jpg
    :param y: Расположение центра модели по оси Y, мм
    :param z: Расположение центра модели по оси Z, мм
    :param mu: Коэффициент рентгеновского затухания лучей
    :param w_pix: Ширина результирующего изображения, пиксели
    :param d_pix: Высота результирующего изображения, пиксели
    """
    for it in ['model_path', 'save_slices', 'slices_path', 'x_ray_detection_step', 'show_progress', 'scale',
               'phi_x', 'phi_y', 'phi_z', 'x', 'y', 'z', 'mu', 'w_pix', 'd_pix']:
        exec(f'project.{it.upper()} = {it}')

    for it in ['l', 'L', 'h', 'H', 'W']:
        exec(f'project.{it} = {it}')

    project.DELTA_Y = project.W / project.W_PIX
    project.DELTA_J = (project.l + project.H) / project.D_PIX
    project.l_pix = int(project.l / project.DELTA_J)
    project.H_pix = project.D_PIX - project.l_pix
    project.L_pix = int(project.L / project.DELTA_J)
    project.h_pix = int(project.h / project.DELTA_J)


start = time()
settings(phi_y=2)
display = project.workflow()
print(f'Время работы программы {round(time() - start, 2)} сек.')

imsave("D:/X-Ray/results/cylinders.png", display, cmap='grey')
