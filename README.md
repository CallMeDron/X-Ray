![Sphinx](https://github.com/CallMeDron/X-Ray/assets/62312385/218b2c2b-46a5-44e1-beff-8daa3577b73f)

# Проект X-Ray посвящён численному созданию рентгеновских снимков трёхмерных объектов

## О чём речь и зачем это нужно?
Представим, что перед нами стоит задача обеспечить проверку багажа пассажиров аэропорта на наличие запрещённых предметов (сейчас этим вручную занимаются специально обученные люди, которые рассматривают рентегновские изображения). Цель - быстрый и надёжный автоматизированную процесс, исключающий человеческие ошибки. Для этого мы наверняка захотим обучить ML-модель, а значит, нам потребуется большое количество обучающих данных. Этих данных у нас скорее всего нет, либо их недостаточно: слишком долго и дорого 10.000 раз пропускать через рамку рентгена чемоданы с разными предметами. Однако эти данные можно сгенерировать искусственно, потратив лишь небольшое время на генерацию изображений и не ограничиваясь в объектах. Именно эту задачу решает данная программа.

## Как это работает?
На вход подаётся трёхмерная модель в формате .stl. Она размещается в заданном положении, скалируется и поворачивается, затем нарезается на двумерные сечения, которые представляют собой матрицы типа bool определённого размера. После нарезания двумерные слайсы помещаются в трёхмерный массив bool, соответствующий пространству внутри сканера. Далее происходит имитация снижения интенсивности рентгеновского луча при прохождении через материал по экспоненциальному закону с помощью дискретной равномерной сетки. Результат - двумерное ч/б изображение, образованное разной интенсивностью проходящих лучей.

## Что в репозитории?
Папка schemes - три изображения, которые помогут понять геометрию задачи, параметры и направления осей
Папка models - три .stl модели, с которыми можно поработать для примера
Папка results - примеры результирующих изображений
requirements.txt - необходимые для работы программы зависимости
main.py - файл, через который можно запускать программу с разными параметрами
project.py - основной файл, содержащий все содержательные функции
