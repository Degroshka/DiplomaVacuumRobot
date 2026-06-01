# Cleaning Robot RGB-D Navigation Webots Project

Дипломный демонстрационный проект: система навигации мобильного робота в помещении на базе компьютерного зрения.

## Что внутри

- `worlds/cleaning_world.wbt` — Webots-сцена.
- `controllers/rgbd_navigation_cleaner/rgbd_navigation_cleaner.py` — короткая точка входа Webots.
- `controllers/rgbd_navigation_cleaner/navigation_controller.py` — основная логика навигации.
- `controllers/rgbd_navigation_cleaner/config.py` — статические настройки контроллера.
- `controllers/rgbd_navigation_cleaner/*.py` — вспомогательные модули для карты, safety, recovery, wall-follow и motion primitives.
- `controllers/rgbd_navigation_cleaner/maps/` — папка для сохраняемых карт.
- `controllers/rgbd_navigation_cleaner/camera_frames/` — папка для сохраняемых RGB-D/debug кадров.

## Терминология

Проект не называется vSLAM и не использует ORB-SLAM/RTAB-Map. Это учебный RGB-D navigation prototype:

- RGB camera — источник изображения для OpenCV-обработки;
- RangeFinder — depth-канал RGB-D системы, не лидар;
- wheel odometry + IMU — оценка положения и курса;
- bumpers — контактное подтверждение препятствий;
- occupancy map / coverage map / frontier map — внутренние карты навигации.

## Запуск

1. Открыть `worlds/cleaning_world.wbt` в Webots.
2. Убедиться, что у робота указан controller `rgbd_navigation_cleaner`.
3. Запустить симуляцию.

## Клавиши контроллера

- `S` — сохранить карту;
- `R` — сброс карты;
- `K` — переключить режим known-map evaluation;
- `+` / `-` — масштаб карты;
- `C` — авто/полный вид карты;
- `0` — сброс вида;
- `Q` / `Esc` — скрыть окна OpenCV.

## Что было удалено из дипломной версии

Из архива убраны старые backup-файлы, экспериментальные заметки, patch-файлы, внешние reference-проекты, видео, ROS/URDF-заготовки, пустые stub-модули и `__pycache__`. Оставлены только файлы, необходимые для запуска Webots-сцены и контроллера.
