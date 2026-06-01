# Cleanup manifest

База: step112_bumper_foremost_shell_fix.

Выполнено:
- корневая папка переименована в `cleaning_robot_rgbd_navigation_webots`;
- контроллер переименован в `rgbd_navigation_cleaner`;
- основной файл контроллера переименован в `rgbd_navigation_cleaner.py`;
- Robot name в WBT заменён на `cleaning_robot_rgbd`;
- из пользовательских строк и комментариев убрано некорректное название;
- удалены `_refs`, старые markdown-заметки, `.bak`-файлы, патчи, видео, ROS/URDF-заготовки и `__pycache__`;
- оставлены только Webots world и controller files;
- добавлены `README.md` и `requirements.txt`.

Не менялось:
- логика движения;
- логика построения карты;
- геометрия робота и бамперов из step112;
- настройки камеры/RangeFinder из текущей рабочей версии.
