# Cleanup manifest

Выполнено:
- корневая папка называется `cleaning_robot_rgbd_navigation_webots`;
- контроллер называется `rgbd_navigation_cleaner`;
- Robot name в WBT: `cleaning_robot_rgbd`;
- из пользовательских строк и комментариев убраны старые названия и patch-history markers;
- удалены backup-файлы, старые markdown-заметки, `.bak`-файлы, патчи, видео, ROS/URDF-заготовки и `__pycache__`;
- удалены пустые stub-модули без рабочей логики;
- статические настройки вынесены в `controllers/rgbd_navigation_cleaner/config.py`;
- оставлены только Webots world, controller files, README и requirements.

Не менялось намеренно:
- Webots API, сенсоры и управление моторами остаются в основном контроллере;
- runtime-state не вынесен в отдельный процесс, чтобы не получить рассинхрон карты, pose и owner;
- RangeFinder по-прежнему описывается как depth-канал RGB-D системы, не лидар.
