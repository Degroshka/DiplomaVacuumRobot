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


## Quantitative experiment metrics

The controller writes quantitative experiment data automatically while Webots is running.
Files are stored in `controllers/rgbd_navigation_cleaner/metrics/`:

- `*_timeseries.csv` — time series: coverage percent, pose, travelled path length, frontier/gray-gap counts, owner source, route status, planner time, map-quality proxy values, **localization error** (odometry vs ground-truth GPS) and **false-occupied / precision / recall** of the learned obstacle map vs the world geometry;
- `*_events.csv` — discrete events: bumper contacts, recovery starts, route aborts, known-map waits and low-obstacle marks;
- `*_summary.json` and `latest_summary.json` — aggregate values for the diploma report (final/max coverage, path length, localization-error mean/RMSE/max, false-occupied ratio, obstacle precision/recall, event counts, time-by-owner).

The CSV is written **continuously every second** and the summary auto-exports periodically — no keypress is required.  Press `M` to force an export; `S` saves maps and also exports.

### Ground-truth metrics (localization accuracy, false-occupied cells)

Two metrics need a reference the navigation stack does not use:

- **Localization accuracy** — a metrics-only `GPS` node on the robot provides the simulator's true pose.  Navigation still runs purely on visual + wheel/IMU odometry; the GPS is read only to measure odometry drift (`localization_error_m`, summarised as mean/RMSE/max).
- **False-occupied cells** — `ground_truth_map.py` parses the `.wbt` world, keeps the collision solids whose height overlaps the robot body, rasterises them (plus arena walls) into a reference occupancy grid aligned to the controller's map, and compares the learned obstacle mask against it (`false_occupied_ratio`, `obstacle_precision`, `obstacle_recall`), restricted to explored cells.

Both are **evaluation-only**.  If the world has no GPS node, both are silently skipped.

To build report-ready tables and plots after a run:

```bash
python tools/analyze_metrics.py controllers/rgbd_navigation_cleaner/metrics/<metrics_file>_timeseries.csv
```

The analysis script generates a summary table, `coverage(t)` plot, robot trajectory plot and motion-owner distribution plot. These values are intended for the testing section of the diploma instead of hand-written estimates.
