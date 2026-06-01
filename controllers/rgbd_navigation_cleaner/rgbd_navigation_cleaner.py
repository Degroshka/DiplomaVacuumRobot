"""Webots entry point for the RGB-D navigation controller.

The executable controller file stays intentionally small. The implementation
lives in navigation_controller.py, while static tunables live in config.py.
Webots imports this file as the controller script, and the import below starts
the controller loop.
"""

# Importing navigation_controller runs the Webots controller loop.
import navigation_controller  # noqa: F401
