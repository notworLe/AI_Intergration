from pathlib import Path

current_path = Path.cwd()
print(current_path)
print(current_path.stem)
print(current_path / ".." / "new")