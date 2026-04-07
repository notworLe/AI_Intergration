
def is_path_existed(path):
    current = os.getcwd()
    path = os.path.join(current, path)
    if os.path.exists(path):
       return True
    else:
        raise ValueError(f"Path: {path} isn't existed")
