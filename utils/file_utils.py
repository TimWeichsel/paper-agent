def load_file(path: str, default_error_message: str = "File not found") -> str:
    try:
        with open(path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return default_error_message
    
def save_file(path: str, content: str) -> None:
    with open(path, "w") as file:
        file.write(content)

def append_to_file(path: str, content: str) -> None:
    with open(path, "a") as file:
        file.write(content)