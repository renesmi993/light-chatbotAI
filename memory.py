import json
import os


def get_memory_filename(session_id: str) -> str:
    return f"memory_{session_id}.json"


def load_memory_from_file(session_id: str):
    """Загружает всю история переписки для сессии из файла."""
    filename = get_memory_filename(session_id)
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_memory_to_file(memory, session_id: str):
    filename = get_memory_filename(session_id)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def add_memory(message: str, role: str, session_id: str):
    memory = load_memory_from_file(session_id)
    memory.append({"role": role, "message": message})
    save_memory_to_file(memory, session_id)


def get_recent_memories(session_id: str, limit: int = 10):
    memory = load_memory_from_file(session_id)
    return memory[-limit:]


def clear_memory(session_id: str):
    filename = get_memory_filename(session_id)
    if os.path.exists(filename):
        os.remove(filename)
