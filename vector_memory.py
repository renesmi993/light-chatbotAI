import faiss
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_contextual_summary(text: str, session_id: str) -> str:
    """Генерирует краткое содержание текста с помощью OpenAI с учетом контекста сессии."""
    prompt = (
        f"Пользователь в сессии '{session_id}' написал следующее сообщение:\n"
        f'"{text}"n\n'
        f"Сформулируй краткое обобщение этого сообщения: суть, намерения, интересы или чувства пользователя. "
        f"Формулируй в третьем лице, как будто ты кратко объясняешь суть другому ИИ. Коротко, только важная суть."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Ты помощник, который делает краткие, но точные смысловые записи.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=100,
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"[OpenAI ERROR: {e}")
        return f"Сообщение от пользователя в сессии '{session_id}': {text}"


# Модель эмбеддингов, размерность 384
model = SentenceTransformer("all-MiniLm-L6-v2")
embedding_dim = 384

# Папка для хранения памяти сессий
VECTOR_MEMORY_DIR = "vector_sessions"
os.makedirs(VECTOR_MEMORY_DIR, exist_ok=True)

# Кэш индексов и текстов
session_indices = {}  # session_id -> FAISS index
session_texts = {}  # session_id -> list of texts


def embed_text(text: str) -> np.ndarray:
    """Переобразует текст в эмбеддинг."""
    embedding = model.encode([text])
    return np.array(embedding).astype("float32")


def get_faiss_path(session_id: str) -> str:
    """Возвращает путь к файлу FAISS индекса для сессии."""
    return os.path.join(VECTOR_MEMORY_DIR, f"{session_id}.faiss")


def get_texts_path(session_id: str) -> str:
    """Возвращает путь к файлу с текстами для сессии."""
    return os.path.join(VECTOR_MEMORY_DIR, f"{session_id}_texts.json")


def load_vector_memory(session_id: str):
    """Загружает векторную память для сессии, создавая по необходимости."""
    faiss_path = get_faiss_path(session_id)
    texts_path = get_texts_path(session_id)

    if os.path.exists(faiss_path):
        index = faiss.read_index(faiss_path)
    else:
        index = faiss.IndexFlatL2(embedding_dim)

    if os.path.exists(texts_path):
        with open(texts_path, "r", encoding="utf-8") as f:
            texts = json.load(f)
    else:
        texts = []

    session_indices[session_id] = index
    session_texts[session_id] = texts


def save_session_memory(session_id: str):
    """Сохраняет векторную память сессии на диск."""
    faiss_path = get_faiss_path(session_id)
    texts_path = get_texts_path(session_id)

    index = session_indices.get(session_id)
    texts = session_texts.get(session_id)

    if index is not None:
        faiss.write_index(index, faiss_path)
    if texts is not None:
        with open(texts_path, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)


def add_to_vector_memory(text: str, session_id: str):
    """Добавляет текст в векторную память конкретной сессии."""
    if session_id not in session_indices:
        load_vector_memory(session_id)

    summary = generate_contextual_summary(text, session_id)
    embedding = embed_text(summary)

    session_indices[session_id].add(embedding)
    session_texts[session_id].append(summary)

    # Сохраняем изменения в памяти сессии
    save_session_memory(session_id)


def search_similar(text: str, session_id: str, top_k: int = 3) -> list[str]:
    """Находит похожие тексты по смыслу в рамках текущей сессии."""
    if session_id not in session_indices:
        load_vector_memory(session_id)

    index = session_indices[session_id]
    texts = session_texts[session_id]

    if index.ntotal == 0:
        return []
    embedding = embed_text(text)
    distances, indices = index.search(embedding, top_k)

    return [texts[i] for i in indices[0] if i < len(texts)]
