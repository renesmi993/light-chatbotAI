
VECTOR_MEMORY.PY
# в соотвествии с данными в канва на 1:48


import faiss
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Модель эмбеддингов, размерность 384
model = SentenceTransformer("all-MiniLm-L6-v2")
embedding_dim = 384


# Пути к файлам
indexes = {}  # Словарь для хранения индексов по сессиям
vector_to_texts = {}  # Словарь для хранения индексов и текстов по сессиям

STORAGE_DIR = "vector_storage"

os.makedirs(STORAGE_DIR, exist_ok=True)


def embed_text(text: str) -> np.ndarray:
    """Переобразует текст в эмбеддинг."""
    embedding = model.encode([text])
    return np.array(embedding).astype("float32")


def get_index(session_id: str):
    """Возвращает индекс для сессии, создавая по необходимости."""
    if session_id not in indexes:
        index_path = os.path.join(STORAGE_DIR, f"{session_id}_index.npy")
        text_path = os.path.join(STORAGE_DIR, f"{session_id}_texts.json")

        index = faiss.IndexFlatL2(embedding_dim)
        texts = []

        if os.path.exists(index_path) and os.path.exists(text_path):
            try:
                data = np.load(index_path)
                if data.shape[0] > 0:
                    index.add(data)
                with open(text_path, "r", encoding="utf-8") as f:
                    texts = json.load(f)
            except Exception as e:
                print(
                    f"Ошибка при загрузке индекса или текстов для сессии {session_id}: {e}"
                )

        indexes[session_id] = index
        vector_to_texts[session_id] = texts

    return indexes[session_id]


def save_session_memory(session_id: str):
    index = indexes[session_id]
    texts = vector_to_texts[session_id]

    vectors = index.reconstruct_n(0, index.ntotal)
    index_path = os.path.join(STORAGE_DIR, f"{session_id}_index.npy")
    text_path = os.path.join(STORAGE_DIR, f"{session_id}_texts.json")

    np.save(index_path, vectors)

    with open(text_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)


def add_to_vector_memory(text: str, session_id: str):
    """Добавляет текст в память в виде эмбеддинга."""
    index = get_index(session_id)
    embedding = embed_text(text)
    index.add(embedding)
    vector_to_texts[session_id].append(text)
    save_session_memory(session_id)


def search_similar(text: str, session_id: str, top_k: int = 3) -> list[str]:
    """Находит похожие тексты по смыслу в рамках текущей сессии."""
    if session_id not in indexes or indexes[session_id].ntotal == 0:
        return []
    embedding = embed_text(text)
    distances, indices = indexes[session_id].search(embedding, top_k)
    return [
        vector_to_texts[session_id][i]
        for i in indices[0]
        if i < len(vector_to_texts[session_id])
    ]


CHAT.PY


import os
from openai import OpenAI
from dotenv import load_dotenv
from memory import get_recent_memories, add_memory, clear_memory
from vector_memory import add_to_vector_memory, search_similar

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# (Установливаем режим общения для сессии по предпочтению пользователя)
session_modes = {}  # session_id -> "default" , "mentor", "funny", и т.д.

SYSTEM_INSTRUCTIONS = {
    "default": "Ты - умный и отзывчивый чат-бот, который помогает пользователю, учитывает прошлый контекст и помогает пользователю.",
    "mentor": "Ты — опытный наставник, коуч и преподаватель, который помогает пользователю развиваться, учиться, справляться с трудностями "
    "и находить путь в жизни. Ты говоришь с уважением, вдохновляюще, но просто. Делишься знаниями, объясняешь по шагам, приводишь примеры. "
    "Твоя задача — не просто ответить, а помочь человеку расти и мыслить шире. Если вопрос не ясен — переспрашивай, уточняй, помогай разобраться.",
    "funny": "Ты - веселый чат-бот, который отвечает на вопросы с юмором и шутками. Делай общение легким и непринужденным.",
    "reflection": "Ты - рефлексивный чат-бот, который помогает пользователю анализировать свои мысли и чувства. Задавай вопросы, чтобы углубить понимание.",
}


def chat_with_memory(user_message: str, session_id: str) -> str:
    if user_message.strip().lower().startswith("/mode"):
        parts = user_message.strip().split()
        if len(parts) < 2:
            return "❗ Пожалуйста, укажи режим общения. Доступные режимы: /mode default, /mode mentor, /mode funny, /mode reflection."
        mode = parts[1]
        if mode not in SYSTEM_INSTRUCTIONS:
            return f"❗ Неверный режим общения. Доступные режимы: {','.join(SYSTEM_INSTRUCTIONS.keys())}"
        session_modes[session_id] = mode

        greetings = {
            "default": "🔧 Режим по умолчанию активирован. Готов помочь!",
            "mentor": "🎓 Режим наставника активирован. Готов обучать и помогать!",
            "funny": "😂 Веселый режим активирован. Готов шутить и развлекать!",
            "reflection": "🧠 Рефлексивный режим активирован. Готов помогать анализировать и понимать!",
        }

        return f"✅ Режим '{mode}'активирован!\n\n{greetings[mode]}"

    # Обработка комманд
    if user_message.strip().lower() == "/help":
        return (
            "🤖 Привет! Я Light Chatbot. Вот что я умею:\n"
            "• /help - показать список команд\n"
            "• /clear - очистить память\n"
            "• /exit - выйти из чата\n"
            "• /summary - показать краткое содержание диалога\n"
            "• /save - сохранить историю диалога в файл\n"
            "✨У меня также есть режимы общения:\n"
            "• /mode default - обычный режим\n"
            "• /mode mentor - режим наставника\n"
            "• /mode funny - веселый режим\n"
            "• /mode reflection - рефлексивный режим\n"
            "Просто напиши мне что-нибудь, и я отвечу!"
        )

    if (
        user_message.strip().lower() == "/mentor"
        or user_message.strip().lower() == "/funny"
        or user_message.strip().lower() == "/reflection"
        or user_message.strip().lower() == "/default"
    ):
        return "❗ Пожалуйста, используй команду /mode для смены режима общения."

    if user_message.strip().lower() == "/save":
        history = get_recent_memories(session_id)
        if not history:
            return "📜 Память пуста. Напиши что-нибудь, чтобы начать диалог!"

        formatted_history = "\n".join(
            f"{m['role'].capitalize()}: {m['message']}" for m in history
        )
        filename = f"session_{session_id}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(formatted_history)
        return f"💾 История сохранена в файл {filename}!"

    if user_message.strip().lower() == "/exit":
        return "👋 До встречи!"

    if user_message.strip().lower() == "/clear":
        clear_memory(session_id)
        return "🧠 Память очищена!"

    if user_message.strip().lower() == "/summary":
        history = get_recent_memories(session_id)
        if not history:
            return "📜 Память пуста. Напиши что-нибудь, чтобы начать диалог!"
        summary_prompt = [
            {
                "role": "system",
                "content": (
                    "Ты — искусственный интеллект, который создает краткое содержание всей беседы между пользователем и ассистентом. "
                    "Извлеки ключевые вопросы пользователя и ответы ассистента. Представь результат в виде краткого резюме, без воды и без приветствий."
                ),
            },
        ] + [{"role": m["role"], "content": m["message"]} for m in history]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.5,
            messages=summary_prompt,
        )
        return "📝 Содержание:\n" + response.choices[0].message.content

    # Обычный режим общения
    add_memory(user_message, "user", session_id)
    # Добавленная строка.(добавление сообщения юзера в векторную память)
    add_to_vector_memory(user_message, session_id)

    # Ищем похожие сообщения

    similar_context = search_similar(user_message, session_id, top_k=3)
    similar_messages = [{"role": "user", "content": txt} for txt in similar_context]

    recent = get_recent_memories(session_id)

    # Получаем текущий режим
    mode = session_modes.get(session_id, "default")
    system_message = {"role": "system", "content": SYSTEM_INSTRUCTIONS[mode]}

    messages = (
        [system_message]
        + similar_messages
        + [{"role": m["role"], "content": m["message"]} for m in recent]
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=messages,
    )

    reply = response.choices[0].message.content
    add_memory(reply, "assistant", session_id)

    return reply


    MAIN.PY


    from chat import chat_with_memory


def main():
    print(
        "🧠 Light Chatbot: начни диалог (type 'exit' to leave). Напиши /help для посмотра команд."
    )
    name = input("Как тебя зовут? ").strip()
    session_id = f"session_{name.lower()}"

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("👋 До встречи!")
            break
        response = chat_with_memory(user_input, session_id)
        print("Bot:", response)

        if response.strip() == "👋 До встречи!":
            break


if __name__ == "__main__":
    main()


    MEMORY.PY


    import json
import os


def _get_session_file(session_id: str) -> str:
    return f"{session_id}.json"


def add_memory(message: str, role: str, session_id: str):
    session_file = _get_session_file(session_id)
    memory_store = []

    if os.path.exists(session_file):
        with open(session_file, "r", encoding="utf-8") as f:
            memory_store = json.load(f)

    memory_store.append({"role": role, "message": message})

    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(memory_store, f, indent=2, ensure_ascii=False)


def get_recent_memories(session_id: str, n=10):
    session_file = _get_session_file(session_id)
    if os.path.exists(session_file):
        with open(session_file, "r", encoding="utf-8") as f:
            memory_store = json.load(f)
            return memory_store[-n:]
    return []


def clear_memory(session_id: str):
    session_file = _get_session_file(session_id)
    if os.path.exists(session_file):
        os.remove(session_file) 



REQUIREMENTS.TXT


openai
python-dotenv
sentence-transformers
faiss-cpu





+ .env ФАЙЛ
