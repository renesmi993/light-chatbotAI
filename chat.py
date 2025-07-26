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
