import gradio as gr
from chat import chat_with_memory
from memory import get_recent_memories

session_storage = {}  # глбальна переменная для хранение session_id


def handle_name_submit(name):
    if not name.strip():
        return gr.update(), [], "Пожалуйста, введите имя."

    session_id = f"session_{name.lower().strip()}"

    # Загружаем историю если лна есть
    history = get_recent_memories(session_id)
    chat_history = []  # Создаем пустую историю #абоба

    if history:
        chat_history.append(
            (None, f"👋 С возвращением, {name.strip()}! Вот ваша история чата:")
        )

        # переобразуем [(role, message), ...] → [ (user, assistant), ... ]
        temp_user_msg = None
        for item in history:
            if item["role"] == "user":
                temp_user_msg = item["message"]
            elif item["role"] == "assistant" and temp_user_msg:
                chat_history.append((temp_user_msg, item["message"]))
                temp_user_msg = None

        notification = f"Сущевствующая сессия '{name.strip()}' восстановлена."

    else:
        greeting = f"👋 Привет, {name.strip()}!"
        help_text = chat_with_memory("/help", session_id)
        chat_history = [(None, greeting), (None, help_text)]
        notification = f" Новая сессия '{name.strip()}' создана."

    # АПвтоматическоре приветсвтие и список комнад
    return gr.update(visible=True), chat_history, notification


def handle_chat(name: str, message: str, chat_history: list):
    if not name.strip():
        return "Пожалуйста, введите ваше имя.", chat_history

    session_id = f"session_{name.lower().strip()}"

    if session_id not in session_storage:
        session_storage[session_id] = []

    response = chat_with_memory(message, session_id)

    chat_history.append((message, response))

    return "", chat_history


with gr.Blocks(title="Light Chatbot AI") as demo:
    gr.Markdown("## 🧠 Light Chatbot")
    gr.Markdown("Введите имя и начните диалог!")

    name_input = gr.Textbox(
        label="Имя пользователя", placeholder="Введите имя", value=""
    )
    status_output = gr.Markdown("")  # Новое поле для уведомлений
    chatbot = gr.Chatbot(label="Диалог")
    msg_input = gr.Textbox(label="Сообщение", placeholder="Напиши что-нибудь...")

    # Добавили обработку ввода имени (Enter по имени)
    name_input.submit(
        handle_name_submit, inputs=name_input, outputs=[chatbot, chatbot, status_output]
    )

    msg_input.submit(
        handle_chat,
        inputs=[name_input, msg_input, chatbot],
        outputs=[msg_input, chatbot],
    )

if __name__ == "__main__":
    demo.launch(share=True)


# Продолжаем, заяц, сразу заходи в чат и начинай с создания README файла

# Смотрим в ютубе как создать репозиторий конкретно в гитхабе,
# и что делать c README кошка.
# (Спрашивай у чата как крепить реадми в гитхабе)


# чекни папку lightchatbotAi слева, и спроси чат,
# норм ли это то что в папке папка.
# Затем, двигайся дальше, к видео

# Сразу зохадим в чат, и на крайне сообщение
