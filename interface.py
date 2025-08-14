import gradio as gr
from chat import chat_with_memory
from memory import get_recent_memories
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

session_storage = {}  # глобальная переменная для хранения session_id


def handle_name_submit(name):
    if not name.strip():
        return [], "Пожалуйста, введите имя."

    session_id = f"session_{name.lower().strip()}"

    # Загружаем историю, если она есть
    history = get_recent_memories(session_id)
    chat_history = []

    if history:
        chat_history.append(
            {
                "role": "assistant",
                "content": f"👋 С возвращением, {name.strip()}! Вот ваша история чата:",
            }
        )

        # Восстанавливаем формат сообщений для Chatbot
        temp_user_msg = None
        for item in history:
            if item["role"] == "user":
                temp_user_msg = item["message"]
            elif item["role"] == "assistant" and temp_user_msg:
                chat_history.append({"role": "user", "content": temp_user_msg})
                chat_history.append({"role": "assistant", "content": item["message"]})
                temp_user_msg = None

        notification = f"Существующая сессия '{name.strip()}' восстановлена."

    else:
        greeting = f"👋 Привет, {name.strip()}!"
        help_text = chat_with_memory("/help", session_id)
        chat_history = [
            {"role": "assistant", "content": greeting},
            {"role": "assistant", "content": help_text},
        ]
        notification = f"Новая сессия '{name.strip()}' создана."

    return chat_history, notification


def handle_chat(name: str, message: str, chat_history: list):
    if not name.strip():
        return "Пожалуйста, введите ваше имя.", chat_history

    session_id = f"session_{name.lower().strip()}"

    if session_id not in session_storage:
        session_storage[session_id] = []

    response = chat_with_memory(message, session_id)

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})

    return "", chat_history


with gr.Blocks(title="Light Chatbot AI") as demo:
    gr.Markdown("## 🧠 Light Chatbot")
    gr.Markdown("Введите имя и начните диалог!")

    name_input = gr.Textbox(
        label="Имя пользователя", placeholder="Введите имя", value=""
    )
    status_output = gr.Markdown("")
    chatbot = gr.Chatbot(label="Диалог", type="messages")  # Новый формат
    msg_input = gr.Textbox(label="Сообщение", placeholder="Напиши что-нибудь...")

    # Обработка ввода имени
    name_input.submit(
        handle_name_submit, inputs=name_input, outputs=[chatbot, status_output]
    )

    # Обработка чата
    msg_input.submit(
        handle_chat,
        inputs=[name_input, msg_input, chatbot],
        outputs=[msg_input, chatbot],
    )

if __name__ == "__main__":
    demo.launch(share=True)
