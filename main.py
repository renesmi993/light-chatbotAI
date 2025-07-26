from chat import chat_with_memory
from memory import load_memory_from_file


def main():
    print(
        "🧠 Light Chatbot: начни диалог (type 'exit' to leave). Напиши /help для посмотра команд."
    )
    name = input("Как тебя зовут? ").strip()
    session_id = name.lower()

    # Загружаем историю переписки из файла
    previous_history = load_memory_from_file(session_id)
    if previous_history:
        print(f"👋 Привет, {name}!")
        print("📜 Твоя предыдущая переписка:")
        for m in previous_history:
            role = "Ты" if m["role"] == "user" else "Бот"
            print(f"{role}: {m['message']}")
        print("———")
    else:
        print(f"👋 Привет, {name}! Это твоя первая сессия.")

    while True:
        user_input = input("Ты: ").strip()
        if user_input.lower() == "exit":
            print("👋 До встречи!")
            break
        response = chat_with_memory(user_input, session_id)
        print("Bot:", response)

        if response.strip() == "👋 До встречи!":
            break


if __name__ == "__main__":
    main()
