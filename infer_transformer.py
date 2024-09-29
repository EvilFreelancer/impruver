from transformers import AutoTokenizer, AutoModelForCausalLM


class ChatHistory:
    def __init__(self, history_limit: int = None):
        self.history_limit = history_limit
        self.system_prompt = (f"You are a helpful assistant you help people with answers to their questions. "
                              f"If you don't know the answer, then say 'I don't know'.")
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def add_message(self, role, message):
        self.messages.append({"role": role, "content": message})
        self.trim_history()

    def add_user_message(self, message):
        self.add_message("user", message)

    def add_assistant_message(self, message):
        self.add_message("assistant", message)

    def trim_history(self):
        appendix = 0
        if self.system_prompt is not None:
            appendix = 1
        if self.history_limit is not None and len(self.messages) > self.history_limit + appendix:
            overflow = len(self.messages) - (self.history_limit + appendix)
            self.messages = [self.messages[0]] + self.messages[overflow + appendix:]

    def get_messages(self) -> list:
        return self.messages


# Init pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./output")
model = AutoModelForCausalLM.from_pretrained("./output")

# Start chat loop
chat_history = ChatHistory(history_limit=10)
while True:
    user_message = input("User: ")

    # Reset chat command
    if user_message.strip() == "/reset":
        chat_history = ChatHistory(history_limit=10)
        print("History reset completed!")
        continue

    # Skip empty messages from user
    if user_message.strip() == "":
        continue

    # Add user message to chat history
    chat_history.add_user_message(user_message)

    # Get list of messages
    messages = chat_history.get_messages()

    print(messages)
    exit()
