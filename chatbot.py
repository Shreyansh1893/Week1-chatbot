import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


class ChatAgent:
    
    def __init__(
        self,
        model: str,
        system_prompt: str = "You are a helpful assistant.",
        max_turns: int = 7,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.last_call_usage = None

        self.client = OpenAI(
            base_url=base_url,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def call_model(self, user_input: str) -> str:
        
        self.messages.append({
            "role": "user",
            "content": user_input
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )

        self.last_call_usage = response.usage

        assistant_reply = response.choices[0].message.content

        self.messages.append({
            "role": "assistant",
            "content": assistant_reply
        })

        self.clear_history()

        return assistant_reply

    def clear_history(self):
        
        system_message = self.messages[0]
        conversation_messages = self.messages[1:]

        max_messages = self.max_turns * 2

        if len(conversation_messages) > max_messages:
            conversation_messages = conversation_messages[-max_messages:]

        self.messages = [system_message] + conversation_messages

    def reset(self):
        
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.last_call_usage = None

    def show_tokens(self):
        return self.last_call_usage


def choose_model() -> str:
    
    models = {
        "1": "deepseek/deepseek-v4-flash:free",
        "2": "openai/gpt-oss-120b:free",
        "3": "openai/gpt-oss-20b:free",
    }

    print("Choose a model:")
    print("1. deepseek/deepseek-v4-flash:free")
    print("2. openai/gpt-oss-120b:free")
    print("3. openai/gpt-oss-20b:free")

    choice = input("Enter choice number: ")

    if choice in models:
        return models[choice]

    print("Invalid choice. Using default model: deepseek/deepseek-v4-flash:free")
    return "deepseek/deepseek-v4-flash:free"


def run_chatbot():
    model = choose_model()

    agent = ChatAgent(
        model=model,
        system_prompt="You are a helpful assistant.",
        max_turns=7,
    )

    print("\nChat started.")
    print("Type 'exit' or 'quit' to stop.")
    print("Type '/reset' to clear history.")
    print("Type '/tokens' to show last token usage.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Model: Goodbye!")
            break

        if user_input.lower() == "/reset":
            agent.reset()
            
            continue

        if user_input.lower() == "/tokens":
            usage = agent.show_tokens()

            if usage is None:
                print("Model: No API call yet.")
            else:
                print("Token usage:", usage)

            continue

        reply = agent.call_model(user_input)
        print("Model:",reply)


if __name__ == "__main__":
    run_chatbot()