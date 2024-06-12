# Copyright 2024 GDCorner
# No BS Intro To Developing With LLMs

from openai import OpenAI, ChatCompletion
from pprint import pprint

# Load the model
client = OpenAI(base_url="http://localhost:8080/",
                api_key="local-no-key-required")

# Begin the message history array, we'll start with only the system prompt, as
# we'll now prompt the user for input
messages_history=[
        {
            "role": "system",
            "content": "You are a helpful teacher who is teaching students about astronomy and the Solar System."
        },
    ]

# A little util function to pull out the interesting response from the OpenAI
# chat completion response.
# This just pulls out the dict with the content and role fields.
def get_message_from_openai_response(response: ChatCompletion):
    chat_message = response.choices[0].message
    response = {
        "role": chat_message.role,
        "content": chat_message.content
    }
    return response
    
def add_user_message(message):
    """Add a user message to the message history."""
    messages_history.append(
        {
            "role": "user",
            "content": message
        }
    )

def add_assistant_message(message):
    """Add a user message to the message history."""
    messages_history.append(
        {
            "role": "assistant",
            "content": message
        }
    )
    
# A function to print the entire message history
def print_message_history():
    """Print the full message history."""
    print("===============================")
    for message in messages_history:
        print(message["role"], ":", message["content"])

def format_chat_llama3(message_history):
    START_BLOCK = """<|begin_of_text|>"""
    MESSAGE_BLOCK = """<|start_header_id|>{user_id}<|end_header_id|>

{system_prompt}<|eot_id|>"""
    ASSISTANT_START = """<|start_header_id|>{user_id}<|end_header_id|>

"""

    chat_str = START_BLOCK
    for message in message_history:
        chat_str += MESSAGE_BLOCK.format(
            user_id=message["role"],
            system_prompt=message["content"]
        )
    
    # Add the assistant start block ready for the assistant to begin replying:
    chat_str += ASSISTANT_START.format(user_id="assistant")
    return chat_str

def main():
    # Chat loop
    while True:
        # Get a new user input
        user_message = input("(Type exit to quit) User: ")

        # Check if the user has tried to quit
        if user_message.strip() == "exit":
            break

        # Add the user message to the message history
        add_user_message(user_message)

        formatted_chat = format_chat_llama3(messages_history)

        print("Formatted chat:")
        print(formatted_chat)

        # Generate a response from the history of the conversation
        llm_output = client.completions.create(
            model="Meta-Llama-3-8B-Instruct-q5_k_m",
            prompt=formatted_chat
        )

        full_response = llm_output.content

        add_assistant_message(full_response)

        # Print the full conversation
        print_message_history()

    print("Chatbot finished - Goodbye!")

if __name__ == "__main__":
    main()
