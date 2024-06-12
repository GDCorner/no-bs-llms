# Copyright 2024 GDCorner
# No BS Intro To Developing With LLMs

from llama_cpp import Llama

# Load the model
llm = Llama(
    model_path="Meta-Llama-3-8B-Instruct-q5_k_m.gguf",
    chat_format="llama-3",
)

# Begin the message history array, we'll start with only the system prompt, as
# we'll now prompt the user for input
messages_history=[
        {
            "role": "system",
            "content": "You are a helpful teacher who is teaching students about astronomy and the Solar System."
        },
    ]

# A little util function to pull out the interesting response from the LLMs
# chat completion response
# This just pulls out the dict with the content and role fields.
def get_message_from_response(response):
    """Get the message from the response of the LLM."""
    return response["choices"][0]["message"]
    
def add_user_message(message):
    """Add a user message to the message history."""
    messages_history.append(
        {
            "role": "user",
            "content": message
        }
    )
    
# A function to print the entire message history
def print_message_history():
    """Print the full message history."""
    print("===============================")
    for message in messages_history:
        print(message["role"], ":", message["content"])

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

        # Ask the LLM to generate a response from the history of the conversation
        llm_output = llm.create_chat_completion(
            messages=messages_history,
            max_tokens=128,  # Generate up to 128 tokens
        )

        # Add the message to our message history
        messages_history.append(get_message_from_response(llm_output))

        # Print the full conversation
        print_message_history()
        
    print("Chatbot finished - Goodbye!")

if __name__ == "__main__":
    main()