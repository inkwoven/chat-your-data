import os
import openai
import gradio as gr
from threading import Lock

# Define the name of the environment variable where the API key is stored
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"

class ChatWrapper:
    def __init__(self):
        self.lock = Lock()
        self.history = []

    def __call__(self, inp: str):
        with self.lock:
            # Access the API key from the environment variable
            api_key = os.getenv(OPENAI_API_KEY_ENV_VAR)
            if not api_key:
                return [("Error", "API Key is not set in environment variables.")]

            openai.api_key = api_key

            # Append the new user message to the history
            self.history.append({"role": "user", "content": inp})

            # Send the input to the GPT API and append the response to the history
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=self.history
                )
                # Append the new assistant message to the history
                self.history.append(response["choices"][0]["message"])
                # Convert history to the expected list of tuples format
                return [(m["role"], m["content"]) for m in self.history]
            except Exception as e:
                return [("Error", str(e))]

# Instantiate the ChatWrapper
chat = ChatWrapper()

with gr.Blocks(title="LangChain Chatbot", description="A chatbot powered by LangChain") as block:
    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Type your message here...",
            lines=1,
            scale=1
        )

        submit = gr.Button(value="Send", variant="primary")

    # Event handlers for the UI elements
    submit.click(
        chat,
        inputs=[message],
        outputs=[chatbot]
    )
    message.submit(
        chat,
        inputs=[message],
        outputs=[chatbot]
    )

block.launch()
