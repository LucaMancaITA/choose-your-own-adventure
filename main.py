import os
from dotenv import load_dotenv

from langchain_astradb import AstraDBChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

import gradio as gr


# Load environment variables
load_dotenv()

# Message history
message_history = AstraDBChatMessageHistory(
    session_id="test-session",
    api_endpoint=os.environ["ASTRA_DB_ENDPOINT"],
    token=os.environ["ASTRA_DB_TOKEN"]
)
message_history.clear()

# Conversation buffer memory
cass_buff_memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=message_history
)

# Prompt template
template = """
You are now the guide of a mystical journey in the Whispering Woods.
A traveler named Elara seeks the lost Gem of Serenity.
You must navigate her through challenges, choices, and consequences,
dynamically adapting the tale based on the traveler's decisions.
Your goal is to create a branching narrative experience where each choice
leads to a new path, ultimately determining Elara's fate.

Here are some rules to follow:
1. Start by asking the player to choose some kind of weapons that will be used later in the game
2. Have a few paths that lead to success
3. Have some paths that lead to death. If the user dies generate a response that explains the death and ends in the text: "The End.", I will search for this text to end the game

Here is the chat history, use this to understand what to say next: {chat_history}
Human: {human_input}
AI:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template
)

# LLama2 LLM
llm = Ollama(model="llama2")
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=cass_buff_memory
)

# Initialize chatbot with the first message
choice = "start"
initial_bot_message = llm_chain.predict(human_input=choice).strip()

# Gradio Chatbot
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(value=[[None, initial_bot_message]])
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(choice, chat_history):
        response = llm_chain.predict(human_input=choice).strip()
        chat_history.append((choice, response))
        return "", chat_history

    msg.submit(
        fn=respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )


if __name__ == "__main__":
    demo.launch()
