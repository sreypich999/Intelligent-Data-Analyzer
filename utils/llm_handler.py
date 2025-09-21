import os
import logging
import warnings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

from utils.config import GEMINI_MODEL_NAME, DEFAULT_TEMPERATURE, DEFAULT_TOP_P

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMHandler:
    """Handles interactions with Google Gemini LLM."""
    def __init__(self, temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is not set in .env file.")
        
        # Suppress specific warnings
        warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")
        
        self.model = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            temperature=temperature,
            top_p=top_p,
            google_api_key=self.api_key,
            convert_system_message_to_human=True
        )
        self.temperature = temperature
        self.top_p = top_p
        logging.info(f"LLMHandler initialized with {GEMINI_MODEL_NAME} (temp={temperature}, top_p={top_p}).")

    def update_model_params(self, temperature: float = None, top_p: float = None):
        """Updates the LLM's generation parameters."""
        if temperature is not None:
            self.temperature = temperature
            self.model.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
            self.model.top_p = top_p
        logging.info(f"LLM params updated: temp={self.temperature}, top_p={self.top_p}")

    def get_model(self):
        return self.model

    def get_response(self, prompt: str, chat_history: list = None, system_message: str = None):
        """Generates a streaming response from the LLM."""
        messages_list = []
        if system_message:
            messages_list.append(SystemMessage(content=system_message))
        if chat_history:
            messages_list.extend(chat_history)
        messages_list.append(("human", prompt))

        try:
            prompt_template = ChatPromptTemplate.from_messages(messages_list)
            chain = prompt_template | self.model
            for chunk in chain.stream({}):
                yield chunk.content
        except Exception as e:
            logging.error(f"Error getting LLM response: {e}")
            yield f"I'm sorry, I encountered an error: {e}"