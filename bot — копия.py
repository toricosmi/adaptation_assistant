import json
import datetime
import requests
from telegram import Update, Document

from telegram import Update
from telegram.ext import (
    Application,
    Updater,
    CommandHandler,
    MessageHandler,
    CallbackContext,
)
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from telegram.ext import filters

import os

from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("HF_API_TOKEN")
# os.environ['TG_TOKEN'] = 'TG_TOKEN'
TG_TOKEN = os.getenv("TG_TOKEN")
API_URL = (
    "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
)

# Настройки ограничений запросов
user_requests = {}
MAX_REQUESTS_PER_DAY = 50
PATH_INDEX = "faiss_index"
embeddings = HuggingFaceEmbeddings(model_name="all-MinilM-L6-v2")

def load_vector_store(embeddings, vector_store_path="faiss_index"):
    vector_store = FAISS.load_local(
        folder_path=vector_store_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store


def create_prompt(student_query: str, relevant_docs: list[Document]) -> str:
    docs_summary = "".join(
        f"[Источник: {doc.metadata['source']}]\n{doc.page_content}\n\n"
        for doc in relevant_docs
    )
    prompt = f"""
Ты — корпоративный ассистент. Отвечай на вопросы стажёров чётко, ясно и по делу, используя приведённые внутренние материалы. 

1. Вопрос от сотрудника:
{student_query}

2. Внутренние справочные материалы:
{docs_summary}

3. Сформулируй понятный и точный ответ, который:
- Прямо отвечает на вопрос
- Может использовать информацию из базы знаний
- Излагается на русском языке
- Логично структурирован (по пунктам, если нужно)

Если вопрос не по теме или отсутствует информация — вежливо скажи, что это не твоя задача.
"""
    return prompt.strip()


# Получение ответа от языковой модели
def get_assistant_response(student_query, vector_store, api_url, api_token):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(student_query)
    prompt = create_prompt(student_query, relevant_docs)

    headers = {
        "Authorization": f"Bearer {api_token}"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 2000,
            "temperature": 0.5,
            "num_return_sequences": 1
        }
    }

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        output = response.json()
        return output[0]["generated_text"][len(prompt)+1:]
    else:
        raise ValueError(f"Ошибка API: {response.status_code}, {response.text}")


# Основная логика Telegram-бога
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "Привет! Я ваш ассистент. Задайте свой вопрос!"
    )  


async def handle_message(
    update: Update, context: CallbackContext, path_index: str = PATH_INDEX
):
    student_query = update.message.text

    try:
        vector_store = load_vector_store(embeddings, path_index)
    except Exception as e:
        await update.message.reply_text("Произошла ошибка при загрузке данных. Попробуйте позже.")
        return

    try:
        response = get_assistant_response(
            student_query, vector_store, api_url=API_URL, api_token=API_TOKEN
        )
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"Произошла ошибка: {e}")


def main():
    app = Application.builder().token(TG_TOKEN).build()

    # Добавление обработчиков
    app.add_handler(CommandHandler("start", start))
    # Обрабатываем текстовые сообщения, исключая команды
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск бота
    app.run_polling()


if __name__ == "__main__":
    main()
