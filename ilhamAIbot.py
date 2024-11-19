import os
import telebot
import gradio as gr
import random
import time
from langchain_community.llms import Ollama
from langchain_nomic import NomicEmbeddings
from langchain_community.vectorstores import Chroma
import cohere
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import cohere

def get_embedding_function():
    embeddings = NomicEmbeddings(
    model='nomic-embed-text-v1.5',
    inference_mode='local',
    device='cuda',
)
    return embeddings
embedding_function = get_embedding_function()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]


CHROMA_PATH = "/home/hendry/johor/johor_chroma"
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
co = cohere.Client("Bj0lFx0ZryIDA6PTnSIHvE5filfO2kxcF7UTtKBK")
retriever = db.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":20}
)


llm = Ollama(model = "mistral")

retriever = db.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":30}
)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt= ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

def rerank_relevance(reformulated_question,list_document):
    response_rerank = co.rerank(
                                model="rerank-multilingual-v3.0",
                                query=reformulated_question,
                                documents=list_document,
                                return_documents=True
                            )
    return [i.document.text for i in response_rerank.results[0:15] ]


qa_system_prompt = """You are an assistant for question-answering tasks. \
                    Use the following pieces of retrieved context to answer the question. \
                    If you don't know the answer, just say that you don't know. \
                    Use three sentences maximum and keep the answer concise.\
                    If you cant give the direct answer, Please give a direct answer instead of beating around the bush and then answering the question.\

                    {context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="context"),
        ("human", "{question}"),
    ]
)

prompt_context_question = qa_prompt | llm | StrOutputParser()




def get_llm_response(history):
    reformulated_question = contextualize_q_chain.invoke({"chat_history":history,
                                        "question": history[-1]['content']})

    print(f"Reformulated Question: {reformulated_question}")
    retrieved_documents = retriever.get_relevant_documents(reformulated_question)
    documents_for_rerank  = [ {"text": i.page_content} for i in retrieved_documents ]
    response_rerank = rerank_relevance(reformulated_question,documents_for_rerank)
    response = prompt_context_question.invoke({"context":response_rerank,
                                 "question":reformulated_question})



    return response


def test1(question):
    response = prompt_context_question.invoke({"context":[],"question":question})
    return response


BOT_TOKEN = '7922832036:AAG_f292lTF9AhcygNuauQW0qnUU_HF6utg'

bot = telebot.TeleBot(BOT_TOKEN)


global history
history = []

@bot.message_handler(commands=['hello','hi'])
def send_welcome(message):
    # bot.reply_to(message, "Knowladge Restart")
    bot.reply_to(message, "Hi I'm Ilham your AI Assistant. how can i help you?")


def reset(message,history):
    return []
    # bot.reply_to(message,"Knowladge Restart")
    



@bot.message_handler(func=lambda msg: True)
def echo_all(message):
    global history
    question = message.text
    if question =="/reset":
        history_reset = reset(message,history)
        history = history_reset
        response_bot = "Knowladge Restart"

    else :
        history.append({'role':'user', 'content': question})
        response_bot =get_llm_response(history)
        history.append({'role':'user', 'content': response_bot})
    print(history)
    bot.reply_to(message, response_bot)

bot.infinity_polling()