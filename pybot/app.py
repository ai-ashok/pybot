import os

from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl


faiss_db_path = "vectordb/python"
device = os.environ["DEVICE"]


custom_prompt_template = """Use the following pieces of information to answer the user's question.

Context: {context}
Question: {question}

If you know the answer, proceed with answer. If you don't know the answer, please just say, `Sorry. I don't know the answer for your query.` and don't try to make up the answer.
Answer: 
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores.
    """

    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )

    return prompt


def load_llm():
    llm = CTransformers(
        model="models/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.5,
    )
    return llm


def retrieval_qa_chain(llm: CTransformers, prompt: PromptTemplate, db: FAISS):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device})
    db = FAISS.load_local(folder_path=faiss_db_path, embeddings=embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm=llm, prompt=qa_prompt, db=db)
    return qa


def result(query: str):
    qa_result = qa_bot()
    response = qa_result(inputs={"query": query})
    return response


@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting PyBot...")
    await msg.send()
    msg.content = "Hi! I'm PyBot."
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: str):
    chain: RetrievalQA = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]

    await cl.Message(content=answer).send()
