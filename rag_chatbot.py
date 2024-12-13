from langchain.chains import RetrievalQA
# from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from langchain_chroma import Chroma
import streamlit as st
import time

def llm_prepared(model, tokenizer):
    pipe = pipeline(
        "text2text-generation",
        model= model,
        tokenizer=tokenizer,
        max_length=837,
        # temperature=0,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def load_embedding_model(path):
    # embedding
    model_name = "BAAI/bge-base-en"
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs=encode_kwargs
    )

    # Load from disk
    vectordb = Chroma(persist_directory=path, embedding_function=embedding)
    return vectordb 

def get_response(text, db, llm):
    # make a retriver
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # create the chain to answer questions
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    return_source_documents=True)
    response = qa_chain(text)
    response = process_llm_response(response)
    return response

def process_llm_response(llm_response):
    data = llm_response['result']+"\n"
    # data = data+"Resources: \n"
    data = data+"Result: \n"
    # print(llm_response['result'])
    # print('\n\nRESULT:')
    if(llm_response['result'] == "Not enough information"):
        return data
    
    for source in llm_response["source_documents"]:
        # print(source.metadata['MODEL']+': '+source.metadata['LINK'])
        data = data+source.metadata['MODEL']+': '+source.metadata['LINK']+'\n'
        # data = data+source.metadata['LINK']+'\n'
        # break
    return data

def generator(text):
    for word in text.split("\n"):
        print(word)
        yield word + "\n"
        time.sleep(0.05)

def main():
    st.title("Assistant System")
    # Choose one of LLMs, in this experiment I use flan-t5
    model_path = "google/flan-t5-base"
    # Specify the embedding database path
    embedding_path = "db"

    # Load/download the language model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map='auto')

    # Setup llm
    llm = llm_prepared(model, tokenizer)

    # Load embedding
    db = load_embedding_model(embedding_path)

    # Accept user input
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response_text = get_response(prompt, db, llm)
            # print(response_text)
            response = st.write_stream(generator(response_text))
            # response = st.write(response_text)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()