import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Directly set the API key (replace with your actual API key)
openai_api_key = "insert open Api key"

# Initialize the OpenAI language model with the API key
llm = OpenAI(temperature=0.9, max_tokens=500, openai_api_key=openai_api_key)

# Set up Streamlit interface
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Input section for URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# Button to process URLs
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"  # File path to store serialized FAISS index

main_placeholder = st.empty()  # Main content placeholder

# Process URLs if the button is clicked
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split data into smaller documents
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings from documents and build FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    pkl = vectorstore_openai.serialize_to_bytes()
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)  # Simulate processing time

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(pkl, f)

# Input field for user query
query = main_placeholder.text_input("Question: ")

# If a query is entered, process it
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            pkl = pickle.load(f)
            # Deserialize FAISS index and create a retrieval question-answering chain
            vectorstore = FAISS.deserialize_from_bytes(embeddings=OpenAIEmbeddings(), serialized=pkl, allow_dangerous_deserialization=True)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            # Display the answer and sources
            st.header("Answer")
            st.write(result["answer"])

            # Display sources if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
