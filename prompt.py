from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
from dotenv import load_dotenv
import langchain

langchain.debug = True

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()
db = Chroma(
    persist_directory="emb", # where all the files in sqlite is stored
    embedding_function=embeddings,
)

# The RetrievalQA chain is a wrapper around the Chroma vector store that allows you to query it with a question and get a response. It's essetially a chain that gets a human message, embeds it, queries the vector store for the most similar document and sends it to an LLM. It does all of this in one go, instead of us having to create these chains manually.


# Use the db as a retriever, specifically the 'get_relevant_documents' method, that is inherently part of the Chroma class and passed tot the chain
# retriever = db.as_retriever() 

# We're creating a custom retriever: it will have a function to find relevant documents in Chroma and remove duplicate records
retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff" # Stuff is actually the default chain_type. Other types are possible, but for more complex cases and this one is used in Production applications
)

result = chain.run("What is an interesting fact about the English language?")

print(result)