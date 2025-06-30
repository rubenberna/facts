from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# 0. Load the embeddings
embeddings = OpenAIEmbeddings()

# example of embedding a query
# emb = embeddings.embed_query("What is the capital of France?")
# print(emb)

# 1. Creaate chunks of text
text_splitter = CharacterTextSplitter(
    separator="\n", # What character to split the text on
    chunk_size=200, # How many characters in each chunk
    chunk_overlap=0 # How many characters to overlap between chunks
)

loader = TextLoader("facts.txt")

docs = loader.load_and_split(
    text_splitter=text_splitter
)

# 2. Store the chunks in the vector store
# We create a chroma instance and initiate it with the documents. It will immediately embed the documents and store them in the vector store.
#Attention: everytime you run this code, it will a copy of the data, so in the tutorial, we use the prompt.py from this point on

db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)

# 3. Query the vector store
# With_score brings tupples with similarity score
results = db.similarity_search_with_score("What is an interesting fact about the English language?", k=1) # The k=1 argument means we only want the most similar document (the one with the highest score)

for result in results:
    print("\n")
    print(result[1])
    print(result[0].page_content)
