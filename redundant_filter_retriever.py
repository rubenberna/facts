# Create a customer redundant_filter_retriever class. Chroma has one, but always recalculates the embeddings, which is not really necessary, so we create a new one.
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever

class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings # Class definition that says that to call this class, you need to pass an an object that can be used to calculate embeddings
    chroma: Chroma # Same for a Chroma instance. This means that instead of creating a new instance of Chroma, we can pass an existing
    def get_relevant_documents(self, query: str):
        # Calculate embeddings for the query string
        # Take embeddings and feed them into max_marginal_relevance_search_by_vector -- this method will retrieve the list of documents and remove duplicates for the Chain's LLM
        emb = self.embeddings.embed_query(query)
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8, # Defines our control over very similar documents
        )
       
    
    async def get_relevant_documents_async(self, query: str):
        # This is the method is mandatory, but not used in the chain
        return[]
