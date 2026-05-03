from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def load_vector_db(config):
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1024
    )

    vectordb = Chroma(
        persist_directory=config['pipeline']['vector_db_path'],
        embedding_function=embedding_model,
        collection_name="tesla_financials"  
    )

    return vectordb


def get_retriever(config, k=5):
    vectordb = load_vector_db(config)
    print("Vector count:", vectordb._collection.count())
    return vectordb.as_retriever(search_kwargs={"k": k })