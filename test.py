import os
import boto3
from dotenv import load_dotenv
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_aws.embeddings import BedrockEmbeddings

# Load environment variables (if needed)
load_dotenv()

# Constants
BEDROCK_MODEL_ID = "amazon.titan-embed-text-v2:0"
OPENSEARCH_URL = "https://search-meeting-asst-caksjskstlseda5rwmizlllrva.ap-south-1.es.amazonaws.com"
OPENSEARCH_INDEX_NAME = "meeting-asst"
OPENSEARCH_BULK_SIZE = 2500
REGION = "ap-south-1"
SERVICE = "es"

# AWS authentication
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    REGION,
    SERVICE,
    session_token=credentials.token
)

def query_vector_db_mmr(query_text: str, k: int = 5, fetch_k: int = 20, lambda_param: float = 0.5):
    """
    Queries the OpenSearch vector database using Maximal Marginal Relevance (MMR).

    Args:
        query_text (str): The user query to retrieve similar documents.
        k (int): Number of top diverse results to return.
        fetch_k (int): Total number of candidates to fetch before applying MMR.
        lambda_param (float): Controls trade-off between relevance and diversity (0 = diverse, 1 = relevant).

    Returns:
        List of top-k matched documents with metadata using MMR.
    """
    try:
        # Initialize Bedrock Embeddings
        embeddings_model = BedrockEmbeddings(model_id=BEDROCK_MODEL_ID)

        # Connect to OpenSearch vector store
        docsearch = OpenSearchVectorSearch(
            opensearch_url=OPENSEARCH_URL,
            index_name=OPENSEARCH_INDEX_NAME,
            embedding_function=embeddings_model,
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            bulk_size=OPENSEARCH_BULK_SIZE,
            connection_class=RequestsHttpConnection
        )

        # Perform MMR search
        results = docsearch.max_marginal_relevance_search(
            query=query_text,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_param
        )

        print(f"\nTop {k} MMR Results for Query: \"{query_text}\"\n")
        for i, doc in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"Text Chunk:\n{doc.page_content}\n")
            print(f"Metadata: {doc.metadata}\n{'-'*60}")

        return results

    except Exception as e:
        print(f"Error querying vector database with MMR: {e}")
        return []

# --- Run a test query ---
if __name__ == "__main__":
    test_query = "ai and llms"
    query_vector_db_mmr(query_text=test_query, k=5, fetch_k=15, lambda_param=0.7)
