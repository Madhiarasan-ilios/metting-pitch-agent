import json
import boto3
from dotenv import load_dotenv
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain_core.runnables import RunnableSequence
from langchain.prompts import PromptTemplate

# Load environment variables (if needed)
load_dotenv()


llm = ChatBedrock(
        model_id="meta.llama3-70b-instruct-v1:0",
        model_kwargs={"temperature": 0, "max_gen_len": 1024}
    )
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
    
def response(summary,documents):
    prompt = PromptTemplate(
    input_variables=["summary", "documents"],
    template="""
You are an academic course recommender. Based on the provided meeting summary and relevant course content, suggest up to three relevant courses that align with the topics discussed. Each course suggestion should include a course name and a brief description (20-30 words) of how it relates to the summary.

Input:
- Meeting Summary: {summary}
- Course Content: {documents}

Instructions:
1. Analyze the summary and course content to identify relevant academic topics.
2. Suggest up to three courses with names and brief descriptions.
3. Output a JSON array of objects, each with "course_name" and "description" keys.
4. Ensure suggestions are precise, relevant, and based on the provided content.
5. If no relevant courses can be suggested, return [{{"message": "nothing yet please wait till process"}}].
6. Do not include any explanations or extra text outside the JSON array.

[
    {{"course_name": "Course Name", "description": "Description of relevance"}},
    ...
]
"""
)

    chain = RunnableSequence(prompt | llm)
    result = chain.invoke({"summary": summary, "documents": documents})
    return result.content



# --- Run a test query ---
if __name__ == "__main__":
    test_query = "Here is a concise summary of the 1-minute transcript segment:\n\nThe speaker discusses the importance of learning to solve specific problems in data science, using a dataset on crime in the United States as an example. They highlight the skills needed to answer essential questions about differences in crime across states, covering topics such as functions, data types, vector operations, and advanced functional programming. The focus is on applying general programming features, data wrangling, analysis, and visualization to tackle real-world problems."
    documents=query_vector_db_mmr(query_text=test_query, k=5, fetch_k=15, lambda_param=0.7)
    response_data = response(test_query, documents)
    try:
        # Extract and parse LLM response into JSON
        json_data = json.loads(response_data.content if hasattr(response_data, "content") else str(response_data))
        print("Parsed JSON Response:\n", json.dumps(json_data, indent=2))
    except json.JSONDecodeError as e:
        print("Error parsing LLM output to JSON:", e)
        print("Raw Response:", response_data)