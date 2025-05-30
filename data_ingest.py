import pandas as pd
import os
import boto3
from dotenv import load_dotenv
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

CSV_FILE_PATH = 'edx_courses.csv'

TEXT_COLUMNS = ['title', 'summary', 'course_type']

METADATA_COLUMNS = ['subject','language']

BEDROCK_MODEL_ID = "amazon.titan-embed-text-v2:0"

OPENSEARCH_URL = "https://search-meeting-asst-caksjskstlseda5rwmizlllrva.ap-south-1.es.amazonaws.com"
OPENSEARCH_INDEX_NAME = "meeting-asst"
OPENSEARCH_BULK_SIZE = 2500 

region = "ap-south-1"
service = "es"

credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    service,
    session_token=credentials.token
)

def prepare_documents_for_ingestion(csv_path: str) -> list[Document]:
    """
    Loads the EdX courses CSV, combines text columns, and prepares
    LangChain Document objects for ingestion.

    Args:
        csv_path (str): Path to the edx_courses.csv file.

    Returns:
        list[Document]: A list of LangChain Document objects.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at '{csv_path}'")
        return []

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")

    # Filter out columns that don't exist in the DataFrame before dropping NA
    existing_text_columns = [col for col in TEXT_COLUMNS if col in df.columns]
    df.dropna(subset=existing_text_columns, inplace=True)
    print(f"After dropping rows with missing text, {len(df)} rows remain.")

    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    for index, row in df.iterrows():
        combined_text = ""
        for col in TEXT_COLUMNS:
            if col in row and pd.notna(row[col]):
                # Add column name as a prefix for better context in embeddings
                combined_text += f"{col.replace('_', ' ').upper()}: {row[col]}. "

        if not combined_text.strip():
            print(f"Skipping row {index} due to empty combined text after cleaning.")
            continue

        # Prepare metadata for the Document object
        metadata = {}
        for col in METADATA_COLUMNS:
            if col in row and pd.notna(row[col]):
                # Convert non-string metadata to string if necessary for OpenSearch
                metadata[col] = str(row[col])
            else:
                metadata[col] = None

        # Add course_id to metadata for easy retrieval, ensuring it's a string
        if 'course_id' in row and pd.notna(row['course_id']):
            metadata['course_id'] = str(row['course_id'])

        # Split the combined text into chunks
        chunks = text_splitter.split_text(combined_text.strip())

        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = i
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))

    print(f"Prepared {len(documents)} document chunks for ingestion.")
    return documents

# --- Execution ---
if __name__ == "__main__":
    # Prepare documents from the CSV
    documents_to_ingest = prepare_documents_for_ingestion(CSV_FILE_PATH)

    if not documents_to_ingest:
        print("No documents to ingest. Exiting.")
    else:
        # Initialize Bedrock Embeddings
        try:
            embeddings_model = BedrockEmbeddings(model_id=BEDROCK_MODEL_ID)
            print(f"BedrockEmbeddings initialized with model ID: {BEDROCK_MODEL_ID}")
        except Exception as e:
            print(f"Error initializing BedrockEmbeddings: {e}")
            print("Please ensure your AWS credentials are configured and the model ID is correct.")
            exit()

        # Ingest data into OpenSearch
        try:
            print(f"Attempting to ingest data into OpenSearch index: {OPENSEARCH_INDEX_NAME}")
            print(f"Connecting to OpenSearch at: {OPENSEARCH_URL}")

            # Initialize OpenSearchVectorSearch client
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

            # Add documents to the OpenSearch index
            docsearch.add_documents(documents_to_ingest)
            print("Documents added to OpenSearch index successfully!")
        except Exception as e:
            print(f"Error during OpenSearch ingestion: {e}")
            print("Please check your OpenSearch URL, AWS credentials, and network connectivity.")

