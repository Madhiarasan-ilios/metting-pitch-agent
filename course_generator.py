import json
import boto3
import logging
import asyncio
from datetime import datetime
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import re

OPENSEARCH_URL = "https://search-meeting-asst-caksjskstlseda5rwmizlllrva.ap-south-1.es.amazonaws.com"
OPENSEARCH_INDEX_NAME = "meeting-asst"
BEDROCK_MODEL_ID = "amazon.titan-embed-text-v2:0"

embeddings_model = BedrockEmbeddings(model_id=BEDROCK_MODEL_ID)

logger = logging.getLogger("CourseGenerator")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

async def generate_course_suggestions(json_path: str = "output/summary.json") -> dict:
    logger.debug("===== AGENT START: COURSE GENERATION =====")

    logger.debug(f"Loading summary file from {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        minute_summaries = data.get("minute_summaries", [])
        if not minute_summaries:
            raise ValueError("No minute_summaries found in JSON")

        latest_entry = max(minute_summaries, key=lambda x: x["minute"])
        summary = latest_entry["summary"]
        meeting_time = latest_entry["minute"]
        logger.debug(f"Using latest minute summary (minute {meeting_time}): {summary[:100]}...")

    except Exception as e:
        logger.error(f"Error loading summary: {e}")
        return {
            "summary": "",
            "meeting_time": -1,
            "retrieved_docs_count": 0,
            "course_suggestions": [{"course_name": "", "description": "nothing yet please wait till process"}],
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

    region = 'ap-south-1'
    service = 'es'
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token
    )

    vector_store = OpenSearchVectorSearch(
        opensearch_url=OPENSEARCH_URL,
        index_name=OPENSEARCH_INDEX_NAME,
        embedding_function=embeddings_model,
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )

    retrieved_docs = []
    try:
        logger.debug("Retrieving documents with MMR reranking...")
        results = vector_store.max_marginal_relevance_search(
            query=summary,
            k=3,
            fetch_k=10,
            lambda_mult=0.5
        )
        retrieved_docs = [doc.page_content for doc in results]
        logger.debug(f"Retrieved {len(retrieved_docs)} documents")
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return {
            "summary": summary,
            "meeting_time": meeting_time,
            "retrieved_docs_count": 0,
            "course_suggestions": [{"course_name": "", "description": "nothing yet please wait till process"}],
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

    llm = ChatBedrock(
        model_id="meta.llama3-70b-instruct-v1:0",
        model_kwargs={"temperature": 0, "max_gen_len": 1024}
    )

    prompt = PromptTemplate(
        input_variables=["summary", "documents"],  # NO 'message' here
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
5. If no relevant courses can be suggested, return an empty array [].
6. Do not include any explanations or extra text outside the JSON array.

Example output:
[
    {{
        "course_name": "Introduction to AI",
        "description": "Covers basics of AI including machine learning and neural networks relevant to the meeting topics."
    }},
    ...
]
"""
    )

    try:
        logger.debug("Generating course suggestions...")
        docs_text = " ".join(retrieved_docs).replace("\n", " ").replace("\r", " ")
        chain = RunnableSequence(prompt | llm)
        result = await chain.ainvoke({"summary": summary, "documents": docs_text})
        logger.debug(f"LLM output: {result}")

        # Clean any markdown code blocks if present
        cleaned_content = re.sub(r'```json', '', result.content.strip())
        cleaned_content = re.sub(r'```', '', cleaned_content).strip()
        logger.debug(f"Cleaned LLM output: {cleaned_content}")

        suggestions = json.loads(cleaned_content)
        if not suggestions:
            suggestions = [{"course_name": "", "description": "nothing yet please wait till process"}]
    except Exception as e:
        logger.error(f"Error generating suggestions: {e}")
        suggestions = [{"course_name": "", "description": "nothing yet please wait till process"}]

    output = {
        "summary": summary,
        "meeting_time": meeting_time,
        "retrieved_docs_count": len(retrieved_docs),
        "course_suggestions": suggestions,
        "error": "",
        "timestamp": datetime.utcnow().isoformat()
    }
    logger.debug(f"Final output: {json.dumps(output, indent=2)}")
    logger.debug("===== AGENT END: COURSE GENERATION =====")
    return output
