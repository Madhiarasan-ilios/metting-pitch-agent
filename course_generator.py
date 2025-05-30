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

OPENSEARCH_URL = "https://search-meeting-asst-caksjskstlseda5rwmizlllrva.ap-south-1.es.amazonaws.com"
OPENSEARCH_INDEX_NAME = "meeting-asst"
BEDROCK_MODEL_ID = "amazon.titan-embed-text-v2:0"

embeddings_model = BedrockEmbeddings(model_id=BEDROCK_MODEL_ID)


# Configure logging
logger = logging.getLogger("CourseGenerator")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

async def generate_course_suggestions(json_path: str = "output/summary.json") -> dict:
    logger.debug("===== AGENT START: COURSE GENERATION =====")

    # Load the latest summary
    logger.debug(f"Attempting to load summary from {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        overall_summary = data.get("overall", {}).get("summary", "")
        meeting_time = 0
        if overall_summary:
            logger.debug(f"Found overall summary: {overall_summary[:100]}...")
        else:
            minute_summaries = data.get("minute_summaries", [])
            if not minute_summaries:
                raise ValueError("No summaries found")
            latest_minute = max(entry["minute"] for entry in minute_summaries)
            overall_summary = next(
                entry["summary"] for entry in minute_summaries if entry["minute"] == latest_minute
            )
            meeting_time = latest_minute
            logger.debug(f"Using latest minute summary (minute {meeting_time}): {overall_summary[:100]}...")
    except Exception as e:
        logger.error(f"Error loading summary: {str(e)}")
        return {
            "summary": f"Error loading summary: {str(e)}",
            "meeting_time": -1,
            "retrieved_docs_count": 0,
            "course_suggestions": [{"message": "nothing yet please wait till process"}],
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

    # OpenSearch setup
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

    vector_store=OpenSearchVectorSearch(
                opensearch_url=OPENSEARCH_URL,
                index_name=OPENSEARCH_INDEX_NAME,
                embedding_function=embeddings_model,
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection
            )

    # Retrieve documents using MMR
    retrieved_docs = []
    try:
        logger.debug("Retrieving documents with MMR reranking...")
        results = vector_store.max_marginal_relevance_search(
            query=overall_summary,
            k=3,
            fetch_k=10,
            lambda_mult=0.5
        )
        retrieved_docs = [doc.page_content for doc in results]
        logger.debug(f"Retrieved {len(retrieved_docs)} documents")
        for i, doc in enumerate(retrieved_docs, 1):
            logger.debug(f"Document {i}: {doc[:100]}...")
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return {
            "summary": overall_summary,
            "meeting_time": meeting_time,
            "retrieved_docs_count": 0,
            "course_suggestions": [{"message": "nothing yet please wait till process"}],
            "error": f"Error retrieving documents: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }

    # LLM setup
    logger.debug("Initializing Meta Llama 3 70B")
    llm = ChatBedrock(
        model_id="meta.llama3-70b-instruct-v1:0",
        model_kwargs={"temperature": 0, "max_gen_len": 1024}
    )

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
5. If no relevant courses can be suggested, return [{"message": "nothing yet please wait till process"}].
6. Do not include any explanations or extra text outside the JSON array.

[
    {{"course_name": "Course Name", "description": "Description of relevance"}},
    ...
]
"""
    )

    try:
        async with asyncio.timeout(30):
            logger.debug("Generating course suggestions...")
            final_docs = " ".join(retrieved_docs).replace("\n", " ").replace("\r", " ") if retrieved_docs else ""
            chain = RunnableSequence(prompt | llm)
            result = await chain.ainvoke({"summary": overall_summary, "documents": final_docs})
            suggestions = json.loads(result.content.strip())
            logger.debug(f"Course suggestions: {suggestions}")
            if not suggestions:
                suggestions = [{"message": "nothing yet please wait till process"}]
    except asyncio.TimeoutError:
        logger.error("Course generation timed out after 30 seconds")
        suggestions = [{"message": "nothing yet please wait till process"}]
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        suggestions = [{"message": "nothing yet please wait till process"}]

    output = {
        "summary": overall_summary,
        "meeting_time": meeting_time,
        "retrieved_docs_count": len(retrieved_docs),
        "course_suggestions": suggestions,
        "error": "",
        "timestamp": datetime.utcnow().isoformat()
    }
    logger.debug(f"Final output: {json.dumps(output, indent=2)}")
    logger.debug("===== AGENT END: COURSE GENERATION =====")
    return output