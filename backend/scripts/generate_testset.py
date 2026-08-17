import json
import logging
import os
import sys
import uuid

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.ragas import RagasTestset
from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document as LCDocument
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from sqlalchemy.orm import Session

# Ensure the backend directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv(find_dotenv())


# ---------------------------------------------------------------------------
# Configuration Variables
# ---------------------------------------------------------------------------
TENANT_ID = "c2a35f4c-4a7d-4084-939a-f8cadd71045d"
DOCUMENT_IDS = [
    "7dea2ab8-d828-4ad9-858d-e8022db7e111",
    "35f48b56-7652-434a-92ac-222b860023ec",
    "060fc824-ce56-4052-89d5-0f7f85904742",
    "7b6582d6-72f8-4fdb-954e-1569cba81ca6",
    "d7072852-817e-420f-a5f1-06be25e7600d",
]  # list of document UUIDs to use, leave empty to use all tenant documents
NUM_SAMPLES = 30  # number of testset samples to generate
TESTSET_OUTPUT_PATH = "scripts/testset.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("generate_testset")


def fetch_chunks_from_qdrant(tenant_id: str, document_ids: list) -> list[LCDocument]:
    """
    Fetch document chunks from Qdrant for the specified tenant and document IDs.
    If document_ids is empty, fetch all chunks for the tenant collection.
    """
    collection_name = f"tenant_{tenant_id}"
    logger.info(f"Connecting to Qdrant at {settings.QDRANT_URL}...")
    qdrant_client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )

    # Check if collection exists
    existing_collections = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name not in existing_collections:
        raise ValueError(
            f"Qdrant collection '{collection_name}' not found for tenant '{tenant_id}'."
        )

    scroll_filter = None
    if document_ids:
        doc_id_strs = [str(doc_id) for doc_id in document_ids]
        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchAny(any=doc_id_strs),
                )
            ]
        )
        logger.info(
            f"Filtering chunks for {len(doc_id_strs)} document ID(s): {doc_id_strs}"
        )
    else:
        logger.info(
            f"No document IDs specified; fetching all chunks for tenant '{tenant_id}'."
        )

    chunks: list[LCDocument] = []
    offset = None

    while True:
        records, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for record in records:
            payload = record.payload or {}
            text = (
                payload.get("chunk_text")
                or payload.get("text")
                or payload.get("content")
                or ""
            )
            if text.strip():
                metadata = {
                    "document_id": str(payload.get("document_id", "")),
                    "chunk_index": payload.get("chunk_index"),
                    "page_number": payload.get("page_number"),
                    "slide_number": payload.get("slide_number"),
                    "file_type": str(payload.get("file_type", "")),
                }
                chunks.append(LCDocument(page_content=text, metadata=metadata))

        if next_offset is None:
            break
        offset = next_offset

    logger.info(
        f"Retrieved {len(chunks)} chunk(s) from Qdrant collection '{collection_name}'."
    )
    return chunks


def convert_testset_to_dicts(testset) -> list[dict]:
    """
    Convert RAGAS generated testset into a list of dicts with keys:
    question, ground_truth, reference_contexts.
    """
    raw_list = []
    if hasattr(testset, "to_list") and callable(testset.to_list):
        try:
            raw_list = testset.to_list()
        except Exception as e:
            logger.debug(f"testset.to_list() failed: {e}")

    if not raw_list and hasattr(testset, "to_pandas") and callable(testset.to_pandas):
        try:
            df = testset.to_pandas()
            raw_list = df.to_dict(orient="records")
        except Exception as e:
            logger.debug(f"testset.to_pandas() failed: {e}")

    if not raw_list and hasattr(testset, "samples"):
        raw_list = testset.samples
    elif not raw_list:
        try:
            raw_list = list(testset)
        except Exception:
            raw_list = []

    samples = []
    for item in raw_list:
        if isinstance(item, dict):
            question = item.get("question") or item.get("user_input") or ""
            ground_truth = item.get("ground_truth") or item.get("reference") or ""
            contexts = (
                item.get("reference_contexts")
                or item.get("contexts")
                or item.get("retrieved_contexts")
                or []
            )
        else:
            question = (
                getattr(item, "question", None) or getattr(item, "user_input", "") or ""
            )
            ground_truth = (
                getattr(item, "ground_truth", None)
                or getattr(item, "reference", "")
                or ""
            )
            contexts = (
                getattr(item, "reference_contexts", None)
                or getattr(item, "contexts", None)
                or getattr(item, "retrieved_contexts", None)
                or []
            )

        if isinstance(contexts, str):
            contexts = [contexts]
        elif not isinstance(contexts, list):
            contexts = list(contexts) if contexts is not None else []

        reference_contexts = [str(c) for c in contexts]

        samples.append(
            {
                "question": str(question),
                "ground_truth": str(ground_truth),
                "reference_contexts": reference_contexts,
            }
        )

    return samples


def save_testset_to_db(samples: list[dict], tenant_id: str) -> None:
    """
    Save generated testset samples into Postgres ragas_testset table.
    """
    db: Session = SessionLocal()
    tenant_uuid = uuid.UUID(str(tenant_id))

    try:
        logger.info(
            f"Saving {len(samples)} sample(s) to 'ragas_testset' table for tenant {tenant_id}..."
        )
        for item in samples:
            testset_row = RagasTestset(
                id=uuid.uuid4(),
                tenant_id=tenant_uuid,
                question=item["question"],
                ground_truth=item["ground_truth"],
                reference_contexts=item["reference_contexts"],
            )
            db.add(testset_row)
        db.commit()
        logger.info("Successfully committed testset samples to Postgres database.")
    except Exception as e:
        db.rollback()
        logger.error(f"Database error while saving testset: {e}", exc_info=True)
        raise
    finally:
        db.close()


def export_testset_to_json(samples: list[dict], output_path: str) -> None:
    """
    Export the full testset to a JSON file as backup.
    """
    target_path = output_path
    if not os.path.isabs(target_path):
        # Resolve relative to current working directory or backend directory
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target_path = os.path.join(backend_dir, target_path)

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    logger.info(f"Exported testset JSON backup to: {target_path}")


def main():
    logger.info("=== Starting RAGAS Testset Generation Pipeline ===")

    # Step 1: Validate Tenant ID
    if not TENANT_ID or TENANT_ID == "your-tenant-uuid-here":
        logger.error(
            "Please set a valid TENANT_ID at the top of scripts/generate_testset.py before running."
        )
        sys.exit(1)

    try:
        uuid.UUID(str(TENANT_ID))
    except ValueError:
        logger.error(f"Invalid UUID format for TENANT_ID: '{TENANT_ID}'")
        sys.exit(1)

    # Step 2: Fetch Chunks from Qdrant
    print("\n")
    logger.info("Step 1/5: Fetching document chunks from Qdrant...")
    chunks = fetch_chunks_from_qdrant(tenant_id=TENANT_ID, document_ids=DOCUMENT_IDS)
    if not chunks:
        logger.error("No chunks retrieved from Qdrant. Aborting testset generation.")
        sys.exit(1)

    # Step 3: Configure Generator LLM and Embeddings
    print("\n")
    logger.info("Step 2/5: Initializing LLM and Embedding models...")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY environment variable is not set.")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        logger.warning("GOOGLE_API_KEY environment variable is not set.")

    generator_llm = LangchainLLMWrapper(
        ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=anthropic_api_key,
        )
    )

    generator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=google_api_key or os.getenv("GEMINI_API_KEY"),
        )
    )

    # Step 4: Initialize TestsetGenerator and Generate
    print("\n")
    logger.info(
        f"Step 3/5: Initializing TestsetGenerator and generating {NUM_SAMPLES} samples..."
    )
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
    )

    try:
        testset = generator.generate_with_chunks(
            chunks=chunks,
            testset_size=NUM_SAMPLES,
        )
    except AttributeError:
        # Fallback to generate_with_langchain_docs if available
        testset = generator.generate_with_langchain_docs(
            documents=chunks,
            testset_size=NUM_SAMPLES,
        )

    # Step 5: Convert and Save Samples
    print("\n")
    logger.info("Step 4/5: Formatting generated testset samples...")
    samples = convert_testset_to_dicts(testset)
    logger.info(f"Generated {len(samples)} valid testset sample(s).")

    if not samples:
        logger.warning("No samples were generated by RAGAS.")
        return

    print("\n")
    logger.info("Step 5/5: Persisting testset to Postgres and JSON backup...")
    save_testset_to_db(samples=samples, tenant_id=TENANT_ID)
    export_testset_to_json(samples=samples, output_path=TESTSET_OUTPUT_PATH)

    logger.info("=== RAGAS Testset Generation Completed Successfully ===")


if __name__ == "__main__":
    main()
