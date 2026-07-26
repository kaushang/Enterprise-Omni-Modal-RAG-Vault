"""
RAG (Retrieval-Augmented Generation) service.

Handles:
  - Excel query execution in a RestrictedPython sandbox
  - Full RAG pipeline: embed query → Qdrant search → Excel pipeline → LLM answer

"""

import asyncio
import logging
import threading
import uuid
from typing import Optional, AsyncGenerator
import pandas as pd
from RestrictedPython import compile_restricted, safe_globals
from sentence_transformers import CrossEncoder
from sqlalchemy.orm import Session

from anthropic import Anthropic, AsyncAnthropic

from app.core.config import settings
from app.models.document import Document
from app.models.enums import (
    FileType,
    EXTENSION_TO_FILE_TYPE,
    TABULAR_FILE_TYPES,
)
from app.services.document_processor import load_dataframe
from app.models.user import User
from app.services.qdrant_service import search_vectors
from app.services.storage_service import get_absolute_path
import json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Anthropic client (ACTIVE)
# ---------------------------------------------------------------------------

_anthropic_client: Anthropic | None = None
_async_anthropic_client: AsyncAnthropic | None = None


def _get_anthropic_client() -> Anthropic:
    """Lazily initialise and return the Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _anthropic_client


def _get_async_anthropic_client() -> AsyncAnthropic:
    """Lazily initialise and return the AsyncAnthropic client."""
    global _async_anthropic_client
    if _async_anthropic_client is None:
        _async_anthropic_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _async_anthropic_client


# ---------------------------------------------------------------------------
# CrossEncoder model (ACTIVE)
# ---------------------------------------------------------------------------

_cross_encoder: Optional["CrossEncoder"] = None


def _get_cross_encoder() -> "CrossEncoder":
    """Lazily load and cache the CrossEncoder model."""
    global _cross_encoder
    if not settings.ENABLE_CROSS_ENCODER_RERANKING:
        raise RuntimeError("Cross-encoder re-ranking is disabled via config.")
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading CrossEncoder model: cross-encoder/ms-marco-MiniLM-L-6-v2")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


# ---------------------------------------------------------------------------
# Excel query execution
# ---------------------------------------------------------------------------

_EXCEL_CODE_PROMPT = """You have access to a pandas dataframe called `df` with the following schema:
Columns: {columns}
Dtypes: {dtypes}
Shape: {rows} rows, {cols} columns
Sample rows: {sample}

User question: {query}

Write a single pandas expression or a short pandas code block to answer this question accurately.
Rules:
- The dataframe is already loaded as `df`
- Do not import any libraries
- Do not use any file I/O operations
- ALWAYS check if a filtered dataframe is empty before accessing `.iloc[0]` or `.index[0]` to avoid IndexErrors. If it is empty, set result to None or an appropriate message.
- Assign the final result to a variable called `result`
- Return only the code, no explanation, no markdown, no backticks"""


def execute_excel_query(
    file_path: str,
    schema: dict,
    query: str,
    file_type: Optional[FileType] = None,
) -> Optional[str]:
    """
    Generate a pandas query via Claude and execute it in a RestrictedPython
    sandbox with a 10-second timeout.

    Returns str(result) on success, None on any failure.
    """
    try:
        print(
            f"Executing Excel query for {file_path} with schema: {json.dumps(schema)} and query: {query}"
        )
        # Load the dataframe
        if file_type is None:
            ext = "." + file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
            file_type = EXTENSION_TO_FILE_TYPE.get(ext)
            if not file_type:
                raise ValueError(f"Unsupported file extension: {ext}")
        df = load_dataframe(file_path, file_type)

        print(
            f"Loaded dataframe with shape: {df.shape} and columns: {df.columns.tolist()}"
        )
        # Build the prompt
        prompt = _EXCEL_CODE_PROMPT.format(
            columns=schema.get("columns", []),
            dtypes=schema.get("dtypes", {}),
            rows=schema.get("shape", {}).get("rows", 0),
            cols=schema.get("shape", {}).get("columns", 0),
            sample=schema.get("sample", []),
            query=query,
        )
        print(f"Generated prompt for Claude: {prompt}")

        # -- Anthropic LLM call (ACTIVE) ------------------------------------------
        client = _get_anthropic_client()
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=8192,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        code = (message.content[0].text or "").strip()

        print(f"Received code from Claude: {code}")

        # Strip markdown fences if present
        if code.startswith("```"):
            code = code.split("\n", 1)[-1]
            if code.endswith("```"):
                code = code[: -len("```")].strip()

        if not code:
            logger.warning("Anthropic returned empty code for Excel query")
            return None

        # Compile with RestrictedPython
        compiled = compile_restricted(code, filename="<excel_query>", mode="exec")
        print(f"Compiled code: {compiled}")
        # Build restricted globals — only allow df and safe builtins
        from RestrictedPython.Guards import guarded_iter_unpack_sequence
        from RestrictedPython.Eval import (
            default_guarded_getitem,
            default_guarded_getiter,
        )

        restricted_globals = dict(safe_globals)
        restricted_globals["_getattr_"] = getattr
        restricted_globals["_getitem_"] = default_guarded_getitem
        restricted_globals["_getiter_"] = default_guarded_getiter
        restricted_globals["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence

        # We use a pass-through write guard because full_write_guard wraps the object
        # and breaks Pandas dataframe item assignments (df['a'] = 1)
        restricted_globals["_write_"] = lambda x: x
        restricted_globals["pd"] = pd

        restricted_locals: dict = {"df": df}

        # Execution with a 10-second timeout (threading approach for Windows)
        result_container: dict = {"result": None, "error": None}

        def _execute():
            try:
                exec(compiled, restricted_globals, restricted_locals)  # noqa: S102
                result_container["result"] = restricted_locals.get("result")
            except Exception as exc:
                result_container["error"] = str(exc)

        thread = threading.Thread(target=_execute, daemon=True)
        thread.start()
        thread.join(timeout=10)
        if thread.is_alive():
            raise TimeoutError("Excel query execution timed out (10s)")

        if result_container["error"]:
            raise RuntimeError(
                f"Excel query execution error: {result_container['error']}"
            )

        result = result_container["result"]
        print(f"Excel query execution result: {result}")
        if result is None:
            raise ValueError("No result was assigned to the 'result' variable.")

        return str(result)

    except Exception as exc:
        logger.error("execute_excel_query failed: %s", exc)
        raise exc


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an intelligent assistant for an enterprise knowledge management system.
Answer the user's question based ONLY on the provided context below.
If the answer cannot be found in the context, say "I could not find relevant information in the available documents."
Do not make up information. Be concise and accurate.

At the very end of your response, after your answer, you MUST suggest 2-3 subsequent questions that the user may likely ask next. These questions must strictly be related to the provided context.
You MUST format this section exactly like this:
[FOLLOW_UP]
- Question 1?
- Question 2?
- Question 3?

CHART OUTPUT RULE:
If and only if your answer contains numerical data that can be meaningfully visualised (comparisons, trends, distributions, rankings, time series), append a chart specification at the very end of your response using this exact format:

CHART_SPEC:{"chart_type":"bar","title":"<descriptive title>","x_key":"<field name>","y_keys":["<field name>"],"data":[{...},{...}]}

Rules for the chart spec:
- chart_type must be one of: bar, line, area, pie
- Choose the most appropriate chart_type for the data
- data must be a JSON array of flat objects, all objects must have the same keys
- x_key is the categorical or time-based field
- y_keys is an array of one or more numeric fields
- For pie charts, y_keys must contain exactly one field
- All values in y_keys fields must be numbers, not strings
- The CHART_SPEC line must be on its own line at the very end of your response, after all text
- Do not emit CHART_SPEC if the answer is narrative, qualitative, or contains no numerical comparisons
- Do not wrap CHART_SPEC in markdown code fences
"""


async def _rerank_chunks(query: str, hits: list[dict], label: str = "") -> list[dict]:
    """
    Re-ranks a list of Qdrant hits against the query using the CrossEncoder.
    Falls back to the original (RRF-ranked) order if re-ranking fails.
    """
    if not hits:
        return hits
    if not settings.ENABLE_CROSS_ENCODER_RERANKING:
        return hits
    try:
        model = _get_cross_encoder()
        pairs = [
            (query, hit.get("payload", {}).get("chunk_text") or "") for hit in hits
        ]
        scores = await asyncio.to_thread(model.predict, pairs)
        for hit, score in zip(hits, scores):
            hit["cross_encoder_score"] = float(score)
        hits.sort(key=lambda x: x.get("cross_encoder_score", -999999.0), reverse=True)
    except Exception as exc:
        logger.error(
            "CrossEncoder re-ranking failed%s: %s",
            f" for {label}" if label else "",
            exc,
        )
        # Fall back to original RRF-ranked order (already the order in `hits`)
    return hits


def _format_chunk_context(filename: str, hit: dict) -> str:
    """Formats a single Qdrant hit into a context block line."""
    payload = hit.get("payload", {})
    page = payload.get("page_number")
    slide = payload.get("slide_number")
    chunk_idx = payload.get("chunk_index", 0)
    chunk_text = payload.get("chunk_text", "")

    location_parts = []
    if page is not None:
        location_parts.append(f"Page: {page}")
    if slide is not None:
        location_parts.append(f"Slide: {slide}")
    location = " | ".join(location_parts) if location_parts else ""
    location_str = f" | {location}" if location else ""

    return f"Source: {filename}{location_str} | Chunk: {chunk_idx}\n{chunk_text}"


async def _run_excel_query(
    doc: Document, query: str
) -> tuple[Optional[dict], Optional[str]]:
    """Runs an Excel code-gen query for a single document. Returns (result_dict, None) on success, (None, error_str) on failure."""
    if not doc.excel_schema:
        return None, "Document has no excel schema defined."
    try:
        abs_path = get_absolute_path(doc.file_path)
        result = await asyncio.to_thread(
            execute_excel_query, abs_path, doc.excel_schema, query, doc.file_type
        )
        if result is not None:
            return {
                "filename": doc.filename,
                "document_id": str(doc.id),
                "result": result,
            }, None
    except Exception as exc:
        logger.warning("Excel query failed for %s: %s", doc.filename, exc)
        return None, str(exc)
    return None, "Unknown execution failure."


async def _run_qdrant_search(
    query: str,
    query_vector,
    collection_name: str,
    role_ids: list[str],
    document_id: Optional[str] = None,
    limit: int = 15,
) -> list[dict]:
    """Runs a Qdrant search, returning [] on failure rather than raising."""
    try:
        return await asyncio.to_thread(
            search_vectors,
            collection_name=collection_name,
            query_text=query,
            query_vector=query_vector,
            role_ids=role_ids,
            limit=limit,
            document_id=document_id,
        )
    except Exception as exc:
        logger.error(
            "Qdrant search failed%s: %s",
            f" for document {document_id}" if document_id else "",
            exc,
        )
        return []


async def _resolve_compare_document(
    did_uuid: uuid.UUID,
    query: str,
    query_vector,
    tenant_id: str,
    role_ids: list[str],
    compare_id_to_doc: dict[str, Document],
    all_authorized_docs: list[Document],
) -> tuple[str, list[str]]:
    """
    Resolves the context lines for a single document in /compare mode.
    Returns (context_text, [the qdrant hits contributed, if any]) so the caller
    can both build the context block and collect citations.
    """
    did_str = str(did_uuid)
    doc_in_db = compare_id_to_doc.get(did_str)

    if not doc_in_db:
        return (
            f"Source: Unknown Document ({did_str})\n[Note: This document was not found in the database.]",
            [],
        )

    filename = doc_in_db.filename
    authorized_doc = next(
        (d for d in all_authorized_docs if str(d.id) == did_str), None
    )

    if not authorized_doc:
        return (
            f"Source: {filename}\n[Note: This document has no available content or is still processing.]",
            [],
        )

    if authorized_doc.file_type in TABULAR_FILE_TYPES:
        if not authorized_doc.excel_schema:
            return (
                f"Source: {filename}\n[Note: This document has no defined schema.]",
                [],
            )
        result, error = await _run_excel_query(authorized_doc, query)
        if result is not None:
            return f"Source: {filename}\nQuery: {query}\nResult: {result['result']}", []
        return (
            f"Source: {filename}\n[Note: No data results were retrieved from this document.]",
            [],
        )

    # Non-Excel: Qdrant search scoped to this one document, then re-rank and keep top 6.
    collection_name = authorized_doc.qdrant_collection or f"tenant_{tenant_id}"
    doc_results = await _run_qdrant_search(
        query=query,
        query_vector=query_vector,
        collection_name=collection_name,
        role_ids=role_ids,
        document_id=did_str,
        limit=15,
    )
    doc_results = [
        hit
        for hit in doc_results
        if hit.get("payload", {}).get("document_id") == did_str
    ]
    doc_results = await _rerank_chunks(query, doc_results, label=filename)
    doc_results = doc_results[:6]

    if not doc_results:
        return (
            f"Source: {filename}\n[Note: This document has no ready or indexed chunks.]",
            [],
        )

    lines = [_format_chunk_context(filename, hit) for hit in doc_results]
    return "\n---\n".join(lines), doc_results


def get_recent_turns(db: Session, session_id: uuid.UUID, limit: int = 5) -> list[dict]:
    from app.models.query_message import QueryMessage
    from app.models.enums import MessageRole

    messages = (
        db.query(QueryMessage)
        .filter(QueryMessage.session_id == session_id)
        .order_by(QueryMessage.created_at.desc())
        .limit(limit * 2)
        .all()
    )
    messages = list(reversed(messages))

    turns = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.role == MessageRole.user:
            if i + 1 < len(messages) and messages[i + 1].role == MessageRole.assistant:
                turns.append(
                    {
                        "question": msg.content,
                        "answer": messages[i + 1].content,
                        "generated_sql": getattr(
                            messages[i + 1], "generated_sql", None
                        ),
                        "query_results": getattr(
                            messages[i + 1], "query_results", None
                        ),
                    }
                )
                i += 2
            else:
                i += 1
        else:
            i += 1

    return turns[-limit:]


async def _execute_llm_stream(cfg, sys_prompt, user_prompt, max_tokens=8192):
    """
    Streams tokens and usage dict from the appropriate provider SDK based on configuration.
    """
    from app.core.utils import get_provider_by_id, get_llm_client

    cfg_provider_id = cfg.provider_id
    cfg_model_string = cfg.model_name or cfg.model_string
    cfg_provider = get_provider_by_id(cfg_provider_id)
    cfg_sdk_type = cfg_provider["sdk_type"]

    if cfg_provider_id == "anthropic":
        if cfg.api_key:
            client = get_llm_client(cfg)
        else:
            client = _get_async_anthropic_client()

        stream_kwargs = {
            "model": cfg_model_string,
            "max_tokens": max_tokens,
            "system": sys_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if "opus" not in cfg_model_string.lower():
            stream_kwargs["temperature"] = 0

        async with client.messages.stream(**stream_kwargs) as stream:
            async for event in stream:
                if event.type == "text":
                    yield "token", event.text
            final_msg = await stream.get_final_message()
            if getattr(final_msg, "usage", None):
                yield (
                    "usage",
                    {
                        "input_tokens": getattr(final_msg.usage, "input_tokens", 0),
                        "output_tokens": getattr(final_msg.usage, "output_tokens", 0),
                    },
                )

    elif cfg_provider_id == "openrouter":
        from app.services.openrouter_service import stream_openrouter_completion

        async for chunk_type, data in stream_openrouter_completion(
            model_string=cfg_model_string,
            system_prompt=sys_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ):
            if chunk_type == "text":
                yield "token", data
            elif chunk_type == "usage":
                yield (
                    "usage",
                    {
                        "input_tokens": data.get("prompt_tokens", 0),
                        "output_tokens": data.get("completion_tokens", 0),
                    },
                )

    elif cfg_sdk_type == "openai_compat":
        client = get_llm_client(cfg)
        formatted_messages = []
        if sys_prompt:
            formatted_messages.append({"role": "system", "content": sys_prompt})
        formatted_messages.append({"role": "user", "content": user_prompt})

        response = await client.chat.completions.create(
            model=cfg_model_string,
            messages=formatted_messages,
            temperature=0,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    yield "token", delta.content
            if getattr(chunk, "usage", None) and chunk.usage:
                yield (
                    "usage",
                    {
                        "input_tokens": getattr(chunk.usage, "prompt_tokens", 0),
                        "output_tokens": getattr(chunk.usage, "completion_tokens", 0),
                    },
                )

    elif cfg_sdk_type == "google":
        client = get_llm_client(cfg)
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=sys_prompt,
            temperature=0,
            max_output_tokens=max_tokens,
        )
        response_stream = await client.aio.models.generate_content_stream(
            model=cfg_model_string,
            contents=user_prompt,
            config=config,
        )
        async for chunk in response_stream:
            text = chunk.text
            if text:
                yield "token", text
            if chunk.usage_metadata:
                yield (
                    "usage",
                    {
                        "input_tokens": chunk.usage_metadata.prompt_token_count or 0,
                        "output_tokens": chunk.usage_metadata.candidates_token_count
                        or 0,
                    },
                )
    else:
        raise ValueError(f"Unsupported sdk_type: {cfg_sdk_type}")


async def run_rag_pipeline(
    query: str,
    user: User,
    db: Session,
    conversation_history: str = "",
    document_id: Optional[uuid.UUID] = None,
    database_id: Optional[uuid.UUID] = None,
    command_instruction: Optional[str] = None,
    compare_document_ids: Optional[list[uuid.UUID]] = None,
    is_compare_mode: bool = False,
    is_summarize_mode: bool = False,
    model_id: Optional[uuid.UUID] = None,
    session_id: Optional[uuid.UUID] = None,
) -> AsyncGenerator[dict, None]:
    from app.services.agents.graph import rag_graph
    from app.services.agents.types import AgentState

    # Build initial state
    initial_state: AgentState = {
        "query": query,
        "user_id": str(user.id),
        "tenant_id": str(user.tenant_id),
        "database_id": str(database_id) if database_id else None,
        "document_id": str(document_id) if document_id else None,
        "conversation_history": conversation_history,
        "model_id": str(model_id) if model_id else None,
        "session_id": str(session_id) if session_id else None,
        "command_instruction": command_instruction,
        "compare_document_ids": [str(d) for d in compare_document_ids]
        if compare_document_ids
        else None,
        "is_compare_mode": is_compare_mode,
        "is_summarize_mode": is_summarize_mode,
        # Initialize all other fields to defaults
        "invoke_sql": False,
        "invoke_rag": False,
        "mode": "doc_only",
        "orchestrator_reasoning": "",
        "progress_tokens": [],
        "sql_result": None,
        "rag_result": None,
        "db_user_id": None,
        "db_connection_id": None,
        "db_authorized_schema": None,
        "db_session_turns": None,
        "db_authorized_cols_by_table": None,
        "db_all_physical_cols_by_table": None,
        "db_valid_tables": None,
        "db_connection_engine": None,
        "db_connection_name": None,
        "db_is_admin": False,
        "context_error": None,
        "query_plan": None,
        "db_filtered_schema": None,
        "generated_sql": None,
        "previous_sql": None,
        "sql_generation_attempts": 0,
        "sql_generation_error": None,
        "final_answer": "",
        "citations": [],
        "follow_up_questions": [],
        "chart_spec": None,
        "query_results": None,
        "model_string": None,
        "resolved_model": None,
        "resolved_model_id": None,
        "was_fallback": False,
        "fallback_model_name": None,
        "execution_time_ms": 0,
        "answer_judge_feedback": None,
        "answer_judge_attempts": 0,
    }

    # Call gather_sql_context upfront if database_id is present
    if database_id:
        from app.services.agents.nodes.sql import gather_sql_context

        context_updates = await gather_sql_context(initial_state, db)
        initial_state.update(context_updates)

    # Thread config for LangGraph checkpointer
    config = {
        "configurable": {
            "thread_id": str(session_id) if session_id else str(uuid.uuid4())
        }
    }

    # Stream graph execution - yield progress tokens as they arrive
    # then yield the final done event when graph completes
    yielded_tokens = set()  # deduplicate progress tokens

    print(f"[Graph] Starting graph execution for query: {query[:100]}")

    async for chunk in rag_graph.astream(initial_state, config=config):
        # Each chunk is a dict of {node_name: partial_state_update}
        for node_name, node_output in chunk.items():
            print(f"[Graph] Node completed: {node_name}")

            # Stream progress tokens to user as they arrive from each node
            if isinstance(node_output, dict) and "progress_tokens" in node_output:
                for token in node_output["progress_tokens"]:
                    token_key = f"{node_name}:{token}"
                    if token_key not in yielded_tokens:
                        yielded_tokens.add(token_key)
                        yield {"type": "token", "content": token}

    # After graph completes, get the final state
    # LangGraph astream yields node outputs - get the last full state
    full_final_state = await rag_graph.aget_state(config)
    state_values = full_final_state.values

    print("[Graph] Graph execution complete.")
    print(f"[Graph] Final mode: {state_values.get('mode')}")
    print(f"[Graph] Final answer length: {len(state_values.get('final_answer', ''))}")

    # Yield the done event using final state values
    yield {
        "type": "done",
        "answer": state_values.get(
            "final_answer", "I could not generate an answer. Please try again."
        ),
        "citations": state_values.get("citations", []),
        "model_string": state_values.get("model_string"),
        "follow_up_questions": state_values.get("follow_up_questions", []),
        "generated_sql": state_values.get("generated_sql"),
        "query_results": state_values.get("query_results"),
        "db_connection_id": state_values.get("db_connection_id"),
        "execution_time_ms": state_values.get("execution_time_ms", 0),
        "status": "success",
        "error_message": None,
        "error_type": None,
        "resolved_model": state_values.get("resolved_model"),
        "resolved_model_id": state_values.get("resolved_model_id"),
        "was_fallback": state_values.get("was_fallback", False),
        "fallback_model_name": state_values.get("fallback_model_name"),
        "chart_spec": state_values.get("chart_spec"),
    }
