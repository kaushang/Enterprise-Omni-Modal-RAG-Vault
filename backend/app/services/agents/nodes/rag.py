# Vector Search Agent node and tool definitions

import uuid
import asyncio
import re
import json
from typing import Optional
from sqlalchemy import or_
from app.db.session import SessionLocal
from app.services.agents.types import AgentState, RAGAgentResult
from app.models.user import User
from app.models.document import Document
from app.models.document_access_policy import DocumentAccessPolicy
from app.models.enums import DocumentStatus
from app.services import embedding_service
import app.services.rag_service as rag_service

logger = rag_service.logger


# --- Tool definitions ---

VECTOR_SEARCH_TOOLS = [
    {
        "name": "search_documents",
        "description": (
            "Runs semantic search and BM25 keyword search in parallel against the authorized document collection in Qdrant, "
            "then merges the results. Call this first with the user query. The query parameter MUST ALWAYS be a complete, "
            "well-formed, grammatically correct sentence preserving the user's intent (NEVER a list or string of keywords). "
            "On retry after judge feedback, call with a reformulated complete sentence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. MUST be a complete, well-formed, grammatically correct sentence that preserves the original "
                        "question's intent and structure. Do NOT extract keywords, strip question words, or convert to a keyword string. "
                        "On the first attempt, use the original user query or a complete rephrased sentence. "
                        "On retry, rephrase into a different complete sentence targeting what the previous retrieval missed."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of hits to retrieve. Default 15, increase to 20 on retry attempts.",
                },
            },
            "required": ["query", "limit"],
        },
    },
    {
        "name": "rerank_results",
        "description": "Reranks the raw hits from search_documents using a cross-encoder. Call this after search_documents. Returns ordered chunks with the most relevant first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The same query used in search_documents.",
                },
                "hits": {
                    "type": "array",
                    "description": "Raw hits returned by search_documents.",
                    "items": {"type": "object"},
                },
            },
            "required": ["query", "hits"],
        },
    },
    {
        "name": "evaluate_context_quality",
        "description": "Evaluates whether the retrieved context is sufficient to answer the user query. Call this after rerank_results. Returns sufficient (bool), confidence (float), reasoning (string), and fix_instruction (string). If not sufficient, use fix_instruction to reformulate the query and call search_documents again.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The original user query."},
                "context_block": {
                    "type": "string",
                    "description": "The assembled context block from reranked chunks.",
                },
            },
            "required": ["query", "context_block"],
        },
    },
]


# --- Tool executor functions ---


# Searches the vector database for the most relevant document chunks matching the user's query.
async def search_documents_tool(
    query: str,
    limit: int,
    collection_name: str,
    role_ids: list[str],
    document_id: Optional[str] = None,
) -> list[dict]:
    print(
        f"[Vector Search Agent] search_documents called with query='{query}', limit={limit}"
    )

    # Generate an embedding vector for the user's query.
    query_vector = embedding_service.embed_text(query)

    # Perform a semantic search in Qdrant using the query embedding and access filters.
    hits = await rag_service._run_qdrant_search(
        query=query,
        query_vector=query_vector,
        collection_name=collection_name,
        role_ids=role_ids,
        document_id=document_id,
        limit=limit,
    )

    print(f"[Vector Search Agent] search_documents returned {len(hits)} hits.")

    return hits


# Re-ranks the retrieved document chunks to improve their relevance to the user's query.
async def rerank_results_tool(query: str, hits: list[dict]) -> list[dict]:
    print(f"[Vector Search Agent] rerank_results called with {len(hits)} hits")

    # Re-rank the retrieved chunks using a relevance scoring model.
    reranked = await rag_service._rerank_chunks(query, hits)
    print(
        f"[Vector Search Agent] rerank_results returned {len(reranked)} reranked hits."
    )
    return reranked


async def evaluate_context_quality_tool(
    query: str,
    context_block: str,
    qdrant_results: list[dict],
) -> dict:
    print("[Vector Search Agent] evaluate_context_quality called")
    if context_block == "No relevant context found." or not context_block.strip():
        judgment = {
            "sufficient": False,
            "confidence": 0.0,
            "reasoning": "No context retrieved",
            "fix_instruction": "Try broader search terms",
        }
    else:
        try:
            client = rag_service._get_async_anthropic_client()
            judge_system = (
                "You are a retrieval quality evaluator. Given a user query and the retrieved context chunks, "
                "evaluate whether the retrieved context contains any information that could help answer the query, "
                "Only mark insufficient if the context is completely unrelated to the query, can not answer the user's query in any way or is entirely empty. "
                "Respond ONLY with a JSON object in this exact format with no other text:\n"
                '{"sufficient": true/false, "confidence": 0.0-1.0, "reasoning": "one sentence explanation", '
                '"fix_instruction": "if not sufficient, one sentence on how to reformulate the query to get better results, else empty string"}'
            )
            judge_prompt = f"""User Query: {query}
            Retrieved Context:
            {context_block}
            Number of chunks retrieved: {len(qdrant_results)}
            Number of Excel results: 0"""

            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                system=judge_system,
                messages=[{"role": "user", "content": judge_prompt}],
            )

            text = response.content[0].text.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)
            judgment = json.loads(text)
            if (
                "sufficient" not in judgment
                or "confidence" not in judgment
                or "reasoning" not in judgment
            ):
                raise ValueError("Invalid JSON keys in judge response")
        except Exception as exc:
            logger.warning("RAG Judge LLM call failed or parsed incorrectly: %s", exc)
            judgment = {
                "sufficient": True,
                "confidence": 0.7,
                "reasoning": "Judge unavailable",
                "fix_instruction": "",
            }

    print(
        f"[Vector Search Agent] judge result: sufficient={judgment['sufficient']}, confidence={judgment['confidence']}"
    )
    print(f"[Vector Search Agent] judge reasoning: {judgment['reasoning']}.")
    return judgment


# --- Excel Agent (kept as-is) ---


async def excel_agent(doc, query: str) -> Optional[dict]:
    print(f"[Excel Agent] Attempt 1 for {doc.filename}")
    result, error = await rag_service._run_excel_query(doc, query)
    if result is not None:
        return result

    print(
        f"[Excel Agent] Attempt 1 failed for {doc.filename}. Retrying with error feedback."
    )

    judgment = {"pandas_code": "", "reasoning": ""}
    try:
        client = rag_service._get_async_anthropic_client()
        system_prompt = (
            "You are a pandas code generation specialist. A previous attempt to generate pandas code to answer a user query against an Excel file failed during execution. \n"
            "Your job is to generate corrected pandas code.\n"
            "The dataframe is already loaded as the variable `df`.\n"
            "Respond ONLY with a JSON object in this exact format with no other text:\n"
            '{"pandas_code": "the corrected pandas code as a single string", "reasoning": "one sentence on what you changed"}'
        )

        user_prompt = f"""Excel File: {doc.filename}
        Schema: {json.dumps(doc.excel_schema)}
        User Query: {query}
        Previous attempt failed during execution.
        Execution Error: {error}
        Generate corrected pandas code to answer the query.
        The dataframe is loaded as `df`. Return only the final result as a variable named `result`."""

        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = response.content[0].text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        parsed = json.loads(text)
        if "pandas_code" in parsed:
            judgment = parsed
    except Exception as exc:
        logger.warning("Excel Agent LLM call failed or parsed incorrectly: %s", exc)
        print(
            f"[Excel Agent] Attempt 2 also failed for {doc.filename}. Returning None."
        )
        return None

    code = judgment.get("pandas_code", "").strip()
    if not code:
        print(
            f"[Excel Agent] Attempt 2 also failed for {doc.filename}. Returning None."
        )
        return None

    if code.startswith("```"):
        code = code.split("\n", 1)[-1]
        if code.endswith("```"):
            code = code[: -len("```")].strip()

    if not code:
        print(
            f"[Excel Agent] Attempt 2 also failed for {doc.filename}. Returning None."
        )
        return None

    try:
        import pandas as pd
        import threading
        from RestrictedPython import compile_restricted, safe_globals
        from RestrictedPython.Guards import guarded_iter_unpack_sequence
        from RestrictedPython.Eval import (
            default_guarded_getitem,
            default_guarded_getiter,
        )

        abs_path = rag_service.get_absolute_path(doc.file_path)
        df = rag_service.load_dataframe(abs_path, doc.file_type)

        compiled = compile_restricted(code, filename="<excel_query>", mode="exec")

        restricted_globals = dict(safe_globals)
        restricted_globals["_getattr_"] = getattr
        restricted_globals["_getitem_"] = default_guarded_getitem
        restricted_globals["_getiter_"] = default_guarded_getiter
        restricted_globals["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
        restricted_globals["_write_"] = lambda x: x
        restricted_globals["pd"] = pd

        restricted_locals = {"df": df}

        result_container = {"result": None, "error": None}

        def _execute():
            try:
                exec(compiled, restricted_globals, restricted_locals)
                result_container["result"] = restricted_locals.get("result")
            except Exception as exc:
                result_container["error"] = str(exc)

        thread = threading.Thread(target=_execute, daemon=True)
        thread.start()
        thread.join(timeout=10)

        if thread.is_alive():
            logger.warning("Excel query execution timed out (10s) on attempt 2")
            print(
                f"[Excel Agent] Attempt 2 also failed for {doc.filename}. Returning None."
            )
            return None

        if result_container["error"]:
            logger.debug(
                "Excel query execution error on attempt 2: %s",
                result_container["error"],
            )
            print(
                f"[Excel Agent] Attempt 2 also failed for {doc.filename}. Returning None."
            )
            return None

        result = result_container["result"]
        if result is None:
            print(
                f"[Excel Agent] Attempt 2 also failed for {doc.filename}. Returning None."
            )
            return None

        print(f"[Excel Agent] Attempt 2 succeeded for {doc.filename}")
        return {
            "filename": doc.filename,
            "document_id": str(doc.id),
            "result": str(result),
        }

    except Exception as exc:
        logger.error("excel_agent attempt 2 execution failed: %s", exc)
        print(
            f"[Excel Agent] Attempt 2 also failed for {doc.filename}. Returning None."
        )
        return None


def get_db_session():
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise


# --- vector_search_node function ---


async def vector_search_node(state: AgentState) -> dict:
    # Extract query parameters and mode flags from state
    query = state["query"]
    document_id = state.get("document_id")
    compare_document_ids = state.get("compare_document_ids")
    is_compare_mode = state.get("is_compare_mode", False)
    is_summarize_mode = state.get("is_summarize_mode", False)

    # Read pre-computed shared values from state - no DB access needed here
    search_role_ids = state["search_role_ids"]
    collection_name = state["collection_name"]
    doc_id_to_filename = state["doc_id_to_filename"]
    authorized_doc_ids = state["authorized_doc_ids"]

    # Guard - exit immediately if no authorized documents exist
    if not authorized_doc_ids:
        print("[Vector Search Agent] No authorized documents found.")
        return {
            "rag_result": RAGAgentResult(
                success=False,
                reasoning="No authorized documents found.",
                confidence=0.0,
            )
        }

    # Normalize single document ID to UUID if provided
    doc_id_uuid = (
        uuid.UUID(document_id)
        if isinstance(document_id, str) and document_id
        else (document_id if isinstance(document_id, uuid.UUID) else None)
    )

    # Normalize compare document IDs to UUIDs if provided
    compare_doc_ids_uuids = (
        [
            uuid.UUID(did) if isinstance(did, str) else did
            for did in compare_document_ids
        ]
        if compare_document_ids
        else None
    )

    print(
        f"[Vector Search Agent] Starting. Query: '{query}', authorized docs: {len(authorized_doc_ids)}"
    )

    # Compare mode - run one direct search per document in parallel, skip the ReAct loop
    if is_compare_mode and compare_doc_ids_uuids:
        print(
            f"[Vector Search Agent] Compare mode detected, resolving {len(compare_doc_ids_uuids)} documents in parallel"
        )

        # Embed the query once and reuse across all per-document searches
        query_vector = embedding_service.embed_text(query)

        # Run one Qdrant search per document concurrently, each scoped to its own document ID
        per_doc_searches = await asyncio.gather(
            *[
                rag_service._run_qdrant_search(
                    query=query,
                    query_vector=query_vector,
                    collection_name=collection_name,
                    role_ids=search_role_ids,
                    document_id=str(did_uuid),
                    limit=8,
                )
                for did_uuid in compare_doc_ids_uuids
            ],
            return_exceptions=True,
        )

        # Assemble context block with each document's chunks clearly labeled and separated
        context_parts, qdrant_results = [], []
        for did_uuid, hits in zip(compare_doc_ids_uuids, per_doc_searches):
            filename = doc_id_to_filename.get(str(did_uuid), f"Document ({did_uuid})")
            if isinstance(hits, BaseException):
                logger.error(
                    "Compare search failed for document %s: %s", did_uuid, hits
                )
                context_parts += [
                    f"Source: {filename}\n[Note: Failed to retrieve content for this document.]",
                    "---",
                ]
                continue
            if not hits:
                context_parts += [
                    f"Source: {filename}\n[Note: No relevant content found in this document.]",
                    "---",
                ]
                continue

            # Rerank each document's hits independently before adding to context
            reranked_hits = await rag_service._rerank_chunks(query, hits)
            reranked_hits = reranked_hits[:5]
            qdrant_results.extend(reranked_hits)

            # Format each document's chunks under its own labeled section
            context_parts.append(f"[{filename}]")
            for hit in reranked_hits:
                context_parts.append(rag_service._format_chunk_context(filename, hit))
                context_parts.append("---")

        context_block = (
            "\n".join(context_parts) if context_parts else "No relevant context found."
        )
        print(
            f"[Vector Search Agent] Compare mode complete. Total hits: {len(qdrant_results)}"
        )
        return {
            "rag_result": RAGAgentResult(
                success=bool(qdrant_results),
                qdrant_results=qdrant_results,
                excel_results=[],
                context_block=context_block,
                doc_id_to_filename=doc_id_to_filename,
                confidence=1.0,
                reasoning="Compare mode - direct per-document retrieval, no judge evaluation needed.",
                attempts=1,
            )
        }

    # ReAct loop setup - determine chunk limit based on summarize mode
    max_attempts = 3
    attempts = 0
    current_query = query
    final_limit = 8 if (is_summarize_mode and doc_id_uuid) else 5
    client = rag_service._get_async_anthropic_client()

    # System prompt instructs the agent on tool usage and query formulation rules
    system_prompt = f"""You are a vector search agent. Your job is to retrieve the most relevant document chunks to answer the user's query.

    CRITICAL INSTRUCTIONS FOR QUERY FORMULATION:
    - When calling `search_documents`, you MUST pass a complete, well-formed, grammatically correct sentence.
    - NEVER strip queries down to a string of keywords or bag of words (for example, NEVER turn "What is the significance of X and why is Y a problem?" into "significance x problem y").
    - If a user query has multiple parts or sub-questions, EACH of them should be addressed and represented in the query sentence you pass to `search_documents`.
    - Both on the first attempt and on retries, the query parameter MUST preserve the original question's intent and sentence structure.
    - Rephrasing is permitted only to make the sentence clearer or more retrieval-friendly, but it MUST ALWAYS remain a complete, grammatical sentence.

    You have three tools:
    - search_documents: searches the document collection using semantic and keyword search, returns raw hits
    - rerank_results: reranks a list of raw hits by relevance to the query, returns ordered chunks  
    - evaluate_context_quality: evaluates whether the retrieved context is sufficient to answer the query, returns sufficient/confidence/reasoning/fix_instruction

    If evaluate_context_quality returns sufficient=false, reformulate the query into a new complete, well-formed sentence based on the fix_instruction and call search_documents again with the new query sentence and a limit of 20. You have a maximum of {max_attempts} total search attempts.

    When you are done (context is sufficient, or you have exhausted attempts), stop calling tools."""

    # Initialize ReAct loop tracking variables
    messages = []
    current_hits = []
    current_context_block = "No relevant context found."
    last_judgment = {"sufficient": False, "confidence": 0.7, "reasoning": ""}
    fix_instruction = ""
    sufficient = False

    # ReAct loop - each iteration is one retrieval attempt with its own tool call sequence
    while attempts < max_attempts:
        print(f"[Vector Search Agent] ReAct loop attempt {attempts + 1}/{max_attempts}")

        # First attempt uses original query; retries include judge feedback for reformulation
        user_msg = (
            (
                f"Query: {query}\n"
                f"Available documents: {len(authorized_doc_ids)} documents authorized for this user.\n"
                f"Final chunk limit after reranking: {final_limit}\n"
                f"Retrieve the best possible context to answer this query. Remember: pass a complete, well-formed sentence to search_documents. Do NOT convert it into a keyword list."
            )
            if attempts == 0
            else (
                f"Previous attempt was insufficient. Judge feedback: {fix_instruction}\n"
                f"Reformulate the query into a new complete, well-formed sentence preserving original question intent and try again. Attempts remaining: {max_attempts - attempts}. Remember: NEVER use a keyword list."
            )
        )
        messages.append({"role": "user", "content": user_msg})

        # Inner tool call loop - continues until agent stops calling tools or sufficient context is found
        while True:
            try:
                response = await client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    system=system_prompt,
                    tools=VECTOR_SEARCH_TOOLS,
                    messages=messages,
                )
            except Exception as exc:
                logger.error("[Vector Search Agent] Claude API error: %s", exc)
                break

            messages.append({"role": "assistant", "content": response.content})

            # Extract tool use blocks - if none present or agent is done, exit inner loop
            tool_use_blocks = [
                b for b in response.content if getattr(b, "type", None) == "tool_use"
            ]
            if not tool_use_blocks or response.stop_reason == "end_turn":
                break

            # Execute each tool the agent called and collect results to feed back
            tool_results = []
            for block in tool_use_blocks:
                tool_name = block.name
                tool_input = block.input

                if tool_name == "search_documents":
                    # Run semantic + BM25 search with the agent's formulated query
                    search_q = tool_input.get("query", current_query)
                    current_query = search_q
                    hits = await search_documents_tool(
                        query=search_q,
                        limit=tool_input.get("limit", 15),
                        collection_name=collection_name,
                        role_ids=search_role_ids,
                        document_id=str(doc_id_uuid) if doc_id_uuid else None,
                    )
                    current_hits = hits
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(
                                {"hits_count": len(hits), "hits": hits}
                            ),
                        }
                    )

                elif tool_name == "rerank_results":
                    # Rerank retrieved hits and build a formatted context block from top results
                    reranked = await rerank_results_tool(
                        query=tool_input.get("query", current_query),
                        hits=tool_input.get("hits") or current_hits,
                    )
                    reranked = reranked[:final_limit]
                    current_hits = reranked

                    # Format each reranked chunk with its source filename
                    context_parts = []
                    if current_hits:
                        context_parts.append("[Document Chunks]")
                        for hit in current_hits:
                            filename = doc_id_to_filename.get(
                                hit.get("payload", {}).get("document_id", ""), "Unknown"
                            )
                            context_parts.append(
                                rag_service._format_chunk_context(filename, hit)
                            )
                            context_parts.append("---")
                    current_context_block = (
                        "\n".join(context_parts)
                        if context_parts
                        else "No relevant context found."
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(
                                {
                                    "reranked_count": len(reranked),
                                    "context_block": current_context_block,
                                }
                            ),
                        }
                    )

                elif tool_name == "evaluate_context_quality":
                    # Judge evaluates whether retrieved context sufficiently answers the query
                    judgment = await evaluate_context_quality_tool(
                        query=tool_input.get("query", query),
                        context_block=tool_input.get(
                            "context_block", current_context_block
                        ),
                        qdrant_results=current_hits,
                    )
                    last_judgment = judgment
                    sufficient = judgment.get("sufficient", False)
                    if not sufficient:
                        fix_instruction = judgment.get("fix_instruction", "")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(judgment),
                        }
                    )

            messages.append({"role": "user", "content": tool_results})

            # Exit inner loop immediately if judge confirmed sufficient context
            if sufficient:
                break

        attempts += 1
        if sufficient:
            break

    # Filter out any hits whose document ID is not in the authorized set
    qdrant_results = [
        hit
        for hit in current_hits
        if hit.get("payload", {}).get("document_id") in doc_id_to_filename
    ]

    print(
        f"[Vector Search Agent] Done. Attempts: {attempts}, hits: {len(qdrant_results)}, sufficient: {sufficient}"
    )

    return {
        "rag_result": RAGAgentResult(
            success=bool(qdrant_results),
            qdrant_results=qdrant_results,
            excel_results=[],
            context_block=current_context_block,
            doc_id_to_filename=doc_id_to_filename,
            confidence=last_judgment.get("confidence", 0.7),
            reasoning=last_judgment.get("reasoning", ""),
            attempts=attempts,
            reformulated_query=current_query if current_query != query else None,
        )
    }


# --- excel_agent_node (Placeholder) ---


async def excel_agent_node(state: AgentState) -> dict:
    print("[Excel Agent Node] Placeholder - returning empty result")
    return {
        "rag_result": RAGAgentResult(
            success=False,
            excel_results=[],
            reasoning="Excel agent not yet implemented",
        )
    }


# --- rag_pipeline_node function ---


async def rag_pipeline_node(state: AgentState) -> dict:
    # Extract query and document targeting from state
    query = state["query"]
    user_id = state["user_id"]
    document_id = state.get("document_id")
    compare_document_ids = state.get("compare_document_ids")

    db = get_db_session()
    try:
        # Fetch the requesting user
        user = db.query(User).filter(User.id == uuid.UUID(user_id)).first()
        if not user:
            return {
                "rag_result": RAGAgentResult(
                    success=False, reasoning="User not found.", confidence=0.0
                )
            }

        # Normalize document ID(s) to UUID objects
        doc_id_uuid = (
            uuid.UUID(document_id)
            if isinstance(document_id, str) and document_id
            else (document_id if isinstance(document_id, uuid.UUID) else None)
        )

        compare_doc_ids_uuids = (
            [
                uuid.UUID(did) if isinstance(did, str) else did
                for did in compare_document_ids
            ]
            if compare_document_ids
            else None
        )

        # Build the base authorization query - only ready, non-archived docs the user can access
        docs_query = (
            db.query(Document)
            .outerjoin(
                DocumentAccessPolicy, Document.id == DocumentAccessPolicy.document_id
            )
            .filter(
                Document.tenant_id == user.tenant_id,
                Document.status == DocumentStatus.ready,
                Document.is_archived.is_(False),
                or_(
                    DocumentAccessPolicy.role_id == user.role_id,
                    Document.uploaded_by == user.id,
                ),
            )
        )

        # Narrow scope to specific document(s) if provided
        if compare_doc_ids_uuids:
            docs_query = docs_query.filter(Document.id.in_(compare_doc_ids_uuids))
        elif doc_id_uuid:
            docs_query = docs_query.filter(Document.id == doc_id_uuid)

        all_authorized_docs = docs_query.distinct().all()

        # Derive shared values for child agents - computed once, passed via state
        tenant_id = str(user.tenant_id)
        search_role_ids = [str(user.role_id), str(user.id)]
        doc_id_to_filename = {str(doc.id): doc.filename for doc in all_authorized_docs}
        authorized_doc_ids = [str(doc.id) for doc in all_authorized_docs]
        collection_name = (
            all_authorized_docs[0].qdrant_collection
            if doc_id_uuid and all_authorized_docs
            else f"tenant_{tenant_id}"
        )
        print(
            f"[RAG Pipeline] Computed shared state: tenant_id={tenant_id}, collection={collection_name}, authorized_docs={len(all_authorized_docs)}"
        )

        # Separate Excel/tabular files from text documents in a single pass
        excel_file_types = {*rag_service.TABULAR_FILE_TYPES, "xlsx", "xls", "csv"}
        excel_docs, non_excel_docs = [], []
        for d in all_authorized_docs:
            (
                excel_docs
                if (d.file_type or "").lower() in excel_file_types
                else non_excel_docs
            ).append(d)

        # Summarize available document types for the classifier
        doc_types_str = (
            ", ".join(
                {
                    (d.file_type or "").lower()
                    for d in all_authorized_docs
                    if d.file_type
                }
            )
            or "None"
        )
        has_text, has_excel = bool(non_excel_docs), bool(excel_docs)

        # Default routing - run both agents unless classifier overrides
        run_vector_search, run_excel_agent, reasoning = True, True, ""

        try:
            # Ask Haiku to classify which agents are needed based on query + available doc types
            client = rag_service._get_async_anthropic_client()
            classifier_system = (
                "You are a query routing classifier. Given a user query and the types of documents available, "
                "decide which retrieval agents to run.\n\n"
                "Respond ONLY with a JSON object in this exact format with no other text:\n"
                '{"run_vector_search": true/false, "run_excel_agent": true/false, "reasoning": "one sentence explanation"}\n\n'
                "Rules:\n"
                "- run_vector_search should be true if the query could be answered from text documents (PDF, DOCX, PPTX, TXT, audio transcripts)\n"
                "- run_excel_agent should be true if the query could be answered from tabular data (Excel, CSV files)\n"
                "- If uncertain, set both to true\n"
                "- If no Excel/CSV files are available, always set run_excel_agent to false\n"
                "- If no text documents are available, always set run_vector_search to false\n"
                "- Never set both to false"
            )
            classifier_user = (
                f"User Query: {query}\n\n"
                f"Available document types: {doc_types_str}\n"
                f"Text documents (PDF, DOCX, PPTX, TXT) available: {'yes' if has_text else 'no'}\n"
                f"Excel/CSV files available: {'yes' if has_excel else 'no'}\n"
            )
            print(
                f"[Classifying query routing] Query: '{query}', doc types: {doc_types_str}, has_text={has_text}, has_excel={has_excel}"
            )
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                system=classifier_system,
                messages=[{"role": "user", "content": classifier_user}],
            )

            # Parse the classifier JSON response
            text = response.content[0].text.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            result_json = json.loads(match.group(0) if match else text)

            run_vector_search = bool(result_json.get("run_vector_search", True))
            run_excel_agent = bool(result_json.get("run_excel_agent", True))
            reasoning = str(result_json.get("reasoning", ""))

            # Hard overrides based on actual document availability
            if not has_excel:
                run_excel_agent = False
            if not has_text:
                run_vector_search = False

            # Safety fallback - never let both be false
            if not run_vector_search and not run_excel_agent:
                run_vector_search = has_text or not has_excel
                run_excel_agent = has_excel or not has_text

            print(
                f"[RAG Pipeline] Classification result: run_vector_search={run_vector_search}, run_excel_agent={run_excel_agent}, reasoning={reasoning}"
            )

        except Exception as exc:
            # On classifier failure, default to running both agents
            logger.warning("[RAG Pipeline] Classification failed: %s", exc)
            run_vector_search, run_excel_agent = True, True
            print("[RAG Pipeline] Classification failed, defaulting to both agents.")

    finally:
        db.close()

    # Build pipeline state with shared values for child agents
    pipeline_state = {
        **state,
        "tenant_id": tenant_id,
        "search_role_ids": search_role_ids,
        "doc_id_to_filename": doc_id_to_filename,
        "authorized_doc_ids": authorized_doc_ids,
        "collection_name": collection_name,
    }

    # Build list of agent coroutines to run based on classification decision
    coroutines, agent_names = [], []
    if run_vector_search:
        coroutines.append(vector_search_node(pipeline_state))
        agent_names.append("vector_search_node")
    if run_excel_agent:
        coroutines.append(excel_agent_node(pipeline_state))
        agent_names.append("excel_agent_node")

    print(f"[RAG Pipeline] Running agents: {agent_names}")

    # Guard - should not happen due to safety fallback above, but handle gracefully
    if not coroutines:
        print("[RAG Pipeline] Merge complete. Qdrant hits: 0, Excel results: 0")
        return {
            "rag_result": RAGAgentResult(success=False, reasoning="No agents ran"),
            "tenant_id": tenant_id,
            "search_role_ids": search_role_ids,
            "doc_id_to_filename": doc_id_to_filename,
            "authorized_doc_ids": authorized_doc_ids,
            "collection_name": collection_name,
        }

    # Run selected agents in parallel and collect results
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    vec_res: Optional[RAGAgentResult] = None
    excel_res: Optional[RAGAgentResult] = None

    # Unpack results by agent name, logging any agent-level failures
    for agent_name, result in zip(agent_names, results):
        if isinstance(result, BaseException):
            logger.error("[RAG Pipeline] Agent %s failed: %s", agent_name, result)
            continue
        res_rag = result.get("rag_result") if isinstance(result, dict) else None
        if agent_name == "vector_search_node":
            vec_res = res_rag
        elif agent_name == "excel_agent_node":
            excel_res = res_rag

    # Merge agent results - vector search is the base, excel results are injected into it
    if vec_res and vec_res.success:
        merged_dict = {**vec_res.__dict__}
        if excel_res and excel_res.excel_results:
            merged_dict["excel_results"] = excel_res.excel_results
        merged_rag_result = RAGAgentResult(**merged_dict)

    elif excel_res and excel_res.success:
        merged_dict = {**excel_res.__dict__, "qdrant_results": []}
        if (
            merged_dict.get("context_block") == "No relevant context found."
            and excel_res.excel_results
        ):
            merged_dict["context_block"] = "\n---\n".join(
                f"Source: {er.get('filename', 'Unknown')}\nResult: {er.get('result', '')}"
                for er in excel_res.excel_results
            )
        merged_rag_result = RAGAgentResult(**merged_dict)

    elif vec_res:
        # Neither succeeded but vector search ran - return its result as best effort
        merged_rag_result = vec_res

    else:
        merged_rag_result = RAGAgentResult(
            success=False, reasoning="No agents produced results"
        )
    print(
        f"[RAG Pipeline] Merge complete. Qdrant hits: {len(merged_rag_result.qdrant_results)}, Excel results: {len(merged_rag_result.excel_results)}"
    )

    return {
        "rag_result": merged_rag_result,
        "tenant_id": tenant_id,
        "search_role_ids": search_role_ids,
        "doc_id_to_filename": doc_id_to_filename,
        "authorized_doc_ids": authorized_doc_ids,
        "collection_name": collection_name,
    }
