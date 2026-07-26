# Vector Search Agent node and tool definitions

import asyncio
import json
import re
import uuid

from sqlalchemy import or_

import app.services.rag_service as rag_service
from app.db.session import SessionLocal
from app.models.document import Document
from app.models.document_access_policy import DocumentAccessPolicy
from app.models.enums import DocumentStatus
from app.models.user import User
from app.services import embedding_service
from app.services.agents.tools.rag_tools import (
    EXCEL_AGENT_TOOLS,
    VECTOR_SEARCH_TOOLS,
    evaluate_context_quality_tool,
    execute_pandas_code_tool,
    generate_pandas_code_tool,
    get_excel_schemas_tool,
    get_sample_values_tool,
    identify_relevant_files_tool,
    rerank_results_tool,
    search_documents_tool,
)
from app.services.agents.types import AgentState, RAGAgentResult

logger = rag_service.logger


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


# --- excel_agent_node ---


async def excel_agent_node(state: AgentState) -> dict:
    query = state["query"]
    authorized_doc_ids = state["authorized_doc_ids"]
    doc_id_to_filename = state["doc_id_to_filename"]

    db = get_db_session()
    try:
        doc_uuids = [
            uuid.UUID(did) if isinstance(did, str) else did
            for did in authorized_doc_ids
        ]
        excel_file_types = {
            *rag_service.TABULAR_FILE_TYPES,
            "xlsx",
            "xls",
            "csv",
            "xlsb",
            "xlsm",
            "tsv",
            "ods",
        }
        docs = db.query(Document).filter(Document.id.in_(doc_uuids)).all()
        excel_docs = [
            d
            for d in docs
            if (d.file_type or "").lower() in excel_file_types
            and d.excel_schema is not None
        ]
    finally:
        db.close()

    if not excel_docs:
        print("[Excel Agent] No Excel/CSV files with schemas found.")
        return {
            "rag_result": RAGAgentResult(
                success=False,
                excel_results=[],
                reasoning="No Excel files available.",
            )
        }

    print(
        f"[Excel Agent] Starting. Found {len(excel_docs)} Excel/CSV files with schemas."
    )

    system_prompt = f"""You are an Excel data agent. Your goal is to retrieve accurate data from Excel and CSV files that answers the user's query.

    You have tools available:
    - get_excel_schemas: retrieves the schema of all authorized Excel and CSV files including filenames, column names, and data types
    - get_sample_values: takes a document ID and returns 5-10 sample rows from that file. Use this when column names alone are ambiguous and you need more context to decide if a file is relevant to the query.
    - identify_relevant_files: takes the retrieved schemas and the user query, makes an LLM call to determine which files are capable of answering the query, returns a list of relevant document IDs. If no files are relevant, returns an empty list.
    - generate_pandas_code: takes a document ID and the user query, generates pandas code for that file.
    - execute_pandas_code: takes a document ID and the generated pandas code, executes it against the file, and returns the result. Call generate_pandas_code first before calling this.

    You have access to {len(excel_docs)} Excel/CSV files. If no files are relevant to the query, stop and return nothing - do not force execution on irrelevant files.

    Think carefully about which tools to use to best achieve your goal."""

    messages = [
        {
            "role": "user",
            "content": (
                f"Query: {query}\n"
                "Find and return all data from Excel/CSV files that can answer this query."
            ),
        }
    ]
    cached_schemas = []
    cached_code: dict[str, str] = {}
    client = rag_service._get_async_anthropic_client()
    excel_results = []

    while True:
        try:
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
                system=system_prompt,
                tools=EXCEL_AGENT_TOOLS,
                messages=messages,
            )
        except Exception as exc:
            logger.error("[Excel Agent] Claude API error: %s", exc)
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_use_blocks = [
            b for b in response.content if getattr(b, "type", None) == "tool_use"
        ]
        if not tool_use_blocks or response.stop_reason == "end_turn":
            break

        tool_results = []
        for block in tool_use_blocks:
            tool_name = block.name
            tool_input = block.input

            if tool_name == "get_excel_schemas":
                cached_schemas = await get_excel_schemas_tool(excel_docs)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"schemas": cached_schemas}),
                    }
                )

            elif tool_name == "get_sample_values":
                doc_id = tool_input.get("document_id")
                samples = await get_sample_values_tool(doc_id, excel_docs)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"samples": samples}),
                    }
                )

            elif tool_name == "identify_relevant_files":
                if not cached_schemas:
                    cached_schemas = await get_excel_schemas_tool(excel_docs)
                relevant_ids = await identify_relevant_files_tool(cached_schemas, query)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"relevant_document_ids": relevant_ids}),
                    }
                )

            elif tool_name == "generate_pandas_code":
                doc_id = tool_input.get("document_id")
                code = await generate_pandas_code_tool(doc_id, query, excel_docs)
                if code:
                    cached_code[doc_id] = code
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(
                            {"pandas_code": code, "success": code is not None}
                        ),
                    }
                )

            elif tool_name == "execute_pandas_code":
                doc_id = tool_input.get("document_id")
                # Skip if this document already has a successful result
                if any(r.get("document_id") == doc_id for r in excel_results):
                    print(
                        f"[Excel Agent] Skipping duplicate execution for document_id={doc_id}"
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(
                                {
                                    "success": True,
                                    "note": "Already executed successfully for this document.",
                                }
                            ),
                        }
                    )
                    continue
                code = tool_input.get("pandas_code") or cached_code.get(doc_id)
                if not code:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(
                                {
                                    "error": "No pandas code available for this document. Call generate_pandas_code first.",
                                    "success": False,
                                }
                            ),
                        }
                    )
                    continue
                result = await execute_pandas_code_tool(doc_id, code, query, excel_docs)
                if result is not None:
                    excel_results.append(result)
                print(
                    f"[Excel Agent] execute_pandas_code returned result for {doc_id}: {result}"
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(
                            {
                                "result": result,
                                "success": result is not None,
                                "error": "Execution failed. Call generate_pandas_code again with this error as context to get corrected code."
                                if result is None
                                else "",
                            }
                        ),
                    }
                )

        messages.append({"role": "user", "content": tool_results})

    print(f"[Excel Agent] Done. {len(excel_results)} file(s) returned results.")
    return {
        "rag_result": RAGAgentResult(
            success=bool(excel_results),
            excel_results=excel_results,
            qdrant_results=[],
            context_block="No relevant context found.",
            doc_id_to_filename=doc_id_to_filename,
            confidence=1.0 if excel_results else 0.0,
            reasoning=(
                f"Excel agent returned results from {len(excel_results)} file(s)."
                if excel_results
                else "No relevant Excel data found."
            ),
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
        excel_file_types = {
            *rag_service.TABULAR_FILE_TYPES,
            "xlsx",
            "xls",
            "csv",
            "xlsb",
            "xlsm",
            "tsv",
            "ods",
        }
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

    vec_res: RAGAgentResult | None = None
    excel_res: RAGAgentResult | None = None

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
