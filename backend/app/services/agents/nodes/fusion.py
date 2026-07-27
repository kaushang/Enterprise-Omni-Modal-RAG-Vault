# Fusion node - final answer generation

import uuid
import re
import json
import logging
from app.db.session import SessionLocal
from app.services.agents.types import AgentState
from app.models.user import User
from app.models.available_model import AvailableModel
from app.services.model_router import route_model, get_default_model_config
import app.services.rag_service as rag_service
from app.core.utils import call_llm_with_fallback, extract_chart_spec


async def _resolve_model_config(
    model_id,
    db,
    user,
    query: str = "",
    context_chunks: list[str] = None,
    has_attachments: bool = False,
):
    print("[Fusion Agent] Resolving model config...")
    selected_model_string = "claude-haiku-4-5"
    selected_provider_id = "anthropic"
    model_config = None
    db_model = None

    if model_id:
        if str(model_id) == "auto":
            db_models = (
                db.query(AvailableModel)
                .filter(
                    AvailableModel.is_active,
                    (AvailableModel.tenant_id == user.tenant_id)
                    | (AvailableModel.tenant_id.is_(None)),
                )
                .all()
            )
            available_models_list = [
                {
                    "id": m.id,
                    "display_name": m.display_name,
                    "model_name": m.model_name,
                    "tier": m.tier,
                    "provider_id": m.provider_id,
                    "base_url": m.base_url,
                    "api_key": m.api_key,
                    "is_default": m.is_default,
                }
                for m in db_models
            ]

            chosen_dict = route_model(
                query=query,
                context_chunks=context_chunks or [],
                has_attachments=has_attachments,
                available_models=available_models_list,
            )
            db_model = next((m for m in db_models if m.id == chosen_dict["id"]), None)
        else:
            db_model = (
                db.query(AvailableModel).filter(AvailableModel.id == model_id).first()
            )

        if db_model:
            selected_model_string = db_model.model_name or db_model.model_string
            selected_provider_id = db_model.provider_id
            if not selected_provider_id:
                provider_val = db_model.provider
                if hasattr(provider_val, "value"):
                    provider_val = provider_val.value
                if provider_val == "anthropic":
                    selected_provider_id = "anthropic"
                elif provider_val == "openrouter":
                    selected_provider_id = "openrouter"
                else:
                    selected_provider_id = "openai_compat"
            model_config = db_model
            model_config.provider_id = selected_provider_id
            model_id = db_model.id

    if not model_config:
        model_config = AvailableModel(
            provider_id=selected_provider_id,
            model_name=selected_model_string,
            api_key="",
        )
    print(
        f"[Fusion Agent] Model resolved: {selected_model_string} via {selected_provider_id}"
    )
    return selected_model_string, selected_provider_id, model_config, db_model


logger = logging.getLogger(__name__)


async def fusion_node(state: AgentState) -> dict:
    mode = state["mode"]
    sql_result = state.get("sql_result")
    rag_result = state.get("rag_result")
    query = state["query"]
    conversation_history = state["conversation_history"]
    command_instruction = state["command_instruction"]
    model_id = state["model_id"]
    user_id = state["user_id"]

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == uuid.UUID(user_id)).first()
        if not user:
            raise ValueError("User not found")

        # Resolve model configuration
        context_chunks = []
        if rag_result:
            for hit in rag_result.qdrant_results or []:
                context_chunks.append(hit.get("payload", {}).get("chunk_text", ""))
            for er in rag_result.excel_results or []:
                context_chunks.append(str(er.get("result", "")))

        model_id_val = model_id
        if model_id_val and model_id_val != "auto" and isinstance(model_id_val, str):
            try:
                model_id_val = uuid.UUID(model_id_val)
            except Exception:
                pass

        (
            selected_model_string,
            selected_provider_id,
            model_config,
            db_model,
        ) = await _resolve_model_config(
            model_id=model_id_val,
            db=db,
            user=user,
            query=query,
            context_chunks=context_chunks,
            has_attachments=bool(state.get("document_id")),
        )

        default_model = get_default_model_config(db, user.tenant_id)

        # Degradation logic for cross_source
        actual_mode = mode
        if mode == "cross_source":
            sql_ok = bool(sql_result and sql_result.success)
            rag_ok = bool(rag_result and rag_result.success)
            if not sql_ok and not rag_ok:
                actual_mode = "doc_only"
            elif not sql_ok:
                actual_mode = "doc_only"
            elif not rag_ok:
                actual_mode = "db_only"

        # Prompt construction
        if actual_mode == "db_only":
            system_prompt = (
                "You are a helpful data analyst assistant. Your job is to analyze the executed SQL query and its returned results, "
                "and provide a clear, concise, and professional natural language answer to the user's original question. "
                "Make sure to format the output nicely (e.g. use markdown tables or bullet points where appropriate). "
                "Always cite values directly from the query results. If the results are empty, explain that no matching records were found.\n\n"
                "CHART OUTPUT RULE:\n"
                "If and only if your answer contains numerical data that can be meaningfully visualised (comparisons, trends, distributions, rankings, time series), "
                "append a chart specification at the very end of your response using this exact format:\n\n"
                'CHART_SPEC:{"chart_type":"bar","title":"<descriptive title>","x_key":"<field name>","y_keys":["<field name>"],"data":[{...},{...}]}\n\n'
                "Rules for the chart spec:\n"
                "- chart_type must be one of: bar, line, area, pie\n"
                "- Choose the most appropriate chart_type for the data\n"
                "- data must be a JSON array of flat objects, all objects must have the same keys\n"
                "- x_key is the categorical or time-based field\n"
                "- y_keys is an array of one or more numeric fields\n"
                "- For pie charts, y_keys must contain exactly one field\n"
                "- All values in y_keys fields must be numbers, not strings\n"
                "- The CHART_SPEC line must be on its own line at the very end of your response, after all text\n"
                "- Do not emit CHART_SPEC if the answer is narrative, qualitative, or contains no numerical comparisons\n"
                "- Do not wrap CHART_SPEC in markdown code fences"
            )
            sql_query = sql_result.sql_query if sql_result else ""
            formatted_results_str = sql_result.formatted_results if sql_result else ""
            prompt = f"""User Question: {query}
            Generated SQL Query: {sql_query}
            Query Results:
            {formatted_results_str}

            Please summarize and answer the user's question based on the query results. Do not make up any facts."""

        elif actual_mode == "doc_only":
            context_block = (
                rag_result.context_block if rag_result else "No relevant context found."
            )
            prompt = f"""Context:
            {context_block}

            Conversation History (recent messages):
            {conversation_history if conversation_history else "No previous messages."}

            User Question: {query}

            Answer:"""
            system_prompt = rag_service._SYSTEM_PROMPT
            if command_instruction:
                system_prompt += f"\n\n[Instructions]\n{command_instruction}"

        else:  # cross_source
            db_sql_query = sql_result.sql_query
            db_formatted_results = sql_result.formatted_results
            db_connection_name = sql_result.connection_name
            context_block = rag_result.context_block

            system_prompt = (
                "You are a helpful data analyst assistant. You have been given results from two sources: "
                "an external database (via SQL) and one or more documents or files. "
                "Your job is to answer the user's question using both sources. "
                "Decide the best format for your answer based on the question: "
                "if the question asks for a direct comparison, present both results clearly and then give a conclusion. "
                "If the question can be answered as a unified narrative using both sources, do that instead. "
                "Always cite which source each piece of information comes from. "
                "Do not make up any facts. Only use information present in the provided results."
            )

            prompt = f"""User Question: {query}

            --- Database Source: {db_connection_name} ---
            SQL Query: {db_sql_query}
            Results:
            {db_formatted_results}

            --- Document/File Source ---
            {context_block}

            Answer the user's question using both sources above."""

        print(f"[Fusion Node] Mode: {mode}")
        print(
            f"[Fusion Node] SQL result available: {bool(sql_result and sql_result.success)}"
        )
        print(
            f"[Fusion Node] RAG result available: {bool(rag_result and rag_result.success)}"
        )
        print(f"[Fusion Node] Model resolved: {selected_model_string}")
        print("[Fusion Node] Calling LLM...")

        max_tokens = 8192 if actual_mode in ("cross_source", "doc_only") else 4096

        # Call LLM with fallback
        stream_result, was_fallback, fallback_model_name = await call_llm_with_fallback(
            primary_model_config=model_config,
            default_model_config=default_model,
            call_fn=lambda cfg: rag_service._execute_llm_stream(
                cfg, system_prompt, prompt, max_tokens=max_tokens
            ),
        )

        full_answer_list = []
        input_tokens = 0
        output_tokens = 0
        async for event_type, data in stream_result:
            if event_type == "token":
                full_answer_list.append(data)
            elif event_type == "usage":
                input_tokens = data["input_tokens"]
                output_tokens = data["output_tokens"]

        full_answer = "".join(full_answer_list).strip()
        print(
            f"[Fusion Node] LLM complete. Input tokens: {input_tokens}, Output tokens: {output_tokens}"
        )

        # Resolve final active model details (for usage logging and citations)
        if was_fallback and default_model:
            db_model = default_model
            selected_model_string = (
                default_model.model_name or default_model.model_string
            )
            selected_provider_id = default_model.provider_id
        else:
            selected_model_string = model_config.model_name or model_config.model_string
            selected_provider_id = model_config.provider_id

        # Save usage log
        try:
            from app.models.usage_log import UsageLog

            usage_log = UsageLog(
                tenant_id=user.tenant_id,
                user_id=user.id,
                provider=selected_provider_id,
                model_string=selected_model_string,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            db.add(usage_log)
            db.commit()

            try:
                import sys

                if "pytest" not in sys.modules:
                    from app.tasks.billing_tasks import check_tenant_budgets_task

                    check_tenant_budgets_task.delay()
            except Exception as task_exc:
                logger.error(
                    "Failed to trigger check_tenant_budgets_task: %s", task_exc
                )
        except Exception as db_exc:
            logger.error("Failed to save usage log to database: %s", db_exc)
            db.rollback()

        # Fallback default if generation was empty
        if not full_answer:
            full_answer = (
                "I could not generate an answer. Please try rephrasing your question."
            )

        # Parse follow-up questions
        answer_parts = re.split(r"(?i)\[follow[-_]up\]", full_answer)
        cleaned_answer = answer_parts[0].strip()
        follow_up_questions = []
        if len(answer_parts) > 1:
            raw_questions = answer_parts[1].strip().split("\n")
            for q in raw_questions:
                q = q.strip().lstrip("-").lstrip("*").lstrip("123456789.").strip()
                if q:
                    follow_up_questions.append(q)

        # Extract chart spec
        cleaned_answer, chart_spec = extract_chart_spec(cleaned_answer)

        # Build Citations
        citations = []

        # 1. DB Citation (only if DB was actually used and succeeded)
        if (
            actual_mode in ("db_only", "cross_source")
            and sql_result
            and sql_result.success
        ):
            db_conn_id = state.get("database_id") or sql_result.connection_id
            db_conn_name = sql_result.connection_name or "Database"
            db_sql_query = sql_result.sql_query
            db_formatted_results = sql_result.formatted_results
            citations.append(
                {
                    "document_id": str(db_conn_id) if db_conn_id else "",
                    "filename": f"Database: {db_conn_name}",
                    "chunk_text": f"SQL: {db_sql_query}\nResults: {db_formatted_results[:1000]}...",
                    "page_number": None,
                    "slide_number": None,
                    "chunk_index": 0,
                }
            )

        # 2. RAG Citations (only if RAG was actually used and succeeded)
        if (
            actual_mode in ("doc_only", "cross_source")
            and rag_result
            and rag_result.success
        ):
            doc_id_to_filename = rag_result.doc_id_to_filename or {}
            for hit in rag_result.qdrant_results or []:
                payload = hit.get("payload", {})
                doc_id = payload.get("document_id", "")
                citations.append(
                    {
                        "document_id": doc_id,
                        "filename": doc_id_to_filename.get(doc_id, "Unknown"),
                        "chunk_text": payload.get("chunk_text", ""),
                        "page_number": payload.get("page_number"),
                        "slide_number": payload.get("slide_number"),
                        "chunk_index": payload.get("chunk_index", 0),
                    }
                )
            for er in rag_result.excel_results or []:
                citations.append(
                    {
                        "document_id": str(er.get("document_id", "")),
                        "filename": er.get("filename", "Unknown"),
                        "chunk_text": f"Query: {query}\nResult: {str(er.get('result', ''))[:1000]}",
                        "page_number": None,
                        "slide_number": None,
                        "chunk_index": 0,
                    }
                )

        print(
            f"[Fusion Node] Done. Answer length: {len(cleaned_answer)} chars, Citations: {len(citations)}"
        )

        return {
            "final_answer": cleaned_answer,
            "citations": citations,
            "follow_up_questions": follow_up_questions,
            "chart_spec": chart_spec,
            "generated_sql": sql_result.sql_query if sql_result else None,
            "query_results": json.loads(
                json.dumps(sql_result.query_results, default=str)
            )
            if sql_result and sql_result.query_results
            else None,
            "model_string": selected_model_string,
            "resolved_model": db_model.display_name
            if db_model
            else selected_model_string,
            "resolved_model_id": str(db_model.id) if db_model else None,
            "was_fallback": was_fallback,
            "fallback_model_name": fallback_model_name,
            "execution_time_ms": sql_result.execution_time_ms if sql_result else 0,
            "db_connection_id": str(sql_result.connection_id)
            if sql_result and sql_result.connection_id
            else None,
        }

    finally:
        db.close()
