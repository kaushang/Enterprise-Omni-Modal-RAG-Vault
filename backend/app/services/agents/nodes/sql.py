# SQL agent node and SQL judge node

import json
import re
import uuid

import app.services.rag_service as rag_service
from app.core.config import settings
from app.db.session import SessionLocal
from app.models.external_database import (
    DatabaseAccessPolicy,
    DatabaseSchemaCache,
    ExternalDatabaseConnection,
)
from app.models.user import User
from app.services import database_service
from app.services.agents.tools.sql_tools import (
    SCHEMA_INTELLIGENCE_TOOLS,
    SQL_GENERATION_TOOLS,
    SQL_JUDGE_TOOLS,
    _execute_schema_tool,
    check_filters,
    check_optimization,
    check_semantic_alignment,
    execute_sql,
    generate_sql,
    validate_sql,
)
from app.services.agents.types import AgentState, JudgeResult, SQLAgentResult

logger = rag_service.logger


def get_db_session():
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise


async def gather_sql_context(state: AgentState, db) -> dict:
    """Plain async function to gather database context upfront before graph runs."""
    try:
        user_id = state.get("user_id")
        if not user_id:
            return {
                "sql_result": SQLAgentResult(success=False, error="User ID is missing"),
                "context_error": "User ID is missing",
            }
        user_id_uuid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id
        user = db.query(User).filter(User.id == user_id_uuid).first()
        if not user:
            return {
                "sql_result": SQLAgentResult(success=False, error="User not found"),
                "context_error": "User not found",
            }

        database_id = state.get("database_id")
        if not database_id:
            return {
                "sql_result": SQLAgentResult(
                    success=False, error="Database ID is missing"
                ),
                "context_error": "Database ID is missing",
            }
        database_id_uuid = (
            uuid.UUID(database_id) if isinstance(database_id, str) else database_id
        )

        connection = (
            db.query(ExternalDatabaseConnection)
            .filter(
                ExternalDatabaseConnection.id == database_id_uuid,
                ExternalDatabaseConnection.tenant_id == user.tenant_id,
            )
            .first()
        )
        if not connection:
            return {
                "sql_result": SQLAgentResult(
                    success=False, error="Database connection not found"
                ),
                "context_error": "Database connection not found",
            }

        if not database_service.check_user_db_access(db, user, database_id_uuid):
            return {
                "sql_result": SQLAgentResult(
                    success=False, error="Database access denied"
                ),
                "context_error": "Database access denied",
            }

        schema_cache = (
            db.query(DatabaseSchemaCache)
            .filter(DatabaseSchemaCache.connection_id == database_id_uuid)
            .first()
        )
        if not schema_cache or not schema_cache.schema_data:
            return {
                "sql_result": SQLAgentResult(
                    success=False, error="Database schema missing"
                ),
                "context_error": "Database schema missing",
            }

        all_tables = [t["name"] for t in schema_cache.schema_data.get("tables", [])]
        authorized_table_names = database_service.get_user_authorized_tables(
            db, user, database_id_uuid, all_tables
        )
        if not authorized_table_names:
            return {
                "sql_result": SQLAgentResult(
                    success=False, error="No authorized tables"
                ),
                "context_error": "No authorized tables",
            }

        policies = []
        if not user.role.is_admin:
            policies = (
                db.query(DatabaseAccessPolicy)
                .filter(
                    DatabaseAccessPolicy.connection_id == database_id_uuid,
                    DatabaseAccessPolicy.role_id == user.role_id,
                )
                .all()
            )

        authorized_cols_by_table = {}
        all_physical_cols_by_table = {}
        valid_tables = {
            t["name"].lower() for t in schema_cache.schema_data.get("tables", [])
        }
        authorized_tables_info = []

        for t in schema_cache.schema_data.get("tables", []):
            t_name = t["name"]
            if t_name in authorized_table_names:
                all_cols = [c["name"] for c in t.get("columns", [])]
                all_physical_cols_by_table[t_name.lower()] = {
                    c.lower() for c in all_cols
                }
                if user.role.is_admin:
                    auth_cols = {c.lower() for c in all_cols}
                else:
                    auth_cols = database_service.get_user_authorized_columns_for_table(
                        policies, t_name, all_cols
                    )

                if auth_cols or user.role.is_admin:
                    authorized_cols_by_table[t_name.lower()] = auth_cols
                    tbl_copy = dict(t)
                    tbl_copy["columns"] = [
                        col
                        for col in t.get("columns", [])
                        if col["name"].lower() in auth_cols
                    ]
                    authorized_tables_info.append(tbl_copy)

        if not authorized_tables_info:
            return {
                "sql_result": SQLAgentResult(
                    success=False, error="No authorized columns"
                ),
                "context_error": "No authorized columns",
            }

        session_id = state.get("session_id")
        session_id_uuid = (
            uuid.UUID(session_id)
            if isinstance(session_id, str) and session_id
            else session_id
        )
        turns = []
        if session_id_uuid:
            turns = rag_service.get_recent_turns(
                db, session_id_uuid, settings.SQL_HISTORY_LIMIT
            )

        return {
            "db_user_id": str(user.id),
            "db_authorized_schema": {"tables": authorized_tables_info},
            "db_authorized_cols_by_table": authorized_cols_by_table,
            "db_all_physical_cols_by_table": all_physical_cols_by_table,
            "db_valid_tables": valid_tables,
            "db_connection_engine": connection.engine,
            "db_connection_name": connection.name,
            "db_connection_id": str(connection.id),
            "db_session_turns": turns,
            "db_is_admin": user.role.is_admin if user.role else False,
            "context_error": None,
        }
    except Exception as exc:
        return {
            "sql_result": SQLAgentResult(success=False, error=str(exc)),
            "context_error": str(exc),
        }


def _parse_schema_selection_json(text: str) -> dict | None:
    # Return None if the response is empty.
    if not text:
        return None

    # Remove leading/trailing whitespace from the response.
    cleaned = text.strip()

    # Remove Markdown code block markers if the JSON is wrapped in them.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Try parsing the cleaned text directly as JSON.
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "selected_tables" in data:
            return data
    except Exception:
        pass

    # As a fallback, extract the first JSON object from the response.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        # Try parsing the extracted JSON object.
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and "selected_tables" in data:
                return data
        except Exception:
            pass

    # Return None if no valid schema selection JSON could be parsed.
    return None


async def schema_selection_node(state: AgentState) -> dict:
    """
    Schema Intelligence Agent following the ReAct pattern.
    Understands user intent and dynamically discovers database schema using tools
    until it has gathered sufficient context to stop.
    """
    if state.get("context_error"):
        return {}

    user_query = state.get("query", "")
    authorized_schema = state.get("db_authorized_schema") or {}
    authorized_tables = authorized_schema.get("tables", [])

    if not authorized_tables:
        print("[Schema Intelligence Agent] No authorized tables available.")
        return {"db_filtered_schema": {"tables": []}}

    history_turns = state.get("db_session_turns") or []
    history_str = ""

    # converting history turns to a string representation for the prompt
    if history_turns:
        history_str = (
            f"\nPrior Conversation Context:\n{json.dumps(history_turns, default=str)}\n"
        )

    system_prompt = (
        "You are a Schema Intelligence Agent operating in a ReAct (Reasoning + Acting) loop.\n"
        "Your objective is to identify the minimum set of database tables required for another agent to generate correct SQL for the user's query.\n"
        "You have access to tools for discovering database schema. At every step, reason about what information you currently have, what information is missing, and whether a tool call is necessary.\n\n"
        "Available tools:\n"
        "1. get_all_table_names\n"
        "   Returns all authorized table names.\n"
        "2. get_table_schema(table_names)\n"
        "   Returns the schema for the specified tables.\n"
        "3. get_all_tables_schema\n"
        "   Returns the schema for every authorized table.\n\n"
        "Rules:\n"
        "- Use tools only when they help you make a better decision.\n"
        "- Avoid unnecessary tool calls and avoid retrieving more schema than required.\n"
        "- When you are confident you have gathered enough schema context to answer the query, DO NOT call any more tools.\n"
        "- End your turn by returning ONLY a valid JSON object in this exact format with no extra commentary:\n"
        "{\n"
        '    "selected_tables": ["table1", "table2"],\n'
        '    "reasoning": "Short explanation of why these tables were chosen."\n'
        "}\n\n"
        "Do not continue exploring after reaching sufficient confidence."
    )

    user_prompt = (
        f"User Query: {user_query}\n"
        f"{history_str}\n"
        "Discover the schema step-by-step using tools and select only the required tables."
    )

    messages = [{"role": "user", "content": user_prompt}]
    selected_table_names = []
    agent_reasoning = ""
    reformat_attempted = False

    try:
        client = rag_service._get_async_anthropic_client()
        max_turns = 7

        for turn in range(max_turns):
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
                tools=SCHEMA_INTELLIGENCE_TOOLS,
            )

            # Store Claude's response
            assistant_content = response.content

            # Save conversation history
            messages.append({"role": "assistant", "content": assistant_content})

            # Extract text blocks for logging
            text_blocks = [
                b.text
                for b in assistant_content
                if isinstance(getattr(b, "text", None), str) and b.text
            ]
            if text_blocks:
                print(
                    f"[Schema Intelligence Agent] ReAct Thought (Turn {turn + 1}): {' '.join(text_blocks)}"
                )

            # Find tool calls
            tool_use_blocks = [
                block
                for block in assistant_content
                if getattr(block, "type", None) == "tool_use"
            ]

            # Check whether it is finished
            if (
                not tool_use_blocks
                or getattr(response, "stop_reason", None) == "end_turn"
            ):
                full_text = " ".join(text_blocks)
                parsed = _parse_schema_selection_json(full_text)

                if parsed and "selected_tables" in parsed:
                    selected_table_names = parsed.get("selected_tables", [])
                    agent_reasoning = parsed.get("reasoning", "")
                    break

                # If JSON is invalid and we haven't tried reformatting yet, ask the model to reformat
                elif not reformat_attempted:
                    reformat_attempted = True
                    print(
                        "[Schema Intelligence Agent] Failed to parse JSON final answer. Prompting model to reformat."
                    )

                    # Ask LLM to reformat
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response could not be parsed as valid JSON. "
                                "Please respond ONLY with a valid JSON object with keys 'selected_tables' (list of strings) and 'reasoning' (string)."
                            ),
                        }
                    )
                    continue
                else:
                    break

            tool_results = []

            # Loop through every requested tool and execute it
            for block in tool_use_blocks:
                tool_name = block.name
                tool_args = block.input or {}
                print(
                    f"[Schema Intelligence Agent] Tool Call (Turn {turn + 1}): {tool_name}({tool_args})"
                )

                res_str = _execute_schema_tool(tool_name, tool_args, authorized_tables)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": res_str,
                    }
                )

            # Give tool result back to Claude
            messages.append({"role": "user", "content": tool_results})

    except Exception as exc:
        logger.warning(
            f"[Schema Intelligence Agent] Exception during ReAct loop: {exc}"
        )

    # Process and filter final schema based on agent decision
    selected_set = {t.lower() for t in selected_table_names}
    filtered_tables = [
        t for t in authorized_tables if t.get("name", "").lower() in selected_set
    ]

    # If LLM somehow selected no valid tables, fall back to Entire authorized schema
    if not filtered_tables:
        print(
            "[Schema Intelligence Agent] No valid tables selected or matched; falling back to full authorized schema."
        )
        filtered_schema = authorized_schema
    else:
        filtered_schema = {"tables": filtered_tables}

    chosen_names = [t.get("name") for t in filtered_schema.get("tables", [])]
    print(f"[Schema Intelligence Agent] Final Selected Tables: {chosen_names}")
    if agent_reasoning:
        print(f"[Schema Intelligence Agent] Reasoning: {agent_reasoning}")

    progress_msg = f"**Schema Intelligence Agent:** Selected table(s) `{', '.join(chosen_names)}` for query resolution.\n\n"

    return {
        "db_filtered_schema": filtered_schema,
        "progress_tokens": [progress_msg],
    }


async def sql_generation_node(state: AgentState) -> dict:
    if state.get("context_error"):
        return {
            "sql_result": SQLAgentResult(success=False, error=state["context_error"]),
            "sql_generation_error": state["context_error"],
        }

    db = get_db_session()
    try:
        query = state["query"]
        filtered_schema = (
            state.get("db_filtered_schema") or state.get("db_authorized_schema") or {}
        )
        engine_type = state["db_connection_engine"]
        conversation_history = state.get("db_session_turns")
        is_admin = state.get("db_is_admin", False)

        authorized_cols_by_table = state.get("db_authorized_cols_by_table", {})
        valid_tables = state.get("db_valid_tables", set())
        all_physical_cols_by_table = state.get("db_all_physical_cols_by_table", {})

        system_prompt = (
            "You are a SQL Agent operating in a ReAct (Reasoning + Acting) loop.\n"
            "Your responsibility is to answer the user's database question by producing correct, authorized, and executable SQL.\n\n"
            "You have three tools:\n"
            "1. generate_sql - generates SQL from the user query and schema. Pass failed_sql and error_message if correcting a previous attempt.\n"
            "2. validate_sql - checks whether the SQL accesses only authorized tables and columns.\n"
            "3. execute_sql - executes the SQL against the database and returns results.\n\n"
            "At each step, reason about what you know so far and decide what to do next.\n"
            "You decide the order, when to retry, and when you are confident enough to stop.\n\n"
            "Constraints:\n"
            "- Maximum 3 generate_sql attempts. Stop after 3 regardless of outcome.\n"
            "- When execution succeeds, stop and return ONLY this JSON with no extra text:\n"
            '{"final_sql": "the executed SQL", "results": "<first 10 rows only>", "execution_time_ms": 0}\n'
            "- The full result set is already stored externally. Include at most 10 rows in your response.\n"
            "- If you cannot produce a working query after all attempts, return:\n"
            '{"final_sql": null, "results": null, "error": "reason why SQL could not be executed"}'
        )

        user_prompt = (
            f"User Query: {query}\n"
            f"Schema: {json.dumps(filtered_schema)}\n"
            f"Engine: {engine_type}"
        )
        judge_res = state.get("judge_result")
        if judge_res:
            feedback = getattr(judge_res, "retry_feedback", None) or (
                judge_res.get("retry_feedback") if isinstance(judge_res, dict) else ""
            )
            if feedback:
                user_prompt += f"\n\nPrevious SQL Feedback from Judge:\n{feedback}"

            crit_hints = (
                getattr(judge_res, "critical_optimization_hints", [])
                if hasattr(judge_res, "critical_optimization_hints")
                else (
                    judge_res.get("critical_optimization_hints", [])
                    if isinstance(judge_res, dict)
                    else []
                )
            )
            if crit_hints:
                user_prompt += (
                    "\nCritical optimization issues to fix in the rewritten SQL:\n"
                )
                for hint in crit_hints:
                    user_prompt += f"- {hint}\n"

        messages = [{"role": "user", "content": user_prompt}]
        final_sql = None
        final_results = None
        execution_time_ms = 0
        agent_error = None
        last_executed_sql = None
        last_executed_results = None
        last_execution_time_ms = 0
        max_turns = 8

        client = rag_service._get_async_anthropic_client()

        for turn in range(max_turns):
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                system=system_prompt,
                tools=SQL_GENERATION_TOOLS,
                messages=messages,
            )

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            print(f"[SQL Agent] Turn {turn + 1}, stop_reason: {response.stop_reason}")

            # Agent decided it is done
            if response.stop_reason == "end_turn":
                text_blocks = [
                    b.text
                    for b in assistant_content
                    if getattr(b, "type", None) == "text" and b.text
                ]
                full_text = " ".join(text_blocks).strip()
                match = re.search(r"\{.*\}", full_text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    final_sql = parsed.get("final_sql")
                    final_results = parsed.get("results")
                    execution_time_ms = parsed.get("execution_time_ms", 0)
                    agent_error = parsed.get("error")
                break

            # Execute tool calls
            tool_use_blocks = [
                b for b in assistant_content if getattr(b, "type", None) == "tool_use"
            ]

            if not tool_use_blocks:
                break

            tool_results = []
            for block in tool_use_blocks:
                tool_name = block.name
                tool_args = block.input or {}
                print(f"[SQL Agent] Tool Call: {tool_name}({tool_args})")

                if tool_name == "generate_sql":
                    try:
                        result = await generate_sql(
                            query=query,
                            schema=filtered_schema,
                            engine_type=engine_type,
                            conversation_history=conversation_history,
                            failed_sql=tool_args.get("failed_sql"),
                            error_message=tool_args.get("error_message"),
                        )
                        res_str = json.dumps({"sql": result})
                        print(f"[SQL Agent] Generated SQL: {result}")
                    except Exception as exc:
                        res_str = json.dumps({"error": str(exc)})
                        print(f"[SQL Agent] generate_sql failed: {exc}")

                elif tool_name == "validate_sql":
                    sql_to_validate = tool_args.get("sql", "")
                    if is_admin:
                        res_str = json.dumps({"valid": True, "reason": ""})
                    else:
                        res_str = validate_sql(
                            sql=sql_to_validate,
                            engine_type=engine_type,
                            authorized_cols_by_table=authorized_cols_by_table,
                            valid_tables=valid_tables,
                            all_physical_cols_by_table=all_physical_cols_by_table,
                        )
                    print(f"[SQL Agent] Validation result: {res_str}")

                elif tool_name == "execute_sql":
                    sql_to_execute = tool_args.get("sql", "")
                    res_str = execute_sql(
                        sql=sql_to_execute,
                        connection_id=state["db_connection_id"],
                        db=db,
                    )
                    try:
                        res_data = json.loads(res_str)
                        if res_data.get("success"):
                            last_executed_sql = sql_to_execute
                            last_executed_results = res_data.get("rows")
                            last_execution_time_ms = res_data.get(
                                "execution_time_ms", 0
                            )
                    except Exception:
                        pass

                else:
                    res_str = json.dumps({"error": f"Unknown tool: {tool_name}"})

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": res_str,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

    except Exception as exc:
        logger.warning(f"[SQL Agent] ReAct loop exception: {exc}")
        agent_error = str(exc)
    finally:
        db.close()

    # Fallback to last successful execution if final_sql wasn't set by end_turn JSON
    if not final_sql and last_executed_sql:
        final_sql = last_executed_sql
        final_results = last_executed_results
        execution_time_ms = last_execution_time_ms

    if not final_sql:
        print(f"[SQL Agent] Failed to produce SQL. Error: {agent_error}")
        return {
            "generated_sql": None,
            "sql_generation_error": agent_error or "SQL generation failed",
            "sql_generation_attempts": state["sql_generation_attempts"] + 1,
            "sql_result": SQLAgentResult(
                success=False,
                error=agent_error or "SQL generation failed",
                attempts=state["sql_generation_attempts"] + 1,
            ),
        }

    print(f"[SQL Agent] Final SQL: {final_sql}")
    return {
        "generated_sql": final_sql,
        "sql_generation_error": None,
        "sql_generation_attempts": state["sql_generation_attempts"] + 1,
        "sql_result": SQLAgentResult(
            success=True,
            sql_query=final_sql,
            query_results=final_results,
            formatted_results=json.dumps(final_results, indent=2, default=str),
            connection_name=state.get("db_connection_name"),
            connection_id=state.get("db_connection_id"),
            execution_time_ms=execution_time_ms,
            attempts=state["sql_generation_attempts"] + 1,
        ),
        "progress_tokens": [
            f"**Generated SQL:**\n```sql\n{final_sql}\n```\n\n*Executing...*\n\n"
        ],
    }


def _parse_judge_json(text: str) -> dict | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "passed" in data:
            return data
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and "passed" in data:
                return data
        except Exception:
            pass

    return None


async def sql_judge_node(state: AgentState) -> dict:
    """
    SQL Judge Agent following the ReAct pattern.
    Evaluates generated SQL against user query using three tools:
    check_semantic_alignment, check_filters, check_optimization.
    """
    if state.get("sql_generation_error") or not state.get("generated_sql"):
        return {}
    db = get_db_session()

    try:
        query = state.get("query", "")
        sql = state.get("generated_sql", "")
        schema_context = json.dumps(
            state.get("db_filtered_schema") or state.get("db_authorized_schema") or {}
        )

        system_prompt = (
            "You are a SQL Judge Agent operating in a ReAct (Reasoning + Acting) loop.\n"
            "Your job is to evaluate the generated SQL against the user's natural language query using the available tools.\n\n"
            "Available tools:\n"
            "1. check_semantic_alignment(query, sql, schema_context)\n"
            "   Checks if SQL semantically matches user intent.\n"
            "2. check_filters(query, sql)\n"
            "   Checks if all required filters are present in SQL.\n"
            "3. check_optimization(sql)\n"
            "   Scans the SQL for anti-patterns and performance issues. Returns critical_optimization_hints and advisory_optimization_hints.\n\n"
            "Rules:\n"
            "- You MUST call all three tools before making your final judgment.\n"
            "- Reason step-by-step about what tools to call.\n"
            "- End your turn by returning ONLY a valid JSON object in this exact format with no extra commentary:\n"
            "{\n"
            '    "passed": true,\n'
            '    "semantic_score": 0.95,\n'
            '    "failed_filters": [],\n'
            '    "critical_optimization_hints": [],\n'
            '    "optimization_hints": [],\n'
            '    "retry_feedback": ""\n'
            "}\n\n"
            "If passed is false or critical_optimization_hints is non-empty, retry_feedback must be a clear instruction string for the SQL generation agent listing what to fix."
        )

        user_prompt = (
            f"User Query: {query}\n\n"
            f"Generated SQL: {sql}\n\n"
            f"Schema Context:\n{schema_context}\n\n"
            "Evaluate this generated SQL using all three available tools and return your judgment JSON."
        )

        messages = [{"role": "user", "content": user_prompt}]
        reformat_attempted = False
        parsed_result = None

        client = rag_service._get_async_anthropic_client()
        max_turns = 6

        for turn in range(max_turns):
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
                tools=SQL_JUDGE_TOOLS,
            )

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            text_blocks = [
                b.text
                for b in assistant_content
                if isinstance(getattr(b, "text", None), str) and b.text
            ]
            if text_blocks:
                print(
                    f"[SQL Judge Agent] ReAct Thought (Turn {turn + 1}): {' '.join(text_blocks)}"
                )

            tool_use_blocks = [
                block
                for block in assistant_content
                if getattr(block, "type", None) == "tool_use"
            ]

            if not tool_use_blocks:
                full_text = " ".join(text_blocks)
                parsed = _parse_judge_json(full_text)

                if parsed and "passed" in parsed:
                    parsed_result = parsed
                    break

                elif not reformat_attempted:
                    reformat_attempted = True
                    print(
                        "[SQL Judge Agent] Failed to parse JSON final answer. Prompting model to reformat."
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response could not be parsed as valid JSON. "
                                "Please respond ONLY with a valid JSON object with keys "
                                "'passed' (boolean), 'semantic_score' (float), 'failed_filters' (list of strings), "
                                "'critical_optimization_hints' (list of strings), "
                                "'optimization_hints' (list of strings), and 'retry_feedback' (string)."
                            ),
                        }
                    )
                    continue
                else:
                    break

            tool_results = []
            for block in tool_use_blocks:
                tool_name = block.name
                tool_args = block.input or {}
                print(
                    f"[SQL Judge Agent] Tool Call (Turn {turn + 1}): {tool_name}({tool_args})"
                )

                if tool_name == "check_semantic_alignment":
                    t_query = tool_args.get("query") or query
                    t_sql = tool_args.get("sql") or sql
                    t_schema = tool_args.get("schema_context") or schema_context
                    sem_passed, sem_score = await check_semantic_alignment(
                        t_query, t_sql, t_schema
                    )
                    res_str = json.dumps(
                        {"passed": sem_passed, "confidence": sem_score}
                    )

                elif tool_name == "check_filters":
                    t_query = tool_args.get("query") or query
                    t_sql = tool_args.get("sql") or sql
                    missing = check_filters(t_query, t_sql)
                    res_str = json.dumps({"failed_filters": missing})

                elif tool_name == "check_optimization":
                    t_sql = tool_args.get("sql") or sql
                    critical_hints, advisory_hints = await check_optimization(
                        sql=t_sql,
                        engine_type=state.get("db_connection_engine", "postgresql"),
                        db=db,
                        connection_id=state.get("db_connection_id"),
                    )
                    res_str = json.dumps(
                        {
                            "critical_optimization_hints": critical_hints,
                            "advisory_optimization_hints": advisory_hints,
                        }
                    )

                else:
                    res_str = json.dumps({"error": f"Unknown tool: {tool_name}"})

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": res_str,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        current_retry_count = state.get("sql_judge_retry_count", 0)
        if parsed_result:
            judge_result = JudgeResult(
                passed=bool(parsed_result.get("passed", False)),
                semantic_score=float(parsed_result.get("semantic_score", 0.0)),
                failed_filters=list(parsed_result.get("failed_filters", [])),
                critical_optimization_hints=list(
                    parsed_result.get("critical_optimization_hints", [])
                ),
                optimization_hints=list(parsed_result.get("optimization_hints", [])),
                retry_feedback=str(parsed_result.get("retry_feedback", "")),
            )
            print(
                f"[SQL Judge Agent] Judgment complete. Passed: {judge_result.passed}, Score: {judge_result.semantic_score:.2f}"
            )
            return {
                "judge_result": judge_result,
                "sql_judge_retry_count": current_retry_count + 1,
            }
        else:
            print("[SQL Judge Agent] ReAct loop completed without valid judgment JSON.")
            return {"sql_judge_retry_count": current_retry_count}

    except Exception as exc:
        logger.warning(f"[SQL Judge Agent] ReAct loop exception: {exc}")
        print(f"[SQL Judge Agent] Exception encountered: {exc}")
        return {"sql_judge_retry_count": state.get("sql_judge_retry_count", 0)}
    finally:
        db.close()
