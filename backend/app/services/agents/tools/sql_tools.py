import json
import re
import time
import uuid
from typing import Optional

import app.services.rag_service as rag_service
from app.models.external_database import (
    ExternalDatabaseConnection,
)
from app.services import database_service

logger = rag_service.logger


SCHEMA_INTELLIGENCE_TOOLS = [
    {
        "name": "get_all_table_names",
        "description": "Returns only the names of every table the user is authorized to access. Contains no column details. Very cheap.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_table_schema",
        "description": "Returns the complete schema (columns, types, keys) only for the specified table names.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of table names to get schema for, e.g. ['orders', 'customers']",
                }
            },
            "required": ["table_names"],
        },
    },
    {
        "name": "get_all_tables_schema",
        "description": "Returns the complete schema of every authorized table. This is expensive and should only be used if selective exploration is insufficient.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


SQL_GENERATION_TOOLS = [
    {
        "name": "generate_sql",
        "description": (
            "Generates a SQL query from the user's natural language question using the provided schema. "
            "Pass failed_sql and error_message if a previous attempt failed, so the model can correct it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "failed_sql": {
                    "type": "string",
                    "description": "The previously generated SQL that failed. Omit on first attempt.",
                },
                "error_message": {
                    "type": "string",
                    "description": "The error from the previous attempt. Omit on first attempt.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "validate_sql",
        "description": (
            "Validates whether the generated SQL only accesses tables and columns the user is authorized to query. "
            "Always call this after generate_sql before proceeding."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL query to validate.",
                }
            },
            "required": ["sql"],
        },
    },
    {
        "name": "execute_sql",
        "description": (
            "Executes the validated SQL query against the database and returns the results. "
            "Only call this after validate_sql has confirmed the SQL is authorized. "
            "If execution fails, use the error to call generate_sql again with the failed SQL and error message."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The validated SQL query to execute.",
                }
            },
            "required": ["sql"],
        },
    },
]


SQL_JUDGE_TOOLS = [
    {
        "name": "check_semantic_alignment",
        "description": "Checks whether the generated SQL semantically matches the user's natural language query. Returns a pass/fail verdict and a confidence score.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The original natural language query",
                },
                "sql": {
                    "type": "string",
                    "description": "The generated SQL to evaluate",
                },
                "schema_context": {
                    "type": "string",
                    "description": "JSON string of the schema context",
                },
            },
            "required": ["query", "sql", "schema_context"],
        },
    },
    {
        "name": "check_filters",
        "description": "Checks whether all filters and constraints mentioned in the user query are present in the generated SQL. Returns a list of missing filters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The original natural language query",
                },
                "sql": {
                    "type": "string",
                    "description": "The generated SQL to evaluate",
                },
            },
            "required": ["query", "sql"],
        },
    },
    {
        "name": "check_optimization",
        "description": "Scans the SQL for anti-patterns and performance issues. Returns a list of optimization hints.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The generated SQL to evaluate",
                }
            },
            "required": ["sql"],
        },
    },
]


async def generate_sql(
    query: str,
    schema: dict,
    engine_type: str,
    conversation_history: Optional[list],
    failed_sql: Optional[str] = None,
    error_message: Optional[str] = None,
) -> str:
    # Conditional rule for case-insensitive comparison based on the database engine type
    rule_case = (
        "9. For equality filters against TEXT/VARCHAR columns, ALWAYS use ILIKE instead of = (e.g. col ILIKE 'value'). Do not apply to non-text columns."
        if engine_type == "postgresql"
        else "9. For equality filters against TEXT/VARCHAR columns, use case-insensitive comparison via LIKE or LOWER(). Do not apply to non-text columns."
    )

    schema_str = json.dumps(schema)
    history_str = (
        f"\nPrior Conversation Context:\n{json.dumps(conversation_history)}\n"
        if conversation_history
        else ""
    )

    # If access is denied, instruct the model to find an alternative query using only authorized tables/columns.
    if failed_sql and error_message and "access denied" in error_message.lower():
        system_prompt = (
            "You are an expert SQL translation assistant. You previously generated a SQL query that accessed an unauthorized table or column (Access Denied).\n"
            "Your task is to correct the SQL query by looking for an alternative path or query structure using other tables or columns in the schema that are authorized to answer the question.\n"
            "Follow these strict rules:\n"
            "1. Output ONLY the raw SQL query. Do not wrap it in markdown code blocks, do not add comments, and do not write any introductory or explanatory text.\n"
            f"2. Use {engine_type} SQL syntax.\n"
            "3. Pay attention to case-sensitivity and quote identifiers correctly if needed (e.g. backticks for MySQL, double quotes for PostgreSQL).\n"
            "4. Only query the tables and columns listed in the schema. Do not invent columns.\n"
            "5. ALWAYS add a LIMIT or TOP clause of 100 to prevent returning too many rows, unless the query is an aggregation (COUNT, SUM, AVG).\n"
            "6. Do NOT output any write queries (INSERT, UPDATE, DELETE, DROP, ALTER). The query must be purely read-only (SELECT).\n"
            "7. Use the 'Prior Conversation Context' to resolve references to concrete values or conditions in the SQL query. If no context exists, treat the question as standalone and generate the best possible query from the schema alone.\n"
            "8. You MUST NOT return any workaround (such as returning dummy/constant values, placeholder fields, or inventing columns not present in the schema). You must strictly return what's being asked. If there is no alternative way to answer the question using only the allowed tables and columns, you must output 'I cannot generate a SQL query, this is ambiguous'.\n"
            f"{rule_case}\n"
            "10. When filtering on a column that has allowed_values listed in the schema, you MUST use the exact canonical value from that list. Do not guess, infer, or change the casing of enum values."
        )

        prompt = f"""Database Schema (Engine: {engine_type}):
        {schema_str}
        {history_str}
        User Question: {query}

        Previously Generated SQL:
        {failed_sql}

        Database Error Message:
        {error_message}

        The previously generated query failed because it accessed an unauthorized table or column. Please find another way to query the database using other authorized tables or columns to answer the question. Do NOT use any unauthorized columns or tables mentioned in the error message. Do NOT return any workarounds (e.g. using dummy/constant values, placeholder fields, or inventing columns). You must strictly return what is asked. If there is no other way to answer the question, output "I cannot generate a SQL query, this is ambiguous".

        SQL Query:"""

        # Instruct the model to correct the SQL query based on the error message and user's original question.
    elif failed_sql:
        system_prompt = (
            "You are an expert SQL translation assistant. You previously generated a SQL query that failed with a database error.\n"
            "Your task is to correct the SQL query based on the database error message and user's original question. Follow these strict rules:\n"
            "1. Output ONLY the raw SQL query. Do not wrap it in markdown code blocks, do not add comments, and do not write any introductory or explanatory text.\n"
            f"2. Use {engine_type} SQL syntax.\n"
            "3. Pay attention to case-sensitivity and quote identifiers correctly if needed (e.g. backticks for MySQL, double quotes for PostgreSQL).\n"
            "4. Only query the tables and columns listed in the schema. Do not invent columns.\n"
            "5. ALWAYS add a LIMIT or TOP clause of 100 to prevent returning too many rows, unless the query is an aggregation (COUNT, SUM, AVG).\n"
            "6. Do NOT output any write queries (INSERT, UPDATE, DELETE, DROP, ALTER). The query must be purely read-only (SELECT).\n"
            "7. Use the 'Prior Conversation Context' to resolve references (pronouns like 'he', 'she', 'it', or phrases like 'that document', 'the same region', 'last quarter') to concrete values or conditions in the SQL query. If no context exists, treat the question as standalone and generate the best possible query from the schema alone.\n"
            "8. Only return 'I cannot generate a SQL query, this is ambiguous' as an absolute last resort - specifically when the question refers to something that has multiple equally valid interpretations AND the schema provides no way to distinguish between them, AND there is no conversation context to resolve it. A question that is broad or open-ended (e.g. 'show me costs', 'which is the worst performing') is NOT ambiguous - map it to the most natural columns in the schema and generate a query. When in doubt, generate a query.\n"
            f"{rule_case}\n"
            "10. When filtering on a column that has allowed_values listed in the schema, you MUST use the exact canonical value from that list. Do not guess, infer, or change the casing of enum values."
        )

        prompt = f"""Database Schema (Engine: {engine_type}):
        {schema_str}
        {history_str}
        User Question: {query}

        Previously Generated SQL:
        {failed_sql}

        Database Error Message:
        {error_message}

        Please correct the SQL query to fix the value/literal mismatch or enum issue reported in the error message. Ensure the SQL is completely valid and follows all rules.

        SQL Query:"""
        # General system prompt
    else:
        system_prompt = (
            "You are an expert SQL translation assistant. Your task is to translate the user's natural language question "
            "into a valid, executable SQL query for the given database schema. Follow these strict rules:\n"
            "1. Output ONLY the raw SQL query. Do not wrap it in markdown code blocks, do not add comments, and do not write any introductory or explanatory text.\n"
            f"2. Use {engine_type} SQL syntax.\n"
            "3. Pay attention to case-sensitivity and quote identifiers correctly if needed (e.g. backticks for MySQL, double quotes for PostgreSQL).\n"
            "4. Only query the tables and columns listed in the schema. Do not invent columns.\n"
            "5. ALWAYS add a LIMIT or TOP clause of 100 to prevent returning too many rows, unless the query is an aggregation (COUNT, SUM, AVG).\n"
            "6. Do NOT output any write queries (INSERT, UPDATE, DELETE, DROP, ALTER). The query must be purely read-only (SELECT).\n"
            "7. Use the 'Prior Conversation Context' to resolve references (pronouns like 'he', 'she', 'it', or phrases like 'that document', 'the same region', 'last quarter') to concrete values or conditions in the SQL query. If no context exists, treat the question as standalone and generate the best possible query from the schema alone.\n"
            "8. Only return 'I cannot generate a SQL query, this is ambiguous' as an absolute last resort - specifically when the question refers to something that has multiple equally valid interpretations AND the schema provides no way to distinguish between them, AND there is no conversation context to resolve it. A question that is broad or open-ended (e.g. 'show me costs', 'which is the worst performing') is NOT ambiguous - map it to the most natural columns in the schema and generate a query. When in doubt, generate a query.\n"
            f"{rule_case}\n"
            "10. When filtering on a column that has allowed_values listed in the schema, you MUST use the exact canonical value from that list because the database requires exact matches. Do not guess, infer, or change the casing of enum values."
        )

        prompt = f"""Database Schema (Engine: {engine_type}):
        {schema_str}
        {history_str}
        User Question: {query}

        SQL Query:"""

    # Call the LLM to generate SQL
    print(f"[SQL Generation] Prompting LLM with query: {query}")
    client = rag_service._get_async_anthropic_client()
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )
    sql = response.content[0].text.strip()

    print(f"[SQL Generation] LLM Response: {sql}")

    # SQL clearing: remove code block markers, comments, and ensure it starts with SELECT or WITH
    cleaned_sql = sql.strip()
    if cleaned_sql.startswith("```"):
        cleaned_sql = cleaned_sql.removeprefix("```sql")
        cleaned_sql = cleaned_sql.removeprefix("```")
        cleaned_sql = cleaned_sql.removesuffix("```").strip()

    lines = cleaned_sql.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        if (
            stripped_line
            and not stripped_line.startswith("--")
            and not stripped_line.startswith("/*")
            and not stripped_line.startswith("#")
        ):
            cleaned_lines.append(stripped_line)

    first_non_comment_line = cleaned_lines[0].lower() if cleaned_lines else ""
    if not (first_non_comment_line.startswith(("select", "with", "explain"))):
        print(f"[SQL Generation] Invalid SQL generated: {sql}")
        raise ValueError(sql)

    print(f"[SQL Generation] Cleaned SQL: {cleaned_sql}")
    return cleaned_sql


# TOOLS
def _execute_schema_tool(
    tool_name: str, tool_args: dict, authorized_tables: list[dict]
) -> str:
    # returns the names of all authorized tables
    if tool_name == "get_all_table_names":
        names = []
        for table in authorized_tables:
            if "name" in table:
                names.append(table["name"])
        return json.dumps(names)

    # returns the schema for the specified table names
    elif tool_name == "get_table_schema":
        table_names = tool_args.get("table_names", [])
        if isinstance(table_names, str):
            table_names = [table_names]
        requested_set = {str(name).lower() for name in table_names}
        matched = [
            t for t in authorized_tables if t.get("name", "").lower() in requested_set
        ]
        return json.dumps({"tables": matched}, default=str)

    # returns the schema for all authorized tables
    elif tool_name == "get_all_tables_schema":
        return json.dumps({"tables": authorized_tables}, default=str)
    else:
        return f"Unknown tool: {tool_name}"


def validate_sql(
    sql: str,
    engine_type: str,
    authorized_cols_by_table: dict,
    valid_tables: set,
    all_physical_cols_by_table: dict,
) -> str:
    try:
        database_service.check_sql_authorized_columns(
            sql_query=sql,
            engine_type=engine_type,
            authorized_cols_by_table=authorized_cols_by_table,
            valid_tables=valid_tables,
            all_physical_cols_by_table=all_physical_cols_by_table,
        )
        return json.dumps({"valid": True, "reason": ""})
    except Exception as exc:
        return json.dumps({"valid": False, "reason": str(exc)})


def execute_sql(sql: str, connection_id: str, db) -> str:
    try:
        # Convert connection_id string to UUID
        conn_uuid = (
            uuid.UUID(connection_id)
            if isinstance(connection_id, str)
            else connection_id
        )

        # Fetch the database connection record
        connection = (
            db.query(ExternalDatabaseConnection)
            .filter(ExternalDatabaseConnection.id == conn_uuid)
            .first()
        )

        # Execute the SQL query and measure execution time
        start_time = time.perf_counter()
        query_results = database_service.run_query_on_connection(
            connection=connection,
            sql_query=sql,
        )
        execution_time_ms = int((time.perf_counter() - start_time) * 1000)
        print(
            f"[SQL Agent] Execution succeeded. Rows: {len(query_results)}, Time: {execution_time_ms}ms"
        )

        # Return success response with results and metadata
        return json.dumps(
            {
                "success": True,
                "rows": query_results,
                "row_count": len(query_results),
                "execution_time_ms": execution_time_ms,
            },
            default=str,
        )

    except Exception as exc:
        print(f"[SQL Agent] Execution failed: {exc}")
        return json.dumps({"success": False, "error": str(exc)})


validate_sql = validate_sql
execute_sql = execute_sql


async def check_semantic_alignment(
    query: str, sql: str, schema_context: str
) -> tuple[bool, float]:
    try:
        client = rag_service._get_async_anthropic_client()
        system_prompt = (
            "you are a SQL semantic validator. Return only a JSON object with two keys: "
            'verdict ("pass" or "fail") and confidence (float 0.0-1.0). No markdown, no explanation.'
        )
        user_prompt = (
            f"User Query: {query}\n\n"
            f"Schema Context:\n{schema_context}\n\n"
            f"Generated SQL:\n{sql}\n\n"
            "Does the generated SQL correctly express the intent of the user query?"
        )
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text_blocks = [
            b.text
            for b in response.content
            if getattr(b, "type", None) == "text" and b.text
        ]
        text = " ".join(text_blocks).strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        data = json.loads(text)
        verdict = str(data.get("verdict", "")).strip().lower()
        confidence = float(data.get("confidence", 0.5))
        passed = verdict == "pass" and confidence >= 0.6
        return (passed, confidence)
    except Exception:
        return (True, 0.5)


def check_filters(query: str, sql: str) -> list[str]:
    # Parse the SQL using sqlglot to normalize it, then lowercase for matching.
    try:
        import sqlglot

        parsed = sqlglot.parse_one(sql)
        sql_str = parsed.sql().lower()
    except Exception:
        return ["SQL could not be parsed; filter correctness check was skipped"]

    try:
        query_lower = query.lower()
        missing_filters = []

        # Define regex patterns for detecting date/time filter expressions in the query.
        date_patterns = [
            (r"\blast\s+\d+\s+days?\b", "date filter"),
            (r"\bthis\s+month\b", "date filter"),
            (r"\bsince\b", "date filter"),
            (r"\bin\s+\d{4}\b", "date filter"),
            (r"\byesterday\b", "date filter"),
            (r"\blast\s+week\b", "date filter"),
        ]

        # Define regex pattern for detecting categorical/status filter words in the query.
        status_pattern = (
            r"\b(active|inactive|pending|completed|failed|approved|rejected)\b",
            "status filter",
        )

        # Define regex patterns for detecting numeric bound expressions in the query.
        numeric_patterns = [
            (r"\bmore\s+than\b", "numeric filter"),
            (r"\bless\s+than\b", "numeric filter"),
            (r"\bat\s+least\b", "numeric filter"),
            (r"\btop\s+\d+\b", "numeric filter"),
            (r"\blimit\s+\d+\b", "numeric filter"),
        ]

        # Define regex pattern for detecting negation expressions in the query.
        # Includes "not" which covers the most common natural language negation.
        negation_pattern = (
            r"\b(not|exclude|except|without)\b",
            "negation filter",
        )

        # Collect all detected filter entities and their categories from the query.
        detected = []

        for pat, cat in date_patterns:
            for m in re.finditer(pat, query_lower):
                detected.append((m.group(0), cat))

        for m in re.finditer(status_pattern[0], query_lower):
            detected.append((m.group(0), status_pattern[1]))

        for pat, cat in numeric_patterns:
            for m in re.finditer(pat, query_lower):
                detected.append((m.group(0), cat))

        for m in re.finditer(negation_pattern[0], query_lower):
            detected.append((m.group(0), negation_pattern[1]))

        # For each detected filter entity, check if a corresponding expression exists in the SQL.
        for entity, category in detected:
            found = False

            # First attempt a direct substring match against the normalized SQL.
            if entity in sql_str:
                found = True

            else:
                if category == "date filter":
                    # Extract any numeric values from the entity (e.g. "30" from "last 30 days").
                    nums = re.findall(r"\d+", entity)

                    # Check for numeric match separately from keyword match to avoid operator precedence issues.
                    numeric_match = bool(nums) and any(num in sql_str for num in nums)

                    # Check for common SQL date-related keywords as a fallback synonym match.
                    keyword_match = any(
                        kw in sql_str
                        for kw in [
                            "date",
                            "day",
                            "days",
                            "month",
                            "year",
                            "week",
                            "interval",
                            "now()",
                            "current_date",
                            "today",
                            "yesterday",
                        ]
                    )

                    # Both conditions are evaluated independently with explicit parentheses to avoid precedence bugs.
                    found = numeric_match or keyword_match

                elif category == "status filter":
                    # Check for the status value with common SQL quoting styles to handle parameterized or quoted values.
                    found = (
                        entity in sql_str
                        or f"'{entity}'" in sql_str
                        or f'"{entity}"' in sql_str
                    )

                elif category == "numeric filter":
                    # Map natural language numeric expressions to their SQL operator equivalents.
                    if "more than" in entity:
                        found = ">" in sql_str or "greater" in sql_str
                    elif "less than" in entity:
                        found = "<" in sql_str or "less" in sql_str
                    elif "at least" in entity:
                        found = ">=" in sql_str or "at least" in sql_str
                    elif "top" in entity or "limit" in entity:
                        # For TOP/LIMIT expressions, check for the keyword or the extracted number.
                        nums = re.findall(r"\d+", entity)
                        found = (
                            "limit" in sql_str
                            or "top" in sql_str
                            or (bool(nums) and any(num in sql_str for num in nums))
                        )

                elif category == "negation filter":
                    # Check for common SQL negation patterns as equivalents to natural language negation words.
                    found = any(
                        kw in sql_str for kw in ["not", "!=", "<>", "except", "without"]
                    )

            # If no match was found for this entity, record it as a missing filter.
            if not found:
                missing_filters.append(
                    f"{category} '{entity}' not found in WHERE clause"
                )

        return missing_filters

    except Exception:
        return []


async def check_optimization(
    sql: str,
    engine_type: str = "postgresql",
    db=None,
    connection_id: str | None = None,
) -> tuple[list[str], list[str]]:
    # Run EXPLAIN against the live database to get the actual query execution plan.
    explain_output = None
    if db and connection_id:
        try:
            # Build the correct EXPLAIN syntax based on the database engine.
            if engine_type.lower() in ("postgresql", "postgres"):
                explain_sql = f"EXPLAIN (FORMAT JSON, ANALYZE false) {sql}"
            elif engine_type.lower() in ("mysql", "mariadb"):
                explain_sql = f"EXPLAIN FORMAT=JSON {sql}"
            elif engine_type.lower() == "sqlite":
                explain_sql = f"EXPLAIN QUERY PLAN {sql}"
            else:
                explain_sql = f"EXPLAIN {sql}"

            # Execute the EXPLAIN query through the existing execution path.
            explain_result_str = execute_sql(
                sql=explain_sql,
                connection_id=connection_id,
                db=db,
            )
            explain_result = json.loads(explain_result_str)

            if explain_result.get("success"):
                explain_output = json.dumps(explain_result.get("rows"), indent=2)
        except Exception as exc:
            # EXPLAIN failure is non-fatal - continue with SQL-only optimization.
            print(f"[SQL Judge Agent] EXPLAIN failed, skipping plan analysis: {exc}")

    # Build the user prompt, conditionally including EXPLAIN output if it was retrieved.
    explain_section = (
        f"\nQuery Execution Plan (EXPLAIN output):\n{explain_output}\n"
        if explain_output
        else "\nNo execution plan available.\n"
    )

    user_prompt = (
        f"Generated SQL:\n{sql}\n"
        f"{explain_section}\n"
        "Analyze the SQL and execution plan above. "
        "Categorize performance issues into 'critical' (issues requiring SQL query rewrite) and 'advisory' (minor suggestions or infrastructure tips). "
        "Return ONLY a JSON object with keys 'optimizable' (boolean), 'critical' (list of strings), and 'advisory' (list of strings):\n"
        "{\n"
        '    "optimizable": true,\n'
        '    "critical": [],\n'
        '    "advisory": []\n'
        "}"
    )

    system_prompt = (
        "You are a SQL optimization expert evaluating SQL queries and EXPLAIN execution plans.\n"
        "Categorize optimization findings into two distinct categories: 'critical' and 'advisory'.\n\n"
        "Critical hints - these trigger SQL regeneration:\n"
        "- Subquery that causes a full table rescan visible in EXPLAIN output\n"
        "- Correlated subquery executing once per row\n"
        "- Missing join condition causing a cartesian product\n"
        "- HAVING filter that could be moved to WHERE to reduce rows before aggregation\n"
        "- Subquery in SELECT or WHERE that can be replaced with a window function or JOIN with measurably better plan\n\n"
        "Important Guidelines for Categorization:\n"
        "- CTE-based queries (WITH clauses) are a valid and standard optimization pattern - do NOT flag them as critical unless EXPLAIN output explicitly shows repeated sequential scans with high cost\n"
        "- Only flag something as critical if the EXPLAIN output confirms it, or if it is a textbook anti-pattern like a correlated subquery executing per row\n"
        "- When no EXPLAIN output is available, be conservative - prefer advisory over critical categorization for anything that is not a clear textbook anti-pattern\n\n"
        "Advisory hints - these are logged only, no regeneration:\n"
        "- JOIN type preference (LEFT vs INNER) when the current results are correct\n"
        "- Index suggestions - these are infrastructure level, not query rewrites\n"
        "- Adding LIMIT when the user did not request it\n"
        "- Minor readability or style improvements\n"
        "- ORDER BY on aggregated columns when it is expected behavior\n\n"
        "Return ONLY a JSON object with keys 'optimizable' (boolean), 'critical' (list of strings), and 'advisory' (list of strings). "
        "No markdown, no preamble, no explanation outside the JSON."
    )

    try:
        client = rag_service._get_async_anthropic_client()

        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract text from the response and strip any markdown formatting.
        raw = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                raw += block.text

        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean)

        # Parse the structured JSON verdict from the LLM response.
        parsed = json.loads(clean)

        if parsed.get("optimizable"):
            critical = parsed.get("critical", [])
            advisory = parsed.get("advisory", [])
            return critical, advisory

        return [], []

    except Exception as exc:
        # On any LLM or parse failure, return empty hints rather than blocking the judge.
        print(f"[SQL Judge Agent] Optimization check failed: {exc}")
        return [], []
