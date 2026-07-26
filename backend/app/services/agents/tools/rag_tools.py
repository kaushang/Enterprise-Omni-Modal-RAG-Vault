# RAG Agent tool definitions and executor functions

import json
import re
import threading
from typing import Any

import pandas as pd
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Eval import (
    default_guarded_getitem,
    default_guarded_getiter,
)
from RestrictedPython.Guards import guarded_iter_unpack_sequence

import app.services.rag_service as rag_service
from app.services import embedding_service

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


EXCEL_AGENT_TOOLS = [
    {
        "name": "get_excel_schemas",
        "description": "Retrieves the schema of all authorized Excel and CSV files available to this user, including filename, column names, and data types. Call this first to understand what data is available.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_sample_values",
        "description": "Returns 5-10 sample rows from a specific Excel or CSV file. Use this when column names alone are ambiguous and you need more context to decide whether a file is relevant to the user query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The document ID of the Excel or CSV file to sample.",
                }
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "identify_relevant_files",
        "description": "Makes an LLM call with all retrieved schemas and the user query to determine which files are capable of answering the query. Returns a list of relevant document IDs. Returns an empty list if no files are relevant.",
        "input_schema": {
            "type": "object",
            "properties": {
                "schemas": {
                    "type": "array",
                    "description": "List of schema objects returned by get_excel_schemas.",
                    "items": {"type": "object"},
                }
            },
            "required": ["schemas"],
        },
    },
    {
        "name": "generate_pandas_code",
        "description": "Generates pandas code to answer the user query against a specific Excel or CSV file. Takes the document ID and returns the generated pandas code as a string. Call execute_pandas_code after this to run the code and get the result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The document ID of the Excel or CSV file to generate pandas code for.",
                }
            },
            "required": ["document_id"],
        },
    },
    {
        "name": "execute_pandas_code",
        "description": "Executes previously generated pandas code against a specific Excel or CSV file and returns the result. If execution fails, automatically attempts to fix and re-execute the code once using the execution error as feedback. Call generate_pandas_code first before calling this.",
        "input_schema": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The document ID of the Excel or CSV file to execute the code against.",
                },
                "pandas_code": {
                    "type": "string",
                    "description": "The pandas code string returned by generate_pandas_code.",
                },
            },
            "required": ["document_id", "pandas_code"],
        },
    },
]


# --- Tool executor functions ---


async def search_documents_tool(
    query: str,
    limit: int,
    collection_name: str,
    role_ids: list[str],
    document_id: str | None = None,
) -> list[dict]:
    print(
        f"[Vector Search Agent] search_documents called with query='{query}', limit={limit}"
    )

    query_vector = embedding_service.embed_text(query)

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


async def rerank_results_tool(query: str, hits: list[dict]) -> list[dict]:
    print(f"[Vector Search Agent] rerank_results called with {len(hits)} hits")

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


async def get_excel_schemas_tool(authorized_docs: list) -> list[dict]:

    # Define all supported tabular file types that can have an Excel schema.
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

    schemas = []

    # Iterate through all documents the user is authorized to access.
    for doc in authorized_docs:
        # Select only tabular files that have an extracted Excel schema.
        if (
            getattr(doc, "file_type", "") or ""
        ).lower() in excel_file_types and getattr(
            doc, "excel_schema", None
        ) is not None:
            # Store the document ID, filename, and schema for later use by the Excel agent.
            schemas.append(
                {
                    "document_id": str(doc.id),
                    "filename": doc.filename,
                    "schema": doc.excel_schema,
                }
            )

    # Log how many Excel schemas were collected.
    print(f"[Excel Agent] get_excel_schemas returned {len(schemas)} schemas.")

    # Return the list of available Excel schemas.
    return schemas


async def get_sample_values_tool(document_id: str, authorized_docs: list):
    print(f"[Excel Agent] get_sample_values called for document_id={document_id}")

    try:
        # Find the requested document from the list of authorized documents.
        doc = next((d for d in authorized_docs if str(d.id) == str(document_id)), None)

        # Return an error if the requested document cannot be found.
        if not doc:
            print("[Excel Agent] get_sample_values returned 0 rows.")
            return {"error": "Could not load sample values"}

        # Resolve the document's absolute file path.
        abs_path = rag_service.get_absolute_path(doc.file_path)

        # Load the Excel/CSV file into a pandas DataFrame.
        df = rag_service.load_dataframe(abs_path, doc.file_type)

        # Extract the first 10 rows as representative sample data.
        samples = df.head(10).to_dict(orient="records")

        print(f"[Excel Agent] get_sample_values returned {len(samples)} rows.")
        return samples

    # Handle any errors while loading the file or extracting samples.
    except Exception as exc:
        logger.warning(
            "[Excel Agent] get_sample_values failed for document_id=%s: %s",
            document_id,
            exc,
        )

        print("[Excel Agent] get_sample_values returned 0 rows.")
        return {"error": "Could not load sample values"}


async def identify_relevant_files_tool(schemas: list[dict], query: str) -> list[str]:

    # Store the total number of available files and collect all document IDs for fallback.
    total = len(schemas)
    all_doc_ids = [s.get("document_id") for s in schemas if s.get("document_id")]

    try:
        client = rag_service._get_async_anthropic_client()

        # Define the system prompt instructing the LLM to identify relevant files.
        system_prompt = (
            "You are a file relevance classifier. Given a user query and a list of Excel/CSV file schemas, determine which files are capable of answering the query.\n"
            "Respond ONLY with a JSON object in this exact format with no other text:\n"
            '{"relevant_document_ids": ["id1", "id2"], "reasoning": "one sentence explanation"}\n'
            "If no files are relevant, return an empty list for relevant_document_ids."
        )

        # Build the user prompt containing the query and all available file schemas.
        user_message = (
            f"User Query: {query}\n"
            f"Available files and schemas:\n{json.dumps(schemas, indent=2)}"
        )

        # Send the prompts to Claude and receive the file selection response.
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        # Extract the text portion of the LLM response.
        text = response.content[0].text.strip()

        # Extract only the JSON object in case the model returns extra text.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

        # Parse the JSON response and extract the relevant document IDs and reasoning.
        parsed = json.loads(text)

        relevant_ids = parsed.get("relevant_document_ids", [])
        reasoning = parsed.get("reasoning", "")

        print(
            f"[Excel Agent] identify_relevant_files: {len(relevant_ids)} of {total} files identified as relevant"
        )
        print(f"[Excel Agent] Relevant files: {reasoning}.")

        return relevant_ids

    # If file identification fails, log the error and fall back to returning all files.
    except Exception as exc:
        logger.warning("[Excel Agent] identify_relevant_files failed: %s", exc)

        print(
            f"[Excel Agent] identify_relevant_files: {len(all_doc_ids)} of {total} files identified as relevant"
        )
        print("[Excel Agent] Relevant files: Fallback to all files due to error.")

        return all_doc_ids


async def generate_pandas_code_tool(
    document_id: str, query: str, excel_docs: list
) -> str | None:

    # Find the requested Excel document from the available documents.
    doc = next((d for d in excel_docs if str(d.id) == str(document_id)), None)

    # Return early if the requested document does not exist.
    if not doc:
        print(
            f"[Excel Agent] generate_pandas_code called for unknown document_id={document_id}."
        )
        return None

    print(f"[Excel Agent] generate_pandas_code called for {doc.filename}")

    try:
        client = rag_service._get_async_anthropic_client()

        # Define the system prompt that instructs the LLM how to generate safe pandas code.
        system_prompt = (
            "You are a pandas code generation specialist. \n"
            "The dataframe is already loaded as the variable `df`.\n"
            "Respond ONLY with a JSON object in this exact format with no other text:\n"
            '{"pandas_code": "the pandas code as a single string", "reasoning": "one sentence on what the code does"}\n\n'
            "Rules:\n"
            "- The dataframe is loaded as `df`, do not load it yourself\n"
            "- Do not import any libraries\n"
            "- Do not use any file I/O operations\n"
            "- ALWAYS check if a filtered dataframe is empty before accessing .iloc[0] or .index[0] to avoid IndexErrors\n"
            "- Assign the final result to a variable called `result`\n"
            "- Return only the code as a plain string, no markdown, no backticks"
        )

        # Build the user prompt containing the Excel metadata and the user's query.
        user_prompt = (
            f"Excel File: {doc.filename}\n"
            f"Schema: {json.dumps(doc.excel_schema)}\n"
            f"User Query: {query}\n"
            "Generate pandas code to answer the query. The dataframe is loaded as `df`. Return only the final result as a variable named `result`."
        )

        # Send the prompts to Claude and receive the generated response.
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract the text portion of the LLM response.
        text = response.content[0].text.strip()

        # Extract only the JSON object in case the model returns extra text.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

        # Parse the JSON response and extract the generated pandas code.
        parsed = json.loads(text)
        code = parsed.get("pandas_code", "").strip()

        # Remove Markdown code fences if the model accidentally included them.
        if code.startswith("```"):
            code = code.split("\n", 1)[-1]
            if code.endswith("```"):
                code = code[: -len("```")].strip()

        # Return None if no valid code was generated.
        if not code:
            return None

        # Log the generated pandas code for debugging.
        print(
            f"[Excel Agent] pandas code generated successfully for {doc.filename}.\n\n",
            code,
        )

        return code

    # Handle any errors during code generation and return None.
    except Exception as exc:
        logger.warning(
            "[Excel Agent] generate_pandas_code failed for %s: %s",
            doc.filename,
            exc,
        )
        return None


def execute_pandas_code_in_sandbox(
    doc, pandas_code: str
) -> tuple[Any | None, str | None]:
    # Strip whitespace and remove markdown code fences if present
    code = pandas_code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[-1]
        if code.endswith("```"):
            code = code[: -len("```")].strip()

    # Guard - return early if code is empty after stripping
    if not code:
        return None, "Empty pandas code provided."

    try:
        # Load the dataframe from disk using the document's file path and type
        abs_path = rag_service.get_absolute_path(doc.file_path)
        df = rag_service.load_dataframe(abs_path, doc.file_type)

        # Compile the pandas code under RestrictedPython to prevent unsafe operations
        compiled = compile_restricted(code, filename="<excel_query>", mode="exec")

        # Build a restricted globals environment - only safe builtins and pandas are allowed
        restricted_globals = dict(safe_globals)
        restricted_globals["_getattr_"] = getattr  # Allow attribute access
        restricted_globals["_getitem_"] = default_guarded_getitem  # Allow item access
        restricted_globals["_getiter_"] = default_guarded_getiter  # Allow iteration
        restricted_globals["_iter_unpack_sequence_"] = (
            guarded_iter_unpack_sequence  # Allow unpacking in for loops
        )
        restricted_globals["_write_"] = lambda x: x  # Allow writing to variables
        restricted_globals["pd"] = pd  # Allow pandas to be used in the code

        # Inject the dataframe as the only local variable available to the sandboxed code
        restricted_locals = {"df": df}
        result_container = {"result": None, "error": None}

        # Execute the compiled code in a separate thread to enforce a 10s timeout
        def _execute():
            try:
                exec(compiled, restricted_globals, restricted_locals)
                result_container["result"] = restricted_locals.get("result")
            except Exception as exc:
                result_container["error"] = str(exc)

        thread = threading.Thread(target=_execute, daemon=True)
        thread.start()
        thread.join(timeout=10)

        # If thread is still alive after timeout, execution took too long
        if thread.is_alive():
            return None, "Execution timed out (10s)."

        # If the thread captured an error during exec, surface it as the error return
        if result_container["error"]:
            return None, result_container["error"]

        # If the result variable was never set inside the code, treat as failure
        res = result_container["result"]
        if res is None:
            return None, "Result variable is None or not set."

        return res, None

    except Exception as exc:
        return None, str(exc)


async def execute_pandas_code_tool(
    document_id: str, pandas_code: str, query: str, excel_docs: list
) -> dict | None:
    # Find the document object matching the given document ID
    doc = next((d for d in excel_docs if str(d.id) == str(document_id)), None)

    # Guard - return early if document ID does not match any authorized Excel doc
    if not doc:
        print(
            f"[Excel Agent] execute_pandas_code called for unknown document_id={document_id}."
        )
        return None

    print(f"[Excel Agent] execute_pandas_code called for {doc.filename}")

    # Run the pandas code inside the restricted sandbox and capture the result or error
    raw_result, error = execute_pandas_code_in_sandbox(doc, pandas_code)

    if raw_result is not None:
        print(f"[Excel Agent] Execution succeeded for {doc.filename}")

        # Format the raw result into a clear answer for the user query
        formatted = f"Question: {query}\nAnswer: {str(raw_result)}"
        return {
            "filename": doc.filename,
            "document_id": str(doc.id),
            "result": str(raw_result),
            "formatted_result": formatted,
        }

    # Execution failed - return None so the agent can retry via generate_pandas_code
    print(f"[Excel Agent] Execution failed for {doc.filename}.")
    return error
