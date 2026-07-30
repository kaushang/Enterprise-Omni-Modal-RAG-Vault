"""
Chat API routes — session management, RAG query, and private document upload.
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from app.core.dependencies import get_current_user
from app.db.session import get_db
from app.models.available_model import AvailableModel
from app.models.document import Document
from app.models.document_access_policy import DocumentAccessPolicy
from app.models.enums import (
    DocumentStatus,
    MessageRole,
    OwnerType,
    Visibility,
)
from app.models.query_citation import QueryCitation
from app.models.query_message import QueryMessage
from app.models.query_session import QuerySession
from app.models.user import User
from app.schemas.chat import (
    CreateSessionResponse,
    ModelResponse,
    QueryRequest,
    SessionDetailResponse,
    SessionResponse,
    TranscriptionResponse,
)
from app.schemas.document import DocumentResponse
from app.services.document_processor import process_document
from app.services.embedding_service import transcribe_audio
from app.services.rag_service import run_rag_pipeline
from app.services.storage_service import save_file

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


SUMMARIZE_DOC_INSTRUCTION = "The user wants a summary, not a detailed answer. Identify the key points, main arguments, and important facts from the provided context. Present them concisely, in your own words, organized in a logical order (e.g. by topic or chronology, whichever fits the source). Omit minor details unless they are essential to understanding the core content. Keep the summary significantly shorter than the source material. If summarizing a document, mention the document's overall purpose or subject in the first sentence before going into specifics."
SUMMARIZE_FOCUSED_INSTRUCTION = "The user wants a concise, summary-style answer rather than an exhaustive one. Answer the question directly in 3-5 sentences or a short bullet list, covering only the most important points. Avoid tangents, background context, or exhaustive detail unless the question explicitly asks for it."
SUMMARIZE_CONV_INSTRUCTION = "Summarize the conversation history so far. Highlight key topics, questions, and answers."
COMPARE_INSTRUCTION = "The user wants a direct comparison between two specific documents. Structure your answer to clearly address both documents side by side, not as two separate summaries. Identify concrete similarities and differences, supported by specific details from each document. If the user provided a specific angle or question to focus the comparison, prioritize that angle; otherwise, compare them across their most salient shared dimensions (e.g. scope, conclusions, figures, recommendations, depending on what the documents actually contain). Clearly attribute each point to the document it came from so the user can tell which document supports which claim."
DETAILED_INSTRUCTION = "The user wants a thorough, in-depth answer, not a brief one. Cover relevant context, explain reasoning or mechanisms where applicable, address nuances or edge cases, and don't omit relevant details from the source material for the sake of brevity. Organize longer answers with clear structure (paragraphs or sections) so the depth doesn't become hard to follow. Still stay grounded strictly in the provided context, do not pad the answer with generic information not supported by the retrieved sources."
TABLE_INSTRUCTION = "The user wants the answer formatted as a table wherever the content has comparable attributes, categories, or structured data (e.g. multiple items with shared properties, numeric data, side-by-side comparisons). Use markdown table syntax. Choose column headers that reflect the actual dimensions in the data. If the answer content genuinely doesn't fit a tabular structure (e.g. a single narrative fact with no comparable rows/columns), briefly explain why and answer in normal prose instead rather than forcing an unnatural table."
BULLETS_INSTRUCTION = "The user wants the answer formatted as bullet points rather than prose paragraphs. Break the answer into clear, concise bullet points, each covering one distinct idea or fact. Use nested sub-bullets only if there's a genuine hierarchy of information. Avoid restating the question as a lead-in sentence, start directly with the bulleted content."
ELI5_INSTRUCTION = "The user wants this explained in very simple terms, as if to someone with no background in the topic. Avoid jargon and technical terminology entirely; where a technical term is unavoidable, immediately explain it in plain words. Use everyday analogies or comparisons where they genuinely help understanding. Keep sentences short. Do not oversimplify to the point of being inaccurate, the goal is accessible language, not losing correctness."

# Simple commands: token -> (flag_name, instruction)
SIMPLE_COMMANDS = {
    "/detailed": DETAILED_INSTRUCTION,
    "/table": TABLE_INSTRUCTION,
    "/bullets": BULLETS_INSTRUCTION,
    "/eli5": ELI5_INSTRUCTION,
}

COMMAND_PATTERN = re.compile(
    r"/compare\s+\[([^\]]+)\]\s+\[([^\]]+)\]"  # structured compare, group 1/2
    r"|/compare"  # unstructured compare
    r"|/summarize"
    r"|/detailed"
    r"|/table"
    r"|/bullets"
    r"|/eli5"
)


def parse_chat_command(
    content: str,
    db: Session,
    user: User,
    attached_doc_id: uuid.UUID | None,
) -> tuple[str, str, str | None, list[uuid.UUID] | None, bool, bool]:
    """
    Parses one or more slash commands out of a user message.

    Returns:
        clean_display_content: raw message, unchanged, shown in chat history as typed.
        clean_retrieval_query: message with all command tokens stripped, used for embedding/search.
        command_instruction: combined instruction text to append to the system prompt, or None.
        compare_document_ids: resolved document IDs for /compare, or None.
        is_compare: True if /compare was found.
        is_summarize: True if /summarize should trigger document/conversation summary mode
                      (only when no question follows it).
    """
    matches = list(COMMAND_PATTERN.finditer(content))
    if not matches:
        return content, content, None, None, False, False

    # Strip all matched command spans out of the message to get the plain question.
    pieces, cursor = [], 0
    for m in matches:
        pieces.append(content[cursor : m.start()])
        cursor = m.end()
    pieces.append(content[cursor:])
    clean_retrieval_query = " ".join("".join(pieces).split())

    has_summarize = (
        "/summarize" in content
    )  # cheap containment check is fine; commands are fixed tokens
    compare_match = next(
        (m for m in matches if m.group(0).startswith("/compare")), None
    )

    instructions: list[str] = []
    compare_document_ids = None
    is_compare = False
    is_summarize = False
    has_question = bool(clean_retrieval_query)

    if compare_match:
        is_compare = True
        instructions.append(COMPARE_INSTRUCTION)
        doc1_name, doc2_name = compare_match.group(1), compare_match.group(2)
        if doc1_name and doc2_name:
            docs = (
                db.query(Document)
                .outerjoin(
                    DocumentAccessPolicy,
                    Document.id == DocumentAccessPolicy.document_id,
                )
                .filter(
                    Document.tenant_id == user.tenant_id,
                    Document.status == DocumentStatus.ready,
                    Document.filename.in_([doc1_name.strip(), doc2_name.strip()]),
                    or_(
                        DocumentAccessPolicy.role_id == user.role_id,
                        Document.uploaded_by == user.id,
                    ),
                )
                .distinct()
                .all()
            )
            compare_document_ids = [d.id for d in docs]
            if not has_question:
                clean_retrieval_query = (
                    f"Compare {doc1_name.strip()} and {doc2_name.strip()}"
                )
        elif not has_question:
            clean_retrieval_query = "Compare documents"

    if has_summarize:
        if has_question:
            # /summarize [question] -> focused-summary style, not full doc/conversation summary
            instructions.append(SUMMARIZE_FOCUSED_INSTRUCTION)
        else:
            is_summarize = True
            if attached_doc_id:
                instructions.append(SUMMARIZE_DOC_INSTRUCTION)
                clean_retrieval_query = "Summarize this document"
            else:
                instructions.append(SUMMARIZE_CONV_INSTRUCTION)
                clean_retrieval_query = "Summarize conversation"

    simple_fallback_text = {
        "/detailed": "Give a detailed answer",
        "/table": "Format as table",
        "/bullets": "Format as bullet points",
        "/eli5": "Explain simply",
    }
    # Preserve the order the user actually typed the commands in, not dict order.
    simple_tokens_found = sorted(
        (token for token in SIMPLE_COMMANDS if token in content),
        key=lambda token: content.index(token),
    )
    for token in simple_tokens_found:
        instructions.append(SIMPLE_COMMANDS[token])
        if not clean_retrieval_query:
            clean_retrieval_query = simple_fallback_text[token]

    command_instruction = "\n\n".join(instructions) if instructions else None
    return (
        content,
        clean_retrieval_query,
        command_instruction,
        compare_document_ids,
        is_compare,
        is_summarize,
    )


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------


@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_session(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new chat session."""
    session = QuerySession(
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
        title="New Chat",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/sessions", response_model=list[SessionResponse])
def list_sessions(
    is_pinned: bool | None = None,
    limit: int | None = None,
    offset: int | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all sessions for the current user, pinned first then newest first."""
    query = db.query(QuerySession).filter(QuerySession.user_id == current_user.id)

    if is_pinned is not None:
        query = query.filter(QuerySession.is_pinned == is_pinned)
        query = query.order_by(QuerySession.updated_at.desc())
    else:
        query = query.order_by(
            QuerySession.is_pinned.desc(), QuerySession.updated_at.desc()
        )

    if limit is not None:
        query = query.limit(limit)

    if offset is not None:
        query = query.offset(offset)

    sessions = query.all()
    return sessions


@router.patch("/sessions/{session_id}/pin", response_model=SessionResponse)
def toggle_pin_session(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Toggle the is_pinned flag for a session owned by the current user."""
    session = (
        db.query(QuerySession)
        .filter(
            QuerySession.id == session_id,
            QuerySession.user_id == current_user.id,
        )
        .first()
    )
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    session.is_pinned = not session.is_pinned
    db.commit()
    db.refresh(session)
    return session


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get a session with all messages and their citations,
    ordered by created_at ascending.
    """
    session = (
        db.query(QuerySession)
        .options(
            joinedload(QuerySession.messages)
            .joinedload(QueryMessage.citations)
            .joinedload(QueryCitation.document),
            joinedload(QuerySession.messages)
            .joinedload(QueryMessage.citations)
            .joinedload(QueryCitation.connection),
            joinedload(QuerySession.messages).joinedload(
                QueryMessage.attached_document
            ),
            joinedload(QuerySession.messages).joinedload(QueryMessage.model),
        )
        .filter(
            QuerySession.id == session_id,
            QuerySession.user_id == current_user.id,
        )
        .first()
    )
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # joinedload does not guarantee ordering — sort messages chronologically
    # before serialisation so the client always receives them oldest-first.
    # Fallback to role if timestamps are identical.
    session.messages.sort(
        key=lambda m: (m.created_at, 0 if m.role == MessageRole.user else 1)
    )

    return session


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a session and all its messages/citations (cascade)."""
    session = (
        db.query(QuerySession)
        .filter(
            QuerySession.id == session_id,
            QuerySession.user_id == current_user.id,
        )
        .first()
    )
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    db.delete(session)
    db.commit()
    return {"message": "Session deleted successfully"}


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------


@router.post("/sessions/{session_id}/query")
async def send_query(
    session_id: uuid.UUID,
    body: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Send a user query to the RAG pipeline and stream the response.

    1. Verify session ownership
    2. Fetch recent conversation history
    3. Store user message and commit (before streaming starts)
    4. Stream response from RAG pipeline
    5. After streaming completes, store assistant message + citations and commit
    """
    # Rate limit check (burst and sustained)
    from app.services.rate_limit_service import check_and_increment_rate_limit
    from app.core.config import settings

    # Check burst limit
    burst_ok = await check_and_increment_rate_limit(
        user_id=str(current_user.id),
        key_suffix="burst",
        max_count=settings.CHAT_BURST_LIMIT,
        window_seconds=settings.CHAT_BURST_WINDOW_SECONDS,
    )
    # Check sustained limit
    sustained_ok = await check_and_increment_rate_limit(
        user_id=str(current_user.id),
        key_suffix="sustained",
        max_count=settings.CHAT_SUSTAINED_LIMIT,
        window_seconds=settings.CHAT_SUSTAINED_WINDOW_SECONDS,
    )

    if not burst_ok or not sustained_ok:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="You're sending messages too quickly. Please wait a moment and try again.",
        )

    # Verify session belongs to this user
    session = (
        db.query(QuerySession)
        .filter(
            QuerySession.id == session_id,
            QuerySession.user_id == current_user.id,
        )
        .first()
    )
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # Server-side database connection lock enforcement
    if session.db_connection_id is not None:
        body.database_id = session.db_connection_id

    content = body.content

    # Parse command prefix and translate it into prompts/instructions
    (
        clean_display_content,
        clean_retrieval_query,
        command_instruction,
        compare_document_ids,
        is_compare,
        is_summarize,
    ) = parse_chat_command(content, db, current_user, body.document_id)

    # Fetch the last 10 messages for conversation history
    recent_messages = (
        db.query(QueryMessage)
        .options(joinedload(QueryMessage.citations).joinedload(QueryCitation.document))
        .filter(QueryMessage.session_id == session_id)
        .order_by(QueryMessage.created_at.desc())
        .limit(10)
        .all()
    )
    # Reverse to chronological order
    recent_messages = list(reversed(recent_messages))

    # Check if this is the first message (no previous messages)
    is_first_message = len(recent_messages) == 0

    # Format conversation history
    conversation_history = ""
    if recent_messages:
        from app.services.database_service import format_query_results_for_prompt
        from app.core.config import settings

        history_lines = []
        for msg in recent_messages:
            role_label = "User" if msg.role == MessageRole.user else "Assistant"
            if msg.role == MessageRole.assistant:
                msg_content = msg.content
                doc_chunks = []
                if msg.citations:
                    for cit in msg.citations:
                        if cit.document:
                            doc_chunks.append(
                                f"Source: {cit.document.filename} | Chunk: {cit.chunk_index}\n{cit.chunk_text}"
                            )
                if doc_chunks:
                    msg_content = (
                        "Retrieved Document Chunks:\n"
                        + "\n---\n".join(doc_chunks)
                        + f"\n\nAnswer: {msg_content}"
                    )
                if getattr(msg, "generated_sql", None):
                    formatted_results = format_query_results_for_prompt(
                        msg.query_results, settings.SQL_RESULT_SUMMARY_THRESHOLD
                    )
                    msg_content = (
                        f"Generated SQL:\n{msg.generated_sql}\n"
                        f"SQL Results:\n{formatted_results}\n"
                        f"{msg_content}"
                    )
                history_lines.append(f"{role_label}: {msg_content}")
            else:
                history_lines.append(f"{role_label}: {msg.content}")
        conversation_history = "\n".join(history_lines)

    # Store the user's message using clean display content
    user_message = QueryMessage(
        session_id=session_id,
        role=MessageRole.user,
        content=clean_display_content,
        created_at=datetime.now(timezone.utc),
        attached_document_id=body.document_id,
    )
    db.add(user_message)

    # If first message, update session title using clean display content
    if is_first_message:
        session.title = clean_display_content[:50]

    # Commit before streaming starts so the user message is saved immediately
    db.commit()

    # Resolve the model to use
    resolved_model_id = None
    if body.model_id:
        if body.model_id == "auto":
            resolved_model_id = "auto"
        else:
            from app.models.available_model import AvailableModel

            try:
                model_exists = (
                    db.query(AvailableModel)
                    .filter(
                        AvailableModel.id == body.model_id, AvailableModel.is_active
                    )
                    .first()
                )
                if model_exists:
                    resolved_model_id = body.model_id
            except Exception:
                pass

    if not resolved_model_id:
        from app.models.available_model import AvailableModel

        if current_user.tenant and current_user.tenant.default_model_id:
            tenant_default = (
                db.query(AvailableModel)
                .filter(
                    AvailableModel.id == current_user.tenant.default_model_id,
                    AvailableModel.is_active,
                )
                .first()
            )
            if tenant_default:
                resolved_model_id = tenant_default.id

    if not resolved_model_id:
        from app.models.available_model import AvailableModel

        default_model = (
            db.query(AvailableModel)
            .filter(
                AvailableModel.is_active,
                AvailableModel.provider_id == "anthropic",
            )
            .order_by(AvailableModel.created_at.asc())
            .first()
        )
        if not default_model:
            default_model = (
                db.query(AvailableModel)
                .filter(AvailableModel.is_active)
                .order_by(AvailableModel.created_at.asc())
                .first()
            )
        if default_model:
            resolved_model_id = default_model.id

    async def event_generator():
        try:
            generator = run_rag_pipeline(
                query=clean_retrieval_query,
                user=current_user,
                db=db,
                conversation_history=conversation_history,
                document_id=body.document_id,
                database_id=body.database_id,
                command_instruction=command_instruction,
                compare_document_ids=compare_document_ids,
                is_compare_mode=is_compare,
                is_summarize_mode=is_summarize,
                model_id=resolved_model_id,
                session_id=session_id,
            )

            full_answer = ""
            citations = []
            actual_model_string = None
            follow_up_questions = []
            generated_sql = None
            query_results = None
            db_connection_id = None
            execution_time_ms = 0
            status_str = None
            error_message = None
            error_type = None
            resolved_model = None
            actual_resolved_model_id = None
            was_fallback = False
            fallback_model_name = None

            async for event in generator:
                if event["type"] == "token":
                    # Stream format: data: {"type": "token", "content": "..."}\n\n
                    yield f"data: {json.dumps({'type': 'token', 'content': event['content']})}\n\n"
                elif event["type"] == "done":
                    full_answer = event["answer"]
                    citations = event["citations"]
                    actual_model_string = event.get("model_string")
                    follow_up_questions = event.get("follow_up_questions", [])
                    generated_sql = event.get("generated_sql")
                    query_results = event.get("query_results")
                    db_connection_id = event.get("db_connection_id")
                    execution_time_ms = event.get("execution_time_ms", 0)
                    status_str = event.get("status")
                    error_message = event.get("error_message")
                    error_type = event.get("error_type")
                    resolved_model = event.get("resolved_model")
                    actual_resolved_model_id = event.get("resolved_model_id")
                    was_fallback = event.get("was_fallback", False)
                    fallback_model_name = event.get("fallback_model_name")

            from app.core.utils import extract_chart_spec

            cleaned_answer, chart_spec = extract_chart_spec(full_answer)

            # Lock database connection to session if a DB query ran successfully for the first time
            if (
                body.database_id is not None
                and session.db_connection_id is None
                and generated_sql is not None
            ):
                session.db_connection_id = body.database_id

            # Store the assistant's response in the database after streaming completes
            assistant_message = QueryMessage(
                session_id=session_id,
                role=MessageRole.assistant,
                content=cleaned_answer,
                created_at=datetime.now(timezone.utc),
                model_id=actual_resolved_model_id
                if actual_resolved_model_id
                else (resolved_model_id if resolved_model_id != "auto" else None),
                follow_up_questions=follow_up_questions,
                generated_sql=generated_sql,
                query_results=query_results
                if isinstance(query_results, list)
                else None,
                chart_spec=chart_spec,
                resolved_model=resolved_model if body.model_id == "auto" else None,
                was_fallback=was_fallback,
                fallback_model_name=fallback_model_name,
            )
            db.add(assistant_message)
            db.flush()

            # Store citations linked to the assistant message
            citation_models = []
            for cit in citations:
                doc_id_val = None
                conn_id_val = None
                try:
                    target_uuid = uuid.UUID(str(cit["document_id"]))
                    doc_exists = (
                        db.query(Document).filter(Document.id == target_uuid).first()
                        is not None
                    )
                    if doc_exists:
                        doc_id_val = target_uuid
                    else:
                        conn_id_val = target_uuid
                except Exception:
                    pass

                citation = QueryCitation(
                    message_id=assistant_message.id,
                    document_id=doc_id_val,
                    connection_id=conn_id_val,
                    qdrant_vector_id=str(cit.get("chunk_index", "")),
                    chunk_text=cit["chunk_text"],
                    page_number=cit.get("page_number"),
                    chunk_index=cit.get("chunk_index", 0),
                )
                db.add(citation)
                citation_models.append(citation)

            # Create QueryLog entry for evaluations
            from app.models.query_log import QueryLog

            query_log = QueryLog(
                tenant_id=session.tenant_id,
                user_id=current_user.id,
                question=clean_display_content,
                contexts=[cit["chunk_text"] for cit in citations],
                answer=cleaned_answer,
                model_string=actual_model_string,
                created_at=datetime.now(timezone.utc),
            )
            db.add(query_log)

            # Create DBQueryLog entry for DB Health & Analytics if database was queried
            if db_connection_id is not None:
                from app.models.db_query_log import DBQueryLog

                db_query_log = DBQueryLog(
                    tenant_id=session.tenant_id,
                    user_id=current_user.id,
                    db_connection_id=db_connection_id,
                    natural_language_query=clean_display_content,
                    generated_sql=generated_sql,
                    execution_time_ms=execution_time_ms,
                    status=status_str or ("success" if not error_message else "failed"),
                    error_message=error_message,
                    error_type=error_type,
                    created_at=datetime.now(timezone.utc),
                )
                db.add(db_query_log)

            # Update session timestamp
            session.updated_at = datetime.now(timezone.utc)
            db.commit()

            # Refresh to get generated IDs
            db.refresh(assistant_message)
            for c in citation_models:
                db.refresh(c)

            # Build citation responses for the done event
            citation_responses = []
            for cit_model, cit_data in zip(citation_models, citations):
                citation_responses.append(
                    {
                        "id": str(cit_model.id),
                        "document_id": str(cit_model.document_id)
                        if cit_model.document_id
                        else None,
                        "connection_id": str(cit_model.connection_id)
                        if cit_model.connection_id
                        else None,
                        "filename": cit_data["filename"],
                        "chunk_text": cit_data["chunk_text"],
                        "page_number": cit_data.get("page_number"),
                        "chunk_index": cit_data.get("chunk_index", 0),
                    }
                )

            # Final event: data: {"type": "done", "citations": [...], "message_id": "...", "follow_up_questions": [...], "generated_sql": "...", "answer": "...", "chart_spec": ...}\n\n
            yield f"data: {json.dumps({'type': 'done', 'citations': citation_responses, 'message_id': str(assistant_message.id), 'follow_up_questions': follow_up_questions, 'generated_sql': generated_sql, 'answer': cleaned_answer, 'chart_spec': chart_spec, 'resolved_model': assistant_message.resolved_model, 'was_fallback': was_fallback, 'fallback_model_name': fallback_model_name})}\n\n"

        except Exception as exc:
            logger.error("Error in event_generator: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'content': 'An error occurred during streaming.'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Private document upload (members only)
# ---------------------------------------------------------------------------


@router.post(
    "/sessions/{session_id}/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_private_document(
    session_id: uuid.UUID,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Upload a private document from the chat interface.
    Only non-admin (member) users can use this route.
    """
    # Only members can upload private documents
    if current_user.role.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admins cannot upload private documents. Use the Documents page instead.",
        )

    # Verify session belongs to this user
    session = (
        db.query(QuerySession)
        .filter(
            QuerySession.id == session_id,
            QuerySession.user_id == current_user.id,
        )
        .first()
    )
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # Validate file using extension & MIME type (python-magic)
    filename = file.filename or ""
    from app.services.document_processor import validate_upload_file

    file_type = validate_upload_file(file)

    # Generate document ID and save file
    document_id = uuid.uuid4()
    file_path = save_file(file, str(current_user.tenant_id), str(document_id))

    tenant_id = current_user.tenant_id
    collection_name = f"tenant_{tenant_id}"

    # Create document record — private, no access policies
    document = Document(
        id=document_id,
        tenant_id=tenant_id,
        uploaded_by=current_user.id,
        filename=filename,
        file_type=file_type,
        owner_type=OwnerType.private,
        visibility=Visibility.private,
        chunk_count=0,
        qdrant_collection=collection_name,
        status=DocumentStatus.pending,
        file_path=file_path,
        file_size=file.size,
    )
    db.add(document)
    db.commit()

    # Process document synchronously
    process_document(str(document_id), db)

    db.refresh(document)
    return document


@router.post("/transcribe", response_model=TranscriptionResponse)
def transcribe_voice_query(
    audio: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Accepts a short audio recording, transcribes it using Gemini 2.5 Flash,
    and returns the transcribed text.
    """
    content_type = audio.content_type or ""
    if "webm" in content_type:
        ext = "webm"
    elif "wav" in content_type:
        ext = "wav"
    elif "mpeg" in content_type or "mp3" in content_type:
        ext = "mp3"
    elif "mp4" in content_type or "m4a" in content_type:
        ext = "m4a"
    else:
        # Fallback to filename extension
        filename = audio.filename or ""
        if "." in filename:
            ext = filename.rsplit(".", 1)[-1].lower()
        else:
            ext = "webm"

    tmp_dir = "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    temp_file_path = os.path.join(tmp_dir, f"{uuid.uuid4()}_voice_query.{ext}")

    try:
        with open(temp_file_path, "wb") as f:
            f.write(audio.file.read())

        transcribed_text = transcribe_audio(temp_file_path)
        return TranscriptionResponse(text=transcribed_text)
    except Exception as exc:
        logger.error(f"Failed to transcribe audio query: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not transcribe audio. Please try again.",
        )
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as delete_exc:
                logger.warning(
                    f"Failed to delete temp file {temp_file_path}: {delete_exc}"
                )


@router.get("/models", response_model=list[ModelResponse])
def get_active_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get all active models for the user model selector.
    Each model has `is_tenant_default=True` if it is the tenant's
    admin-configured default chat model (Tenant.default_model_id).
    """
    models = (
        db.query(AvailableModel)
        .filter(
            AvailableModel.is_active,
            (AvailableModel.tenant_id == current_user.tenant_id)
            | (AvailableModel.tenant_id.is_(None)),
        )
        .order_by(AvailableModel.created_at.asc())
        .all()
    )

    tenant_default_model_id = (
        current_user.tenant.default_model_id if current_user.tenant else None
    )

    result = []
    for model in models:
        model_dict = {
            "id": model.id,
            "display_name": model.display_name,
            "is_active": model.is_active,
            "created_at": model.created_at,
            "provider_id": model.provider_id,
            "base_url": model.base_url,
            "input_cost_per_million_tokens": model.input_cost_per_million_tokens,
            "output_cost_per_million_tokens": model.output_cost_per_million_tokens,
            "tenant_id": model.tenant_id,
            "api_key": model.api_key,
            "is_default": model.is_default,
            "model_name": model.model_name,
            "is_tenant_default": (
                tenant_default_model_id is not None
                and model.id == tenant_default_model_id
            ),
            "tier": model.tier,
        }
        result.append(ModelResponse(**model_dict))

    return result
