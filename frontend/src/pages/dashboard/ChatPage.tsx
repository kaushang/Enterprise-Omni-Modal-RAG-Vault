import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from "react";
import { useSearchParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  SendHorizontal,
  Plus,
  FileText,
  ChevronDown,
  ChevronUp,
  X,
  Loader2,
  File,
  FilePen,
  Presentation,
  FileSpreadsheet,
  FileMusic,
  Search,
  Mic,
  Square,
  Database,
  Lock,
  Table,
} from "lucide-react";
import { UserSchemaModal } from "../../components/UserSchemaModal";
import { useAuthStore } from "../../store/authStore";
import { chatService } from "../../services/chatService";
import { documentService } from "../../services/documentService";
import { databaseService } from "../../services/databaseService";
import type { MessageResponse, AvailableModel } from "../../types/chat";
import type { DocumentResponse, FileType } from "../../types/document";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChartRenderer } from "../../components/dashboard/ChartRenderer";
import { ReportGenerationPanel } from "../../components/dashboard/ReportGenerationPanel";

interface UploadedFile {
  file: File;
  status: "uploading" | "ready";
  error?: string;
  id?: string;
}

const FILE_TYPE_ICON: Record<FileType, React.FC<{ className?: string }>> = {
  text: FileText,
  pdf: File,
  docx: FilePen,
  pptx: Presentation,
  excel: FileSpreadsheet,
  csv: FileSpreadsheet,
  audio: FileMusic,
  xls: FileSpreadsheet,
  xlsm: FileSpreadsheet,
  xlsb: FileSpreadsheet,
  tsv: FileSpreadsheet,
  ods: FileSpreadsheet,
};

const COMMANDS = [
  { name: "/summarize", description: "Summarize document(s) or conversation" },
  { name: "/compare", description: "Compare two documents" },
  { name: "/detailed", description: "Give a thorough, detailed answer" },
  { name: "/table", description: "Format the answer as a table" },
  { name: "/pin", description: "Pin the current chat session" },
  { name: "/new", description: "Start a new conversation" },
  { name: "/bullets", description: "Format the answer as bullet points" },
  { name: "/eli5", description: "Explain like I'm 5. Explain in simple terms" },
];

const autoModel: AvailableModel = {
  id: "auto",
  display_name: "Auto",
  is_active: true,
  created_at: "",
  tier: "balanced",
};

const markdownComponents = {
  table: ({ ...props }: any) => (
    <div className="my-4 overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm custom-scrollbar">
      <table
        className="min-w-full divide-y divide-slate-200 dark:divide-slate-800 text-left border-collapse"
        {...props}
      />
    </div>
  ),
  thead: ({ ...props }: any) => (
    <thead className="bg-slate-50 dark:bg-slate-800/50" {...props} />
  ),
  tbody: ({ ...props }: any) => (
    <tbody
      className="divide-y divide-slate-100 dark:divide-slate-850 bg-white dark:bg-slate-900"
      {...props}
    />
  ),
  tr: ({ ...props }: any) => (
    <tr
      className="hover:bg-slate-50/50 dark:hover:bg-slate-800/30 transition-colors duration-150"
      {...props}
    />
  ),
  th: ({ ...props }: any) => (
    <th
      className="px-4 py-2.5 text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider border-b border-slate-200 dark:border-slate-850"
      {...props}
    />
  ),
  td: ({ ...props }: any) => (
    <td
      className="px-4 py-2.5 text-sm text-slate-700 dark:text-slate-200 whitespace-nowrap"
      {...props}
    />
  ),
  p: ({ ...props }: any) => (
    <p className="mb-2 last:mb-0 leading-relaxed" {...props} />
  ),
  ul: ({ ...props }: any) => (
    <ul className="list-disc pl-5 mb-2 space-y-1" {...props} />
  ),
  ol: ({ ...props }: any) => (
    <ol className="list-decimal pl-5 mb-2 space-y-1" {...props} />
  ),
  li: ({ ...props }: any) => <li className="text-sm" {...props} />,
};

function parseDbQueryMessage(
  msg: MessageResponse,
  isLatestAssistant: boolean,
  hasAttachedDb: boolean,
) {
  if (msg.generated_sql) {
    return {
      isDbQuery: true,
      status: "completed" as const,
      sql: msg.generated_sql,
      answer: msg.content.split("[FOLLOW_UP]")[0],
    };
  }

  const content = msg.content;
  const hasSqlMarker =
    content.includes("**Generated SQL Query:**") ||
    content.includes("**Regenerated SQL Query:**");
  const hasExecutingMarker =
    content.includes("*Executing query...*") ||
    content.includes("*Executing corrected query...*");
  const hasThinkingMarker = content.includes(
    "*Thinking... Translating your request to SQL...*",
  );

  if (hasSqlMarker || hasExecutingMarker) {
    let sql = "";
    const sqlMatch = content.match(
      /\*\*(?:Generated|Regenerated) SQL Query:\*\*\s*\n```sql\n([\s\S]*?)\n```/,
    );
    if (sqlMatch) {
      sql = sqlMatch[1];
    }

    let status: "executing_query" | "completed" = "executing_query";
    let answer = "";

    if (hasExecutingMarker) {
      const parts = content.split(
        /\*(?:Executing query|Executing corrected query)\.\.\.\*\s*/,
      );
      if (parts.length > 1) {
        const after = parts[1];
        if (after.trim().length > 0) {
          status = "completed";
          answer = after;
        }
      }
    }

    return {
      isDbQuery: true,
      status,
      sql,
      answer,
    };
  }

  if (hasThinkingMarker) {
    return {
      isDbQuery: true,
      status: "generating_sql" as const,
      sql: null,
      answer: "",
    };
  }

  if (content === "" && isLatestAssistant && hasAttachedDb) {
    return {
      isDbQuery: true,
      status: "generating_sql" as const,
      sql: null,
      answer: "",
    };
  }

  return {
    isDbQuery: false,
    status: null,
    sql: null,
    answer: content.split("[FOLLOW_UP]")[0],
  };
}

const ChatPage: React.FC = () => {
  const { user } = useAuthStore();
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();

  const activeSessionIdRef = useRef<string | null>(null);

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<MessageResponse[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const isSendingRef = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedCitations, setExpandedCitations] = useState<Set<string>>(
    new Set(),
  );
  const [expandedSqls, setExpandedSqls] = useState<Set<string>>(new Set());
  const [toast, setToast] = useState<{
    message: string;
    type: "success" | "error";
  } | null>(null);
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
  const [attachedDocument, setAttachedDocument] =
    useState<DocumentResponse | null>(null);

  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [audioLevels, setAudioLevels] = useState<number[]>([
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
  ]);
  const [recordingSeconds, setRecordingSeconds] = useState(0);

  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<AvailableModel | null>(
    null,
  );
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  const modelDropdownRef = useRef<HTMLDivElement>(null);

  const [attachedDatabase, setAttachedDatabase] = useState<any | null>(null);
  const [lockedDbConnectionId, setLockedDbConnectionId] = useState<
    string | null
  >(null);
  const [isDbDropdownOpen, setIsDbDropdownOpen] = useState(false);
  const dbDropdownRef = useRef<HTMLDivElement>(null);
  const [isSchemaModalOpen, setIsSchemaModalOpen] = useState(false);

  // Load available models
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const activeModels = await chatService.getModels();
        setAvailableModels(activeModels);

        // localStorage key is scoped per-user so different users on the same
        // browser don't inherit each other's model choice.
        const storageKey = user?.id
          ? `selected_model_id_${user.id}`
          : "selected_model_id";
        const savedModelId = localStorage.getItem(storageKey);

        if (savedModelId === "auto") {
          setSelectedModel(autoModel);
        } else {
          const found = savedModelId
            ? activeModels.find((m) => m.id === savedModelId)
            : null;
          if (found) {
            setSelectedModel(found);
          } else {
            // Priority:
            // 1. The admin-configured tenant default (is_tenant_default)
            // 2. Any model flagged is_default
            // 3. First available model
            const defaultModel =
              activeModels.find((m) => m.is_tenant_default) ??
              activeModels.find((m) => m.is_default) ??
              activeModels[0];

            if (defaultModel) {
              setSelectedModel(defaultModel);
            }
          }
        }
      } catch (err) {
        console.error("Failed to fetch models", err);
      }
    };
    fetchModels();
  }, [user?.id]);

  // Click outside to close model dropdown
  useEffect(() => {
    const handleClickOutsideModel = (e: MouseEvent) => {
      if (
        modelDropdownRef.current &&
        !modelDropdownRef.current.contains(e.target as Node)
      ) {
        setIsModelDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutsideModel);
    return () =>
      document.removeEventListener("mousedown", handleClickOutsideModel);
  }, []);

  // Click outside to close database dropdown
  useEffect(() => {
    const handleClickOutsideDb = (e: MouseEvent) => {
      if (
        dbDropdownRef.current &&
        !dbDropdownRef.current.contains(e.target as Node)
      ) {
        setIsDbDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutsideDb);
    return () =>
      document.removeEventListener("mousedown", handleClickOutsideDb);
  }, []);

  const handleSelectModel = (model: AvailableModel) => {
    setSelectedModel(model);
    const storageKey = user?.id
      ? `selected_model_id_${user.id}`
      : "selected_model_id";
    localStorage.setItem(storageKey, model.id);
    setIsModelDropdownOpen(false);
  };

  const handleDetach = (filenameToRemove?: string) => {
    if (textareaRef.current) {
      const selector = filenameToRemove
        ? `[data-filename="${CSS.escape(filenameToRemove)}"]`
        : "[data-filename]";
      const pills = textareaRef.current.querySelectorAll(selector);
      pills.forEach((pill) => {
        let next = pill.nextSibling;
        if (
          next &&
          next.nodeType === Node.TEXT_NODE &&
          next.textContent === " "
        ) {
          next.remove();
        }
        pill.remove();
      });
      setInputValue(textareaRef.current.innerHTML);
    }

    setAttachedDocument(null);
    setUploadedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    setTimeout(() => {
      textareaRef.current?.focus();
    }, 50);
  };

  const renderMessageContentWithHighlights = (msg: MessageResponse) => {
    const text = msg.content;
    const fileName = msg.attached_file?.name;

    if (!fileName || !text.includes(fileName)) {
      return text;
    }

    const index = text.indexOf(fileName);
    const before = text.substring(0, index);
    const filenameText = text.substring(index, index + fileName.length);
    const after = text.substring(index + fileName.length);

    return (
      <>
        {before}
        <span className="inline-flex items-center gap-1 mb-1 bg-slate-100 px-2.5 py-0.5 rounded-full shadow-sm text-sm font-semibold text-slate-800 dark:text-slate-900 select-none align-middle mx-1 transition-colors">
          <FileText className="w-3 h-3 text-indigo-600 flex-shrink-0" />
          <span className="truncate max-w-[180px]">{filenameText}</span>
        </span>
        {after}
      </>
    );
  };

  // Dropdown UI states
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [dropdownSearch, setDropdownSearch] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);

  // Compare picker states
  const [isComparePickerOpen, setIsComparePickerOpen] = useState(false);
  const [compareSelection, setCompareSelection] = useState<DocumentResponse[]>(
    [],
  );

  const dropdownRef = useRef<HTMLDivElement>(null);
  const dropdownSearchInputRef = useRef<HTMLInputElement>(null);

  // Fetch authorized documents
  const { data: authorizedDocs, isLoading: isLoadingAuth } = useQuery({
    queryKey: ["authorized-documents"],
    queryFn: () => documentService.getAuthorizedDocuments(),
  });

  // Fetch personal documents
  const { data: personalDocs, isLoading: isLoadingPersonal } = useQuery({
    queryKey: ["personal-documents"],
    queryFn: () => documentService.getPersonalDocuments(),
  });

  // Fetch authorized databases
  const { data: userDatabases = [], isSuccess: isUserDatabasesLoaded } =
    useQuery({
      queryKey: ["authorized-databases"],
      queryFn: () => databaseService.getAuthorizedDatabases(),
    });

  // Combined ready documents list
  const allDocs = useMemo(() => {
    const authList = authorizedDocs ?? [];
    const personalList = personalDocs ?? [];
    const docMap = new Map<string, DocumentResponse>();

    authList.forEach((doc) => {
      if (doc.status === "ready") docMap.set(doc.id, doc);
    });
    personalList.forEach((doc) => {
      if (doc.status === "ready") docMap.set(doc.id, doc);
    });

    return Array.from(docMap.values());
  }, [authorizedDocs, personalDocs]);

  // Filtered commands for search query
  const filteredCommands = useMemo(() => {
    if (isComparePickerOpen) return [];
    const query = dropdownSearch.trim().toLowerCase();
    if (!query) return COMMANDS;
    return COMMANDS.filter(
      (cmd) =>
        cmd.name.toLowerCase().includes(query) ||
        cmd.description.toLowerCase().includes(query),
    );
  }, [dropdownSearch, isComparePickerOpen]);

  // Filtered documents for search query
  const filteredDocs = useMemo(() => {
    const query = dropdownSearch.trim().toLowerCase();
    if (!query) return allDocs;
    return allDocs.filter((doc) => doc.filename.toLowerCase().includes(query));
  }, [allDocs, dropdownSearch]);

  const totalDropdownItemsCount = isComparePickerOpen
    ? filteredDocs.length
    : filteredCommands.length + filteredDocs.length;

  // Reset activeIndex when filtered items change
  useEffect(() => {
    setActiveIndex(0);
  }, [filteredDocs, filteredCommands, isComparePickerOpen]);

  // Focus search box when dropdown opens
  useEffect(() => {
    if (isDropdownOpen) {
      setTimeout(() => {
        dropdownSearchInputRef.current?.focus();
      }, 50);
    }
  }, [isDropdownOpen]);

  // Click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node)
      ) {
        setIsDropdownOpen(false);
        setIsComparePickerOpen(false);
        setCompareSelection([]);
      }
    };
    if (isDropdownOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isDropdownOpen]);

  const lastSelectionRangeRef = useRef<Range | null>(null);

  const saveSelectionRange = () => {
    const selection = window.getSelection();
    if (
      selection &&
      selection.rangeCount > 0 &&
      textareaRef.current &&
      textareaRef.current.contains(selection.anchorNode)
    ) {
      lastSelectionRangeRef.current = selection.getRangeAt(0).cloneRange();
    }
  };

  // Resolve target range for insertions, ensuring the trigger slash is deleted.
  const resolveInsertRange = (): Range | null => {
    let selection = window.getSelection();
    let range: Range | null = null;

    if (
      lastSelectionRangeRef.current &&
      textareaRef.current &&
      textareaRef.current.contains(
        lastSelectionRangeRef.current.commonAncestorContainer,
      )
    ) {
      range = lastSelectionRangeRef.current.cloneRange();
    } else if (
      selection &&
      selection.rangeCount > 0 &&
      textareaRef.current &&
      textareaRef.current.contains(selection.anchorNode)
    ) {
      range = selection.getRangeAt(0).cloneRange();
    } else if (textareaRef.current) {
      textareaRef.current.focus();
      range = document.createRange();
      range.selectNodeContents(textareaRef.current);
      range.collapse(false);
      if (selection) {
        selection.removeAllRanges();
        selection.addRange(range);
      }
    }

    if (range) {
      // Remove trigger slash and search text up to cursor
      const node = range.endContainer;
      const offset = range.endOffset;
      if (node.nodeType === Node.TEXT_NODE) {
        const text = node.textContent || "";
        const slashIndex = text.lastIndexOf("/", offset - 1);
        if (slashIndex !== -1) {
          range.setStart(node, slashIndex);
          range.deleteContents();
        }
      } else if (node.nodeType === Node.ELEMENT_NODE && textareaRef.current) {
        // Fallback: search all text nodes from right to left inside the element
        const walker = document.createTreeWalker(
          textareaRef.current,
          NodeFilter.SHOW_TEXT,
        );
        let textNode: Text | null = null;
        let lastTextNode: Text | null = null;
        while ((textNode = walker.nextNode() as Text)) {
          lastTextNode = textNode;
        }
        if (lastTextNode) {
          const text = lastTextNode.textContent || "";
          const slashIndex = text.lastIndexOf("/");
          if (slashIndex !== -1) {
            range.setStart(lastTextNode, slashIndex);
            range.setEnd(lastTextNode, text.length);
            range.deleteContents();
          }
        }
      }
    }

    return range;
  };

  // Helper to insert a pill (non-editable token) at the current cursor in contenteditable
  const insertPillAtCursor = (filename: string) => {
    const range = resolveInsertRange();
    if (!range) return;

    // Create pill element matching ChatGPT/Claude aesthetics
    const pill = document.createElement("span");
    pill.contentEditable = "false";
    pill.setAttribute("data-filename", filename);
    pill.className =
      "inline-flex items-center gap-0.5 mb-1 bg-indigo-700 dark:bg-slate-800/80 px-2.5 py-0.5 rounded-xl shadow-sm text-sm font-semibold text-slate-100 dark:text-slate-200 select-none align-middle mx-1 pointer-events-auto transition-colors cursor-default";

    // Lucide FileText equivalent SVG
    pill.innerHTML = `
        <span class="truncate max-w-[180px]">
      ${filename}
    </span>
      <button type="button" class="ml-1 p-0.5 dark:hover:bg-slate-700 rounded-full flex items-center justify-center text-slate-200 hover:bg-indigo-500/50 dark:text-slate-500 dark:hover:text-slate-300 transition-colors" title="Remove attachment">
        <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    `;

    // Set up detach click handler
    const closeBtn = pill.querySelector("button");
    if (closeBtn) {
      closeBtn.onclick = (e) => {
        e.stopPropagation();
        handleDetach(filename);
      };
    }

    range.insertNode(pill);

    // Append space after pill
    const spaceNode = document.createTextNode(" ");
    pill.after(spaceNode);

    // Move cursor right after the space
    range.setStartAfter(spaceNode);
    range.setEndAfter(spaceNode);
    const selection = window.getSelection();
    if (selection) {
      selection.removeAllRanges();
      selection.addRange(range);
    }

    if (textareaRef.current) {
      setInputValue(textareaRef.current.innerHTML);
    }
  };

  // Select document and remove the trigger slash, inserting document filename instead of slash
  const handleSelectDocument = (doc: DocumentResponse) => {
    setAttachedDocument(doc);
    setUploadedFile(null); // clear any uploaded file to avoid duplicate attachments
    setIsDropdownOpen(false);
    setDropdownSearch("");

    if (textareaRef.current) {
      insertPillAtCursor(doc.filename);
    }
  };

  // Prepend prefix command, remove trigger slash, place cursor after prefix
  const insertCommandPrefix = (cmdName: string) => {
    setIsDropdownOpen(false);
    setDropdownSearch("");

    const prefix = `${cmdName} `;
    const range = resolveInsertRange();
    if (range) {
      const textNode = document.createTextNode(prefix);
      range.insertNode(textNode);
      range.setStartAfter(textNode);
      range.setEndAfter(textNode);
      const selection = window.getSelection();
      if (selection) {
        selection.removeAllRanges();
        selection.addRange(range);
      }

      if (textareaRef.current) {
        setInputValue(textareaRef.current.innerHTML);
      }
    }
  };

  // Prepend compare command, insert filenames, place cursor after command
  const insertCompareCommand = (doc1Name: string, doc2Name: string) => {
    setIsDropdownOpen(false);
    setIsComparePickerOpen(false);
    setCompareSelection([]);
    setDropdownSearch("");

    const cmdText = `/compare [${doc1Name}] [${doc2Name}] `;
    const range = resolveInsertRange();
    if (range) {
      const textNode = document.createTextNode(cmdText);
      range.insertNode(textNode);
      range.setStartAfter(textNode);
      range.setEndAfter(textNode);
      const selection = window.getSelection();
      if (selection) {
        selection.removeAllRanges();
        selection.addRange(range);
      }

      if (textareaRef.current) {
        setInputValue(textareaRef.current.innerHTML);
      }
    }
  };

  // Toggle document selection in comparison picker
  const handleToggleCompareDocument = (doc: DocumentResponse) => {
    setCompareSelection((prev) => {
      const exists = prev.some((d) => d.id === doc.id);
      let newSelection;
      if (exists) {
        newSelection = prev.filter((d) => d.id !== doc.id);
      } else {
        if (prev.length >= 2) return prev;
        newSelection = [...prev, doc];
      }

      if (newSelection.length === 2) {
        const [doc1, doc2] = newSelection;
        insertCompareCommand(doc1.filename, doc2.filename);
      }
      return newSelection;
    });
  };

  // Handle clear session action
  const handleClearSession = () => {
    setSearchParams({}, { replace: true });
  };

  // Handle pin toggle action
  const handlePinSession = async () => {
    if (!sessionId) {
      setToast({
        message: "Send a message first to pin this conversation.",
        type: "error",
      });
      return;
    }
    try {
      await chatService.togglePin(sessionId);
      queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
      setToast({ message: "Conversation pin toggled.", type: "success" });
    } catch (err: any) {
      setToast({ message: "Failed to pin conversation.", type: "error" });
    }
  };

  // Handle command selections
  const handleSelectCommand = (cmd: (typeof COMMANDS)[0]) => {
    if (cmd.name === "/compare") {
      setIsComparePickerOpen(true);
      setCompareSelection([]);
      setDropdownSearch("");
      setTimeout(() => {
        dropdownSearchInputRef.current?.focus();
      }, 50);
      return;
    }

    if (cmd.name === "/pin") {
      handlePinSession();
      setIsDropdownOpen(false);
      setDropdownSearch("");
      return;
    }

    if (cmd.name === "/new") {
      handleClearSession();
      setIsDropdownOpen(false);
      setDropdownSearch("");
      return;
    }

    // Prefixes: /summarize, /detailed, /table, /bullets, /eli5
    insertCommandPrefix(cmd.name);
  };

  // Keyboard navigation for dropdown
  const handleDropdownKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((prev) =>
        totalDropdownItemsCount > 0 ? (prev + 1) % totalDropdownItemsCount : 0,
      );
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((prev) =>
        totalDropdownItemsCount > 0
          ? (prev - 1 + totalDropdownItemsCount) % totalDropdownItemsCount
          : 0,
      );
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (isComparePickerOpen) {
        if (filteredDocs[activeIndex]) {
          handleToggleCompareDocument(filteredDocs[activeIndex]);
        }
      } else {
        if (activeIndex < filteredCommands.length) {
          handleSelectCommand(filteredCommands[activeIndex]);
        } else {
          const docIdx = activeIndex - filteredCommands.length;
          if (filteredDocs[docIdx]) {
            handleSelectDocument(filteredDocs[docIdx]);
          }
        }
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      setIsDropdownOpen(false);
      setIsComparePickerOpen(false);
      setCompareSelection([]);
      textareaRef.current?.focus();
    }
  };

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLDivElement>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const isCancelledRef = useRef(false);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordingTimeoutRef = useRef<number | null>(null);
  const timerIntervalRef = useRef<number | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Toast auto-dismiss
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), 3000);
    return () => clearTimeout(timer);
  }, [toast]);

  // Sync external changes of inputValue (e.g. clear or load) to contenteditable div
  useEffect(() => {
    if (textareaRef.current && textareaRef.current.innerHTML !== inputValue) {
      textareaRef.current.innerHTML = inputValue;
    }
  }, [inputValue]);

  // Returns the current sessionId, creating one lazily if needed
  const ensureSession = useCallback(async (): Promise<string | null> => {
    if (sessionId) return sessionId;
    try {
      const newSession = await chatService.createSession();
      activeSessionIdRef.current = newSession.id;
      setSessionId(newSession.id);
      setSearchParams({ session: newSession.id }, { replace: true });
      queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
      return newSession.id;
    } catch (err: any) {
      const msg =
        err?.response?.data?.detail ||
        err?.message ||
        "Failed to create chat session.";
      setError(msg);
      return null;
    }
  }, [sessionId, setSearchParams, queryClient]);

  const handleSend = useCallback(
    async (content?: string) => {
      if (isSendingRef.current) return;
      isSendingRef.current = true;
      setIsSending(true);

      const rawText = content ?? getPlainText(inputValue);
      const text = rawText.trim();
      if (!text || isLoading || isStreaming) {
        isSendingRef.current = false;
        setIsSending(false);
        return;
      }

      // Clear/lock the input as part of the same synchronous guarded action, not before the guard is set.
      setInputValue("");
      const docId =
        attachedDocument?.id ||
        (uploadedFile?.status === "ready" ? uploadedFile.id : undefined);
      const currentAttachedDocument = attachedDocument;
      const currentUploadedFile = uploadedFile;
      setUploadedFile(null);
      setAttachedDocument(null);

      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      const controller = new AbortController();
      abortControllerRef.current = controller;

      const tempUserMsgId = `temp-${Date.now()}`;
      const tempAssistantId = `temp-assistant-${Date.now()}`;

      try {
        setError(null);

        // Lazily create a session on the first message
        const sid = await ensureSession();
        if (!sid) return;

        let attachedFile = undefined;
        if (currentAttachedDocument) {
          attachedFile = {
            name: currentAttachedDocument.filename,
            size: currentAttachedDocument.file_size || 0,
          };
        } else if (currentUploadedFile?.status === "ready") {
          attachedFile = {
            name: currentUploadedFile.file.name,
            size: currentUploadedFile.file.size,
          };
        }

        // Optimistically add user message
        const tempUserMsg: MessageResponse = {
          id: tempUserMsgId,
          session_id: sid,
          role: "user",
          content: text,
          created_at: new Date().toISOString(),
          citations: [],
          attached_file: attachedFile,
        };

        const tempAssistantMsg: MessageResponse = {
          id: tempAssistantId,
          session_id: sid,
          role: "assistant",
          content: "",
          created_at: new Date().toISOString(),
          citations: [],
        };

        setMessages((prev) => [...prev, tempUserMsg, tempAssistantMsg]);
        setIsLoading(true);
        setIsStreaming(true);

        await new Promise<void>((resolve, reject) => {
          const handleAbort = () => {
            reject(new Error("Aborted"));
          };
          controller.signal.addEventListener("abort", handleAbort);

          chatService.sendQuery(
            sid,
            text,
            (token) => {
              setIsLoading(false);
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === tempAssistantId
                    ? { ...msg, content: msg.content + token }
                    : msg,
                ),
              );
            },
            (
              citations,
              messageId,
              followUpQuestions,
              generatedSql,
              answer,
              chartSpec,
              resolvedModel,
              wasFallback,
              fallbackModelName,
            ) => {
              controller.signal.removeEventListener("abort", handleAbort);
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === tempAssistantId
                    ? {
                        ...msg,
                        id: messageId,
                        citations,
                        follow_up_questions: followUpQuestions,
                        generated_sql: generatedSql,
                        content: answer || msg.content,
                        chart_spec: chartSpec || null,
                        resolved_model: resolvedModel,
                        was_fallback: wasFallback,
                        fallback_model_name: fallbackModelName,
                      }
                    : msg,
                ),
              );
              if (generatedSql && attachedDatabase) {
                setLockedDbConnectionId(attachedDatabase.id);
              }
              queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
              resolve();
            },
            (err, status) => {
              controller.signal.removeEventListener("abort", handleAbort);
              if (status !== 429) {
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === tempAssistantId && msg.content === ""
                      ? { ...msg, content: "Error: Failed to get response." }
                      : msg,
                  ),
                );
              }
              const errorObj: any = new Error(err);
              errorObj.status = status;
              reject(errorObj);
            },
            docId,
            selectedModel?.id,
            controller.signal,
            attachedDatabase?.id || undefined,
          );
        });
      } catch (err: any) {
        if (err.message === "Aborted") {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === tempAssistantId
                ? {
                    ...msg,
                    content: msg.content.trim()
                      ? msg.content
                      : "Response generation was interrupted.",
                  }
                : msg,
            ),
          );
        } else if (err.status === 429) {
          // Restore input and remove optimistic messages
          setInputValue(text);
          setError(err.message);
          setMessages((prev) =>
            prev.filter(
              (msg) => msg.id !== tempUserMsgId && msg.id !== tempAssistantId,
            ),
          );
        } else {
          setError(err.message || String(err));
        }
      } finally {
        isSendingRef.current = false;
        setIsSending(false);
        setIsLoading(false);
        setIsStreaming(false);
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null;
        }
      }
    },
    [
      inputValue,
      isLoading,
      isStreaming,
      ensureSession,
      uploadedFile,
      attachedDocument,
      queryClient,
      selectedModel,
    ],
  );

  const handleCancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  }, []);

  const urlSessionId = searchParams.get("session");
  const autoQuery = searchParams.get("q");

  // Listen to URL changes to load sessions or clear state
  useEffect(() => {
    // If URL has no session, clear current session (New Chat scenario)
    if (!urlSessionId) {
      if (sessionId !== null) {
        activeSessionIdRef.current = null;
        setSessionId(null);
        setLockedDbConnectionId(null);
        setAttachedDatabase(null);
        setMessages([]);
        setUploadedFile(null);
        setAttachedDocument(null);
        setInputValue("");
        if (textareaRef.current) textareaRef.current.style.height = "auto";
      }
      return;
    }

    // If URL session matches the currently loaded session or the active transitioning session, do nothing
    if (
      urlSessionId === sessionId ||
      urlSessionId === activeSessionIdRef.current
    )
      return;

    // Otherwise, load the new session from the backend
    const loadSession = async () => {
      try {
        const session = await chatService.getSession(urlSessionId);
        activeSessionIdRef.current = session.id;
        setSessionId(session.id);
        setLockedDbConnectionId(session.db_connection_id || null);
        // Sort chronologically, fallback to role for identical timestamps
        const sortedMessages = [...session.messages].sort((a, b) => {
          const timeDiff =
            new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
          if (timeDiff !== 0) return timeDiff;
          if (a.role === "user" && b.role === "assistant") return -1;
          if (a.role === "assistant" && b.role === "user") return 1;
          return 0;
        });
        setMessages(sortedMessages);
        setUploadedFile(null); // clear any attached file from previous chat
        setAttachedDocument(null);

        if (autoQuery) {
          // Small delay to let state settle
          setTimeout(() => {
            handleSendDirect(session.id, autoQuery);
          }, 100);
        }
      } catch (err: any) {
        const msg =
          err?.response?.data?.detail ||
          err?.message ||
          "Failed to load chat session.";
        setError(msg);
      }
    };

    loadSession();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [urlSessionId, autoQuery]);

  // Synchronize attached database and lock state based on loaded session and userDatabases
  useEffect(() => {
    if (!sessionId || !isUserDatabasesLoaded) {
      return;
    }

    if (lockedDbConnectionId) {
      const foundDb = userDatabases.find(
        (dbConn: any) => dbConn.id === lockedDbConnectionId,
      );
      if (foundDb) {
        setAttachedDatabase(foundDb);
      } else {
        setAttachedDatabase(null);
        setToast({
          message: "Previously connected database is no longer available",
          type: "error",
        });
      }
    } else {
      setAttachedDatabase(null);
    }
  }, [sessionId, lockedDbConnectionId, userDatabases, isUserDatabasesLoaded]);

  // Direct send that takes sessionId as parameter (for use before state updates)
  const handleSendDirect = async (sid: string, content: string) => {
    const text = content.trim();
    if (!text || isLoading || isStreaming) return;

    const tempUserMsg: MessageResponse = {
      id: `temp-${Date.now()}`,
      session_id: sid,
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
      citations: [],
    };

    const tempAssistantId = `temp-assistant-${Date.now()}`;
    const tempAssistantMsg: MessageResponse = {
      id: tempAssistantId,
      session_id: sid,
      role: "assistant",
      content: "",
      created_at: new Date().toISOString(),
      citations: [],
    };

    setMessages((prev) => [...prev, tempUserMsg, tempAssistantMsg]);
    setIsLoading(true);
    setIsStreaming(true);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    chatService.sendQuery(
      sid,
      text,
      (token) => {
        setIsLoading(false);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === tempAssistantId
              ? { ...msg, content: msg.content + token }
              : msg,
          ),
        );
      },
      (
        citations,
        messageId,
        followUpQuestions,
        generatedSql,
        answer,
        chartSpec,
        resolvedModel,
        wasFallback,
        fallbackModelName,
      ) => {
        setIsStreaming(false);
        setIsLoading(false);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === tempAssistantId
              ? {
                  ...msg,
                  id: messageId,
                  citations,
                  follow_up_questions: followUpQuestions,
                  generated_sql: generatedSql,
                  content: answer || msg.content,
                  chart_spec: chartSpec || null,
                  resolved_model: resolvedModel,
                  was_fallback: wasFallback,
                  fallback_model_name: fallbackModelName,
                }
              : msg,
          ),
        );
        if (generatedSql && attachedDatabase) {
          setLockedDbConnectionId(attachedDatabase.id);
        }
        queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null;
        }
      },
      (err) => {
        setIsStreaming(false);
        setIsLoading(false);
        setError(err);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === tempAssistantId && msg.content === ""
              ? { ...msg, content: "Error: Failed to get response." }
              : msg,
          ),
        );
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null;
        }
      },
      undefined,
      selectedModel?.id,
      controller.signal,
      attachedDatabase?.id || undefined,
    );
  };

  const toggleCitations = (messageId: string) => {
    setExpandedCitations((prev) => {
      const next = new Set(prev);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        next.add(messageId);
      }
      return next;
    });
  };

  const toggleSql = (messageId: string) => {
    setExpandedSqls((prev) => {
      const next = new Set(prev);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        next.add(messageId);
      }
      return next;
    });
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Reset file input immediately so the same file can be re-selected
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    setAttachedDocument(null); // clear selected catalog document

    // Show the file pill in "uploading" state
    setUploadedFile({ file, status: "uploading" });

    // Lazily create a session if this is the first interaction
    const sid = await ensureSession();
    if (!sid) {
      setUploadedFile({
        file,
        status: "ready",
        error: "No session available.",
      });
      return;
    }

    try {
      const response = await chatService.uploadPrivateDocument(sid, file);
      setUploadedFile({ file, status: "ready", id: response.id });
      setToast({
        message: "Document ready. You can now ask questions about it.",
        type: "success",
      });

      // Auto-insert file name as a pill in contenteditable
      setTimeout(() => {
        if (textareaRef.current) {
          textareaRef.current.focus();
          const selection = window.getSelection();
          if (selection) {
            if (
              selection.rangeCount === 0 ||
              !textareaRef.current.contains(selection.anchorNode)
            ) {
              const range = document.createRange();
              range.selectNodeContents(textareaRef.current);
              range.collapse(false); // collapse to end
              selection.removeAllRanges();
              selection.addRange(range);
            }
          }
          insertPillAtCursor(file.name);
        }
      }, 50);
    } catch (err: any) {
      const msg =
        err?.response?.data?.detail ||
        err?.message ||
        "Failed to upload document.";
      setUploadedFile({ file, status: "ready", error: msg });
      setToast({ message: msg, type: "error" });
    }
  };

  const handleRemoveFile = () => {
    setUploadedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const startRecording = async () => {
    try {
      setError(null);
      isCancelledRef.current = false;
      audioChunksRef.current = [];
      setRecordingSeconds(0);

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        if (isCancelledRef.current) {
          return;
        }
        const audioBlob = new Blob(audioChunksRef.current, {
          type: mediaRecorder.mimeType || "audio/webm",
        });
        setIsTranscribing(true);
        try {
          const result = await chatService.transcribeVoiceQuery(audioBlob);
          if (result && result.text) {
            setInputValue((prev) =>
              prev
                ? (prev.endsWith(" ") ? prev : prev + " ") + result.text
                : result.text,
            );
          }
        } catch (err: any) {
          console.error(err);
          setToast({
            message: "Could not transcribe audio. Please try again.",
            type: "error",
          });
        } finally {
          setIsTranscribing(false);
        }
      };

      // Web Audio API setup for waveform visualizer
      const AudioContextClass =
        window.AudioContext || (window as any).webkitAudioContext;
      if (AudioContextClass) {
        const audioContext = new AudioContextClass();
        audioContextRef.current = audioContext;
        const source = audioContext.createMediaStreamSource(stream);
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 64;
        source.connect(analyser);
        analyserRef.current = analyser;

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const updateWaveform = () => {
          if (!analyserRef.current) return;
          analyserRef.current.getByteFrequencyData(dataArray);
          const numBars = 10;
          const levels: number[] = [];
          for (let i = 0; i < numBars; i++) {
            const val = dataArray[i * 2 + 1] || 0;
            levels.push(val / 255);
          }
          setAudioLevels(levels);
          animationFrameRef.current = requestAnimationFrame(updateWaveform);
        };
        animationFrameRef.current = requestAnimationFrame(updateWaveform);
      }

      mediaRecorder.start();
      setIsRecording(true);

      // Auto stop at 2 minutes (120,000ms)
      recordingTimeoutRef.current = window.setTimeout(() => {
        stopRecording();
      }, 120000);

      // Timer interval
      timerIntervalRef.current = window.setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err: any) {
      console.error(err);
      if (
        err.name === "NotAllowedError" ||
        err.name === "PermissionDeniedError"
      ) {
        setError(
          "Microphone access denied. Please allow microphone access to use voice input.",
        );
      } else {
        setError("Could not start recording. Please try again.");
      }
    }
  };

  const stopRecording = useCallback(() => {
    if (recordingTimeoutRef.current) {
      clearTimeout(recordingTimeoutRef.current);
      recordingTimeoutRef.current = null;
    }
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      mediaRecorderRef.current.stop();
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    analyserRef.current = null;

    setIsRecording(false);
  }, []);

  const cancelRecording = useCallback(() => {
    isCancelledRef.current = true;
    stopRecording();
  }, [stopRecording]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (recordingTimeoutRef.current)
        clearTimeout(recordingTimeoutRef.current);
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
      if (animationFrameRef.current)
        cancelAnimationFrame(animationFrameRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (
        audioContextRef.current &&
        audioContextRef.current.state !== "closed"
      ) {
        audioContextRef.current.close();
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const formatTime = (dateStr: string) => {
    return new Date(dateStr).toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
    });
  };

  const formatTimer = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  const getFollowUpQuestions = useCallback((msg: MessageResponse): string[] => {
    if (msg.follow_up_questions && msg.follow_up_questions.length > 0) {
      return msg.follow_up_questions;
    }
    if (msg.content.includes("[FOLLOW_UP]")) {
      const parts = msg.content.split("[FOLLOW_UP]");
      const rawQuestions = parts[1] || "";
      return rawQuestions
        .split("\n")
        .map((q) =>
          q
            .trim()
            .replace(/^[-*\d.]+\s*/, "")
            .trim(),
        )
        .filter(Boolean);
    }
    return [];
  }, []);

  const handleFollowUpClick = useCallback((q: string) => {
    setInputValue(q);
    setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.focus();

        // Move caret to the end of the contenteditable div
        const range = document.createRange();
        const sel = window.getSelection();
        range.selectNodeContents(textareaRef.current);
        range.collapse(false); // false means collapse to end
        sel?.removeAllRanges();
        sel?.addRange(range);
      }
    }, 0);
  }, []);

  const renderFollowUpQuestions = (msg: MessageResponse, index: number) => {
    const isLatestAssistant = index === messages.length - 1;
    if (!isLatestAssistant) return null;

    const questions = getFollowUpQuestions(msg);
    if (questions.length === 0) return null;

    return (
      <div className="mt-3 ml-2 mr-auto max-w-2xl animate-fade-in">
        <p className="text-xs font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-wide mb-2 flex items-center gap-1.5 select-none">
          Suggested Follow-ups
        </p>
        <div className="flex flex-col gap-2">
          {questions.map((q, idx) => (
            <button
              key={idx}
              onClick={() => handleFollowUpClick(q)}
              disabled={isLoading || isStreaming || isSending}
              className="text-left w-full -ml-2 px-2 py-1.5 rounded-lg dark:hover:bg-slate-500/10 hover:bg-slate-400/10 text-indigo-700 dark:text-indigo-300 text-sm font-medium cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:transform-none"
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    );
  };

  const isAdmin = user?.role?.is_admin ?? false;
  const isFileUploading = uploadedFile?.status === "uploading";
  // Send is disabled while the AI is replying OR a document is still being processed OR recording/transcribing is active OR a message is sending
  const getPlainText = (html: string): string => {
    const tempDiv = document.createElement("div");
    tempDiv.innerHTML = html;
    const pills = tempDiv.querySelectorAll("[data-filename]");
    pills.forEach((pill) => {
      const filename = pill.getAttribute("data-filename");
      pill.replaceWith(document.createTextNode(filename || ""));
    });
    return (tempDiv.innerText || tempDiv.textContent || "").trim();
  };

  const isSendDisabled =
    !getPlainText(inputValue) ||
    isLoading ||
    isStreaming ||
    isFileUploading ||
    isRecording ||
    isTranscribing ||
    isSending;

  const handleContentEditableInput = (e: React.FormEvent<HTMLDivElement>) => {
    const html = e.currentTarget.innerHTML;
    setInputValue(html);

    // Clear attachment if filename pill is deleted in input box
    const tempDiv = document.createElement("div");
    tempDiv.innerHTML = html;
    const pills = tempDiv.querySelectorAll("[data-filename]");

    let docAttachedExists = false;
    let uploadedFileExists = false;

    pills.forEach((pill) => {
      const filename = pill.getAttribute("data-filename");
      if (attachedDocument && filename === attachedDocument.filename) {
        docAttachedExists = true;
      }
      if (
        uploadedFile?.status === "ready" &&
        filename === uploadedFile.file.name
      ) {
        uploadedFileExists = true;
      }
    });

    if (attachedDocument && !docAttachedExists) {
      setAttachedDocument(null);
    }
    if (uploadedFile?.status === "ready" && !uploadedFileExists) {
      setUploadedFile(null);
    }

    // Autocomplete dropdown check
    const selection = window.getSelection();
    if (selection && selection.rangeCount > 0 && textareaRef.current) {
      const range = selection.getRangeAt(0);
      const preCaretRange = range.cloneRange();
      preCaretRange.selectNodeContents(textareaRef.current);
      preCaretRange.setEnd(range.endContainer, range.endOffset);
      const textBeforeCursor = preCaretRange.toString();

      const isSlashTrigger =
        textBeforeCursor === "/" ||
        textBeforeCursor.endsWith(" /") ||
        textBeforeCursor.endsWith("\n/");
      if (isSlashTrigger) {
        setIsDropdownOpen(true);
        setDropdownSearch("");
      }
    }
  };

  const handleContentEditableKeyDown = (
    e: React.KeyboardEvent<HTMLDivElement>,
  ) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (isSending) return;
      handleSend();
    }
  };

  const handleContentEditablePaste = (
    e: React.ClipboardEvent<HTMLDivElement>,
  ) => {
    e.preventDefault();
    const text = e.clipboardData.getData("text/plain");

    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return;

    const range = selection.getRangeAt(0);
    range.deleteContents();

    const textNode = document.createTextNode(text);
    range.insertNode(textNode);

    range.setStartAfter(textNode);
    range.setEndAfter(textNode);
    selection.removeAllRanges();
    selection.addRange(range);

    if (textareaRef.current) {
      setInputValue(textareaRef.current.innerHTML);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const greeting = useMemo(() => {
    const greetings = ["Hello", "Hey", "Hi"];
    const randomGreeting =
      greetings[Math.floor(Math.random() * greetings.length)];

    const hour = new Date().getHours();
    let timeGreeting = "evening";
    if (hour >= 5 && hour < 12) timeGreeting = "morning";
    else if (hour >= 12 && hour < 17) timeGreeting = "afternoon";

    const firstName = user?.full_name?.split(" ")[0] || "there";

    return `${randomGreeting} ${firstName}, good ${timeGreeting}!`;
  }, [user?.full_name]);

  const groupedModels = availableModels.reduce(
    (groups, model) => {
      const providerKey = model.provider_id || "openai_compat";
      if (!groups[providerKey]) {
        groups[providerKey] = [];
      }
      groups[providerKey].push(model);
      return groups;
    },
    {} as Record<string, AvailableModel[]>,
  );

  return (
    <div className="flex flex-col -m-6 h-[calc(100vh-4rem)] relative">
      {/* Toast notification */}
      {toast && (
        <div
          className={`fixed top-4 right-4 z-50 rounded-xl px-4 py-3 shadow-lg text-sm font-medium transition-all ${
            toast.type === "success"
              ? "bg-emerald-500 text-white"
              : "bg-red-500 text-white"
          }`}
        >
          {toast.message}
        </div>
      )}

      {/* Chat Session Header */}
      {sessionId && (
        <div className="absolute top-4 right-6 z-20 select-none">
          <ReportGenerationPanel sessionId={sessionId} />
        </div>
      )}

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        {messages.length === 0 && !isLoading ? (
          <div className=""></div>
        ) : (
          <div className="max-w-3xl mx-auto w-full space-y-4">
            {messages.map((msg, index) => {
              const isLatestAssistant =
                index === messages.length - 1 && isStreaming;
              const parsed = parseDbQueryMessage(
                msg,
                isLatestAssistant,
                !!attachedDatabase,
              );

              if (msg.role === "user") {
                return (
                  <div key={msg.id}>
                    <div className="ml-auto max-w-2xl flex flex-col items-end gap-2">
                      {msg.attached_file && (
                        <div className="inline-flex items-center gap-2 px-3 py-2 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-300 text-sm shadow-sm hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors duration-150">
                          <FileText className="w-4 h-4 text-indigo-500 dark:text-indigo-400" />
                          <span className="max-w-[180px] truncate font-bold">
                            {msg.attached_file.name}
                          </span>
                          <span className="text-slate-400 dark:text-slate-500">
                            {formatFileSize(msg.attached_file.size)}
                          </span>
                        </div>
                      )}
                      <div className="bg-indigo-700 dark:bg-indigo-600 text-white rounded-2xl rounded-br-md px-4 py-3">
                        <p className="whitespace-pre-wrap">
                          {renderMessageContentWithHighlights(msg)}
                        </p>
                        <p className="text-indigo-300 dark:text-indigo-200 text-xs mt-1 text-right">
                          {formatTime(msg.created_at)}
                        </p>
                      </div>
                    </div>
                  </div>
                );
              }

              if (
                !parsed.isDbQuery &&
                msg.content === "" &&
                msg.citations.length === 0
              ) {
                return null;
              }

              return (
                <div key={msg.id}>
                  <div className="mr-auto max-w-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm text-slate-800 dark:text-slate-100">
                    {(() => {
                      if (parsed.isDbQuery) {
                        return (
                          <div className="flex flex-col gap-2.5">
                            {parsed.status === "generating_sql" && (
                              <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 italic text-sm select-none py-1">
                                <Loader2 className="w-4 h-4 animate-spin text-indigo-500" />
                                <span>
                                  Thinking... Translating your request to SQL...
                                </span>
                              </div>
                            )}

                            {parsed.sql && (
                              <div>
                                <button
                                  onClick={() => toggleSql(msg.id)}
                                  className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-300 font-medium py-1 transition-colors duration-150 focus:outline-none w-full text-left select-none"
                                >
                                  <span>View SQL</span>
                                  {expandedSqls.has(msg.id) ? (
                                    <ChevronUp className="w-3.5 h-3.5" />
                                  ) : (
                                    <ChevronDown className="w-3.5 h-3.5" />
                                  )}
                                </button>
                                {expandedSqls.has(msg.id) && (
                                  <pre className="mt-2 p-3 bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-850 rounded-lg text-xs font-mono text-slate-700 dark:text-slate-300 overflow-x-auto whitespace-pre-wrap leading-relaxed">
                                    <code>{parsed.sql}</code>
                                  </pre>
                                )}
                              </div>
                            )}

                            {parsed.status === "executing_query" && (
                              <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 italic text-sm select-none py-1">
                                <Loader2 className="w-4 h-4 animate-spin text-indigo-500" />
                                <span>Executing query...</span>
                              </div>
                            )}

                            {parsed.answer && (
                              <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                components={markdownComponents}
                              >
                                {parsed.answer}
                              </ReactMarkdown>
                            )}
                          </div>
                        );
                      }

                      return (
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={markdownComponents}
                        >
                          {parsed.answer}
                        </ReactMarkdown>
                      );
                    })()}

                    <div className="flex items-center gap-2 mt-1 select-none">
                      <span className="text-slate-400 dark:text-slate-500 text-xs">
                        {formatTime(msg.created_at)}
                      </span>
                      {msg.resolved_model ? (
                        <>
                          <span className="text-slate-300 dark:text-slate-700">
                            •
                          </span>
                          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-indigo-50 dark:bg-indigo-950/20 text-indigo-700 dark:text-indigo-400 border border-indigo-200/45 dark:border-indigo-900/50">
                            <span className="font-bold">Auto: </span>{" "}
                            {msg.resolved_model}
                          </span>
                        </>
                      ) : msg.model ? (
                        <>
                          <span className="text-slate-300 dark:text-slate-700">
                            •
                          </span>
                          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200/40 dark:border-slate-700/50">
                            {msg.model.display_name}
                          </span>
                        </>
                      ) : null}
                    </div>
                    {msg.was_fallback && msg.fallback_model_name && (
                      <span className="text-xs text-amber-600 dark:text-amber-500 mt-1 block">
                        Responded using fallback model:{" "}
                        {msg.fallback_model_name}
                      </span>
                    )}

                    {msg.chart_spec && <ChartRenderer spec={msg.chart_spec} />}

                    {msg.citations.length > 0 && (
                      <div className="mt-2">
                        <button
                          onClick={() => toggleCitations(msg.id)}
                          className="text-indigo-600 dark:text-indigo-400 text-sm hover:underline cursor-pointer flex items-center gap-1"
                        >
                          Sources ({msg.citations.length})
                          {expandedCitations.has(msg.id) ? (
                            <ChevronUp className="w-4 h-4" />
                          ) : (
                            <ChevronDown className="w-4 h-4" />
                          )}
                        </button>
                        {expandedCitations.has(msg.id) && (
                          <div className="mt-2 p-3 bg-slate-50 dark:bg-slate-850 rounded-xl border border-slate-200 dark:border-slate-800 space-y-2 max-h-60 overflow-y-auto custom-scrollbar">
                            {msg.citations.map((cite) => (
                              <div
                                key={cite.id}
                                className="text-xs text-slate-600 dark:text-slate-300 pb-2 border-b border-slate-200/60 dark:border-slate-800/80 last:border-b-0 last:pb-0"
                              >
                                <p className="font-semibold text-slate-700 dark:text-slate-200 mb-1 flex items-center gap-1">
                                  <FileText className="w-3.5 h-3.5 text-indigo-500" />
                                  {cite.filename}
                                </p>
                                <div className="leading-relaxed text-slate-500 dark:text-slate-400">
                                  <ReactMarkdown
                                    remarkPlugins={[remarkGfm]}
                                    components={markdownComponents}
                                  >
                                    {cite.chunk_text.length > 150
                                      ? `${cite.chunk_text.slice(0, 150)}...`
                                      : cite.chunk_text}
                                  </ReactMarkdown>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  {msg.role === "assistant" &&
                    renderFollowUpQuestions(msg, index)}
                </div>
              );
            })}

            {/* Typing indicator */}
            {isLoading && !attachedDatabase && (
              <div className="mr-auto w-fit bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
                <div className="flex items-center gap-1.5">
                  <span
                    className="w-2 h-2 bg-slate-400 dark:bg-slate-600 rounded-full animate-bounce"
                    style={{ animationDelay: "0ms" }}
                  />
                  <span
                    className="w-2 h-2 bg-slate-400 dark:bg-slate-600 rounded-full animate-bounce"
                    style={{ animationDelay: "150ms" }}
                  />
                  <span
                    className="w-2 h-2 bg-slate-400 dark:bg-slate-600 rounded-full animate-bounce"
                    style={{ animationDelay: "300ms" }}
                  />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input bar */}
      <div
        className={
          messages.length === 0
            ? "min-h-[calc(100vh-80px)] flex flex-col items-center justify-center"
            : ""
        }
      >
        {messages.length === 0 && !isLoading ? (
          <div className="flex flex-col items-center justify-center pb-8 space-y-2">
            <h2 className="text-3xl font-semibold text-slate-800 dark:text-slate-100 tracking-tight">
              {greeting}
            </h2>
            <p className="text-slate-500 dark:text-slate-400 text-lg">
              How can I help you today?
            </p>
          </div>
        ) : null}

        <div className="w-full">
          <div className="">
            <div
              className={`relative max-w-3xl mx-auto h-full ${messages.length === 0 ? "mb-32" : ""}`}
            >
              {/* Document Autocomplete Dropdown */}
              {isDropdownOpen && (
                <div
                  ref={dropdownRef}
                  className="absolute bottom-full left-0 mb-3 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-2xl p-3 z-50 flex flex-col gap-2 transition-all h-96 w-1/2 text-slate-800 dark:text-slate-100"
                >
                  {/* Search Bar */}
                  <div className="relative flex items-center flex-shrink-0">
                    <Search className="absolute left-3 w-4 h-4 text-slate-400 dark:text-slate-500" />
                    <input
                      ref={dropdownSearchInputRef}
                      type="text"
                      value={dropdownSearch}
                      onChange={(e) => setDropdownSearch(e.target.value)}
                      onKeyDown={handleDropdownKeyDown}
                      placeholder={
                        isComparePickerOpen
                          ? "Search documents to compare..."
                          : "Search commands or documents..."
                      }
                      className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-xl pl-9 pr-4 py-2 text-sm text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 placeholder:text-slate-400 dark:placeholder:text-slate-500"
                    />
                  </div>

                  {/* Compare Selection Header */}
                  {isComparePickerOpen && (
                    <div className="flex flex-col gap-1.5 px-1 py-1 border-b border-slate-100 dark:border-slate-800 flex-shrink-0">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">
                          Compare Documents ({compareSelection.length}/2)
                        </span>
                        <button
                          onClick={() => {
                            setIsComparePickerOpen(false);
                            setCompareSelection([]);
                          }}
                          className="text-xs text-indigo-600 dark:text-indigo-400 hover:text-indigo-700 font-medium"
                        >
                          Cancel
                        </button>
                      </div>
                      {compareSelection.length > 0 && (
                        <div className="flex flex-wrap gap-1.5">
                          {compareSelection.map((doc) => (
                            <span
                              key={doc.id}
                              className="inline-flex items-center gap-1 bg-indigo-50 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 text-xs px-2 py-0.5 rounded-full border border-indigo-200 dark:border-indigo-900"
                            >
                              <span className="truncate max-w-[120px]">
                                {doc.filename}
                              </span>
                              <button
                                onClick={() => handleToggleCompareDocument(doc)}
                                className="text-indigo-500 hover:text-indigo-700 rounded-full"
                              >
                                <X className="w-3 h-3" />
                              </button>
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Loading state */}
                  {isLoadingAuth || isLoadingPersonal ? (
                    <div className="flex-1 flex items-center justify-center gap-2 text-slate-500 dark:text-slate-400 text-sm">
                      <Loader2 className="w-4 h-4 animate-spin text-indigo-600 dark:text-indigo-400" />
                      Loading...
                    </div>
                  ) : isComparePickerOpen ? (
                    /* Compare Picker Mode Documents List */
                    filteredDocs.length === 0 ? (
                      <div className="flex-1 flex items-center justify-center text-slate-400 dark:text-slate-500 text-sm">
                        No accessible documents found.
                      </div>
                    ) : (
                      <div className="flex-1 overflow-y-auto space-y-0.5 custom-scrollbar pr-1">
                        {filteredDocs.map((doc, idx) => {
                          const Icon =
                            FILE_TYPE_ICON[doc.file_type] || FileText;
                          const isSelected = compareSelection.some(
                            (d) => d.id === doc.id,
                          );
                          const isActive = idx === activeIndex;
                          return (
                            <button
                              key={doc.id}
                              onClick={() => handleToggleCompareDocument(doc)}
                              className={`w-full flex items-center justify-between px-3 py-2 rounded-xl text-left transition-colors ${
                                isActive
                                  ? "bg-indigo-700 dark:bg-indigo-600 text-white"
                                  : isSelected
                                    ? "bg-indigo-50 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-900"
                                    : "text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
                              }`}
                            >
                              <div className="flex items-center gap-2.5 min-w-0">
                                <Icon
                                  className={`w-4 h-4 flex-shrink-0 ${isActive ? "text-white" : "text-indigo-600 dark:text-indigo-400"}`}
                                />
                                <span className="truncate text-sm font-medium pr-1.5">
                                  {doc.filename}
                                </span>
                              </div>
                              <div className="flex items-center gap-1.5 flex-shrink-0 text-xs">
                                {isSelected && (
                                  <span
                                    className={`font-semibold ${isActive ? "text-white" : "text-indigo-600 dark:text-indigo-400"}`}
                                  >
                                    ✓
                                  </span>
                                )}
                                <span
                                  className={`px-2 py-0.5 rounded-md ${
                                    isActive
                                      ? "bg-white/20 text-white"
                                      : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400"
                                  }`}
                                >
                                  {doc.owner_type === "private"
                                    ? "Personal"
                                    : "Org"}
                                </span>
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    )
                  ) : /* Standard Mode - Commands & Documents */
                  filteredCommands.length === 0 && filteredDocs.length === 0 ? (
                    <div className="flex-1 flex items-center justify-center text-slate-400 dark:text-slate-500 text-sm">
                      No commands or documents found.
                    </div>
                  ) : (
                    <div className="flex-1 overflow-y-auto space-y-4 custom-scrollbar pr-1">
                      {/* Commands Section */}
                      {filteredCommands.length > 0 && (
                        <div className="space-y-0.5">
                          <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 px-3 py-1">
                            Commands
                          </div>
                          {filteredCommands.map((cmd, idx) => {
                            const isActive = idx === activeIndex;
                            return (
                              <button
                                key={cmd.name}
                                onClick={() => handleSelectCommand(cmd)}
                                className={`w-full flex items-center px-3 py-1.5 rounded-xl text-left transition-colors ${
                                  isActive
                                    ? "bg-indigo-700 dark:bg-indigo-600 text-white"
                                    : "text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
                                }`}
                              >
                                <div className="flex flex-col min-w-0">
                                  <span className="text-sm font-semibold">
                                    {cmd.name}
                                  </span>
                                  <span
                                    className={`text-xs ${isActive ? "text-indigo-200" : "text-slate-400 dark:text-slate-500"} truncate`}
                                  >
                                    {cmd.description}
                                  </span>
                                </div>
                              </button>
                            );
                          })}
                        </div>
                      )}

                      {/* Attach Documents Section */}
                      {filteredDocs.length > 0 && (
                        <div className="space-y-0.5">
                          <div className="text-sm font-semibold text-slate-500 dark:text-slate-400 px-3 py-1">
                            Attach documents
                          </div>
                          {filteredDocs.map((doc, idx) => {
                            const actualIdx = idx + filteredCommands.length;
                            const isActive = actualIdx === activeIndex;
                            const Icon =
                              FILE_TYPE_ICON[doc.file_type] || FileText;
                            return (
                              <button
                                key={doc.id}
                                onClick={() => handleSelectDocument(doc)}
                                className={`w-full flex items-center justify-between px-3 py-2 rounded-xl text-left transition-colors ${
                                  isActive
                                    ? "bg-indigo-700 dark:bg-indigo-600 text-white"
                                    : "text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
                                }`}
                              >
                                <div className="flex items-center gap-2.5 min-w-0">
                                  <Icon
                                    className={`w-4 h-4 flex-shrink-0 ${isActive ? "text-white" : "text-indigo-600 dark:text-indigo-400"}`}
                                  />
                                  <span className="truncate text-sm font-medium pr-1.5">
                                    {doc.filename}
                                  </span>
                                </div>
                                <div className="flex items-center gap-1 flex-shrink-0 text-xs">
                                  <span
                                    className={`px-2 py-0.5 rounded-md ${
                                      isActive
                                        ? "bg-white/20 text-white"
                                        : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400"
                                    }`}
                                  >
                                    {doc.owner_type === "private"
                                      ? "Personal"
                                      : "Org"}
                                  </span>
                                </div>
                              </button>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Input row */}
              <div className="border border-slate-200 dark:border-slate-700 rounded-2xl bg-white dark:bg-slate-900 focus-within:ring-2 focus-within:ring-indigo-500 text-slate-800 dark:text-slate-100 mb-2 p-3 flex flex-col gap-2 shadow-sm">
                {/* File preview inside input */}
                {uploadedFile &&
                  (uploadedFile.status === "uploading" ||
                    uploadedFile.error) && (
                    <div className="px-1 pt-1">
                      <div
                        className={`inline-flex items-center gap-2 px-3 py-2 rounded-xl border text-sm ${
                          uploadedFile.error
                            ? "border-red-200 dark:border-red-900/50 bg-red-50 dark:bg-red-950/20 text-red-700 dark:text-red-400"
                            : uploadedFile.status === "uploading"
                              ? "border-indigo-200 dark:border-indigo-900/50 bg-indigo-50 dark:bg-indigo-950/20 text-indigo-700 dark:text-indigo-400"
                              : "border-emerald-200 dark:border-emerald-900/50 bg-emerald-50 dark:bg-emerald-950/20 text-emerald-700 dark:text-emerald-400"
                        }`}
                      >
                        {uploadedFile.status === "uploading" ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <FileText className="w-4 h-4" />
                        )}

                        <span className="max-w-[180px] truncate">
                          {uploadedFile.file.name}
                        </span>

                        <span>{formatFileSize(uploadedFile.file.size)}</span>

                        {uploadedFile.status !== "uploading" && (
                          <button
                            onClick={handleRemoveFile}
                            className="hover:bg-black/5 dark:hover:bg-white/10 rounded-full p-1"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        )}
                      </div>
                    </div>
                  )}

                {/* Input area */}
                <div className="relative">
                  {isRecording ? (
                    <div className="flex items-center gap-4 py-2.5 h-[40px] w-full px-2 text-slate-800 dark:text-slate-100 z-10">
                      {/* Soundwave animation */}
                      <div className="flex items-center gap-1 h-6">
                        {audioLevels.map((level, idx) => {
                          const height = 6 + level * 18;
                          return (
                            <div
                              key={idx}
                              className="w-0.5 bg-indigo-600 dark:bg-indigo-500 rounded-full transition-all duration-100"
                              style={{ height: `${height}px` }}
                            />
                          );
                        })}
                      </div>
                      {/* Timer */}
                      <span className="text-slate-500 dark:text-slate-400 text-sm font-medium">
                        {formatTimer(recordingSeconds)}
                      </span>
                      {/* Status */}
                      <span className="text-slate-400 dark:text-slate-500 text-sm select-none">
                        Recording voice query...
                      </span>
                    </div>
                  ) : (
                    <>
                      <div
                        ref={textareaRef}
                        contentEditable={
                          !(
                            isFileUploading ||
                            isLoading ||
                            isStreaming ||
                            isTranscribing ||
                            isSending
                          )
                        }
                        onInput={handleContentEditableInput}
                        onKeyDown={handleContentEditableKeyDown}
                        onPaste={handleContentEditablePaste}
                        onKeyUp={saveSelectionRange}
                        onMouseUp={saveSelectionRange}
                        onClick={saveSelectionRange}
                        className="w-full text-slate-800 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500 bg-transparent px-2 py-1.5 focus:outline-none border-0 overflow-y-auto custom-scrollbar relative z-0 min-h-[40px] max-h-[200px] whitespace-pre-wrap break-words outline-none editable-input"
                        data-placeholder={
                          isFileUploading
                            ? "Please wait while the document is processed..."
                            : "Ask about your documents, or type '/' to see commands and attach documents"
                        }
                        data-empty={
                          !inputValue ||
                          inputValue === "<br>" ||
                          inputValue === "<div><br></div>"
                            ? "true"
                            : "false"
                        }
                        style={{
                          fontFamily: "inherit",
                          fontSize: "inherit",
                          lineHeight: "inherit",
                        }}
                      />
                    </>
                  )}
                </div>

                {/* Lower Actions Row (Claude style) */}
                <div className="flex items-center justify-between border-slate-100 dark:border-slate-800/60 pt-2 px-1">
                  {/* Left Group: Attach File, Model Selector */}
                  <div className="flex items-center gap-2">
                    {!isAdmin && !isRecording && (
                      <>
                        <button
                          onClick={() => fileInputRef.current?.click()}
                          disabled={isFileUploading || isLoading || isStreaming}
                          type="button"
                          className="flex items-center justify-center w-8 h-8 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500 dark:text-slate-400 transition-colors"
                          title="Upload file"
                        >
                          <Plus className="w-4 h-4" strokeWidth={2.5} />
                        </button>

                        <input
                          ref={fileInputRef}
                          type="file"
                          className="hidden"
                          onChange={handleFileUpload}
                        />
                      </>
                    )}

                    {availableModels.length > 0 && !isRecording && (
                      <div ref={modelDropdownRef} className="relative">
                        <button
                          onClick={() =>
                            setIsModelDropdownOpen(!isModelDropdownOpen)
                          }
                          disabled={isLoading || isStreaming}
                          type="button"
                          className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white dark:bg-slate-900 text-slate-650 dark:text-slate-350 hover:bg-slate-50 dark:hover:bg-slate-800 transition-all font-semibold text-sm outline-none"
                        >
                          <span>
                            {selectedModel?.display_name || "Select model"}
                          </span>
                          <ChevronDown
                            className={`w-3 h-3 text-slate-450 transition-transform duration-200 ${isModelDropdownOpen ? "rotate-180" : ""}`}
                          />
                        </button>

                        {isModelDropdownOpen && (
                          <div className="absolute bottom-full left-0 mb-2 w-64 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-xl p-2 z-50 flex flex-col gap-1 text-slate-800 dark:text-slate-100 max-h-72 overflow-y-auto">
                            <button
                              type="button"
                              onClick={() => handleSelectModel(autoModel)}
                              className={`w-full flex items-center justify-between px-2.5 py-2 rounded-xl text-left text-xs transition-colors ${
                                selectedModel?.id === "auto"
                                  ? "bg-indigo-50 dark:bg-indigo-950/40 text-indigo-700 dark:text-indigo-400 font-semibold"
                                  : "hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300"
                              }`}
                            >
                              <span className="text-slate-900 dark:text-slate-100">
                                Auto
                              </span>
                              {selectedModel?.id === "auto" && (
                                <span className="text-indigo-600 dark:text-indigo-400 font-bold">
                                  ✓
                                </span>
                              )}
                            </button>

                            <div className="border-t border-slate-200 dark:border-slate-800/60"></div>

                            {Object.entries(groupedModels).map(
                              ([providerKey, models], groupIdx) => (
                                <div
                                  key={providerKey}
                                  className={
                                    groupIdx > 0
                                      ? "border-slate-100 dark:border-slate-800/60"
                                      : ""
                                  }
                                >
                                  {models.map((model) => (
                                    <button
                                      key={model.id}
                                      type="button"
                                      onClick={() => handleSelectModel(model)}
                                      className={`w-full flex items-center justify-between px-2.5 py-2 rounded-xl text-left text-xs transition-colors ${
                                        selectedModel?.id === model.id
                                          ? "bg-indigo-50 dark:bg-indigo-950/40 text-indigo-700 dark:text-indigo-400 font-semibold"
                                          : "hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300"
                                      }`}
                                    >
                                      <span>{model.display_name}</span>
                                      {selectedModel?.id === model.id && (
                                        <span className="text-indigo-600 dark:text-indigo-400 font-bold">
                                          ✓
                                        </span>
                                      )}
                                    </button>
                                  ))}
                                </div>
                              ),
                            )}
                          </div>
                        )}
                      </div>
                    )}
                    {/* Database selector */}
                    {userDatabases.length > 0 && !isRecording && (
                      <div
                        ref={dbDropdownRef}
                        className="relative flex flex-col items-start gap-1"
                      >
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => {
                              if (!lockedDbConnectionId) {
                                setIsDbDropdownOpen(!isDbDropdownOpen);
                              }
                            }}
                            disabled={isLoading || isStreaming}
                            type="button"
                            className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white dark:bg-slate-900 text-slate-650 dark:text-slate-350 transition-all font-semibold text-sm outline-none ${
                              lockedDbConnectionId
                                ? "cursor-default opacity-85"
                                : "hover:bg-slate-50 dark:hover:bg-slate-800"
                            }`}
                          >
                            <Database className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
                            <span>
                              {attachedDatabase
                                ? attachedDatabase.name
                                : "Select DB"}
                            </span>
                            {lockedDbConnectionId ? (
                              <Lock className="w-3 h-3 text-slate-400 dark:text-slate-500" />
                            ) : (
                              <ChevronDown
                                className={`w-3 h-3 text-slate-455 transition-transform duration-200 ${isDbDropdownOpen ? "rotate-180" : ""}`}
                              />
                            )}
                          </button>
                          {/* View Schema Button */}
                          <div className="relative group">
                            <button
                              type="button"
                              onClick={() => {
                                if (attachedDatabase) {
                                  setIsSchemaModalOpen(true);
                                }
                              }}
                              disabled={!attachedDatabase}
                              className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white dark:bg-slate-900 text-slate-650 dark:text-slate-350 transition-all font-semibold text-sm outline-none border border-slate-200/80 dark:border-slate-800 ${
                                !attachedDatabase
                                  ? "opacity-50 cursor-not-allowed"
                                  : "hover:bg-slate-50 dark:hover:bg-slate-800 hover:text-indigo-600 dark:hover:text-indigo-400 cursor-pointer shadow-xs"
                              }`}
                              title={!attachedDatabase ? "Select a database first" : "View schema for this database"}
                            >
                              <Table className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
                              <span>View Schema</span>
                            </button>
                            {!attachedDatabase && (
                              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 hidden group-hover:block whitespace-nowrap px-2 py-1 bg-slate-800 dark:bg-slate-700 text-white text-[11px] font-medium rounded shadow-md z-50 pointer-events-none">
                                Select a database first
                              </div>
                            )}
                          </div>
                          {lockedDbConnectionId && (
                            <span className="text-xs text-slate-400 dark:text-slate-500 font-medium select-none">
                              Locked to this chat
                            </span>
                          )}
                        </div>

                        {isDbDropdownOpen && !lockedDbConnectionId && (
                          <div className="absolute bottom-full left-0 mb-2 w-64 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-xl p-2 z-50 flex flex-col gap-1 text-slate-800 dark:text-slate-100 max-h-72 overflow-y-auto">
                            <div className="px-2.5 py-1 text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider select-none">
                              Query Database
                            </div>
                            <button
                              type="button"
                              onClick={() => {
                                setAttachedDatabase(null);
                                setIsDbDropdownOpen(false);
                              }}
                              className={`w-full flex items-center justify-between px-2.5 py-2 rounded-xl text-left text-xs transition-colors ${
                                !attachedDatabase
                                  ? "bg-indigo-50 dark:bg-indigo-950/40 text-indigo-700 dark:text-indigo-400 font-semibold"
                                  : "hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300"
                              }`}
                            >
                              <span>No Database</span>
                              {!attachedDatabase && (
                                <span className="text-indigo-600 dark:text-indigo-400 font-bold">
                                  ✓
                                </span>
                              )}
                            </button>
                            {userDatabases.map((dbConn: any) => (
                              <button
                                key={dbConn.id}
                                type="button"
                                onClick={() => {
                                  setAttachedDatabase(dbConn);
                                  setIsDbDropdownOpen(false);
                                  setAttachedDocument(null);
                                }}
                                className={`w-full flex items-center justify-between px-2.5 py-2 rounded-xl text-left text-xs transition-colors ${
                                  attachedDatabase?.id === dbConn.id
                                    ? "bg-indigo-50 dark:bg-indigo-950/40 text-indigo-700 dark:text-indigo-400 font-semibold"
                                    : "hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300"
                                }`}
                              >
                                <span>{dbConn.name}</span>
                                {attachedDatabase?.id === dbConn.id && (
                                  <span className="text-indigo-600 dark:text-indigo-400 font-bold">
                                    ✓
                                  </span>
                                )}
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Right Group: Cancel, Mic, Send */}
                  <div className="flex items-center gap-2">
                    {/* Cancel Button */}
                    {isRecording && (
                      <button
                        onClick={cancelRecording}
                        type="button"
                        className="flex items-center justify-center w-8 h-8 rounded-lg text-slate-450 hover:text-red-500 hover:bg-slate-100 dark:text-slate-500 dark:hover:text-red-400 dark:hover:bg-slate-800 transition-colors"
                        title="Cancel recording"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    )}

                    {/* Mic Button */}
                    <button
                      onClick={isRecording ? stopRecording : startRecording}
                      disabled={isTranscribing}
                      type="button"
                      className={`flex items-center justify-center w-8 h-8 rounded-lg transition-all duration-200 ${
                        isRecording
                          ? "bg-red-500 hover:bg-red-600 text-white animate-pulse"
                          : isTranscribing
                            ? "bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-500 cursor-not-allowed"
                            : "text-slate-400 hover:text-indigo-600 dark:text-slate-550 dark:hover:text-indigo-400 hover:bg-slate-100 dark:hover:bg-slate-800"
                      }`}
                      title={
                        isRecording
                          ? "Stop recording"
                          : isTranscribing
                            ? "Transcribing..."
                            : "Record voice query"
                      }
                    >
                      {isTranscribing ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : isRecording ? (
                        <div className="w-2.5 h-2.5 bg-white rounded-sm" />
                      ) : (
                        <Mic className="w-4 h-4" />
                      )}
                    </button>

                    {/* Send or Cancel Button */}
                    {isSending || isLoading || isStreaming ? (
                      <button
                        onClick={handleCancel}
                        type="button"
                        className="flex items-center justify-center w-8 h-8 rounded-lg bg-red-600 hover:bg-red-700 dark:bg-red-550 dark:hover:bg-red-600 text-white transition-colors"
                        title="Cancel message generation"
                      >
                        <Square className="w-3.5 h-3.5 fill-current" />
                      </button>
                    ) : (
                      <button
                        onClick={() => handleSend()}
                        disabled={isSendDisabled}
                        type="button"
                        className="flex items-center justify-center w-8 h-8 rounded-lg bg-indigo-700 dark:bg-indigo-500 hover:bg-indigo-600 dark:hover:bg-indigo-400 text-white disabled:opacity-50 transition-colors"
                        title="Send message"
                      >
                        <SendHorizontal className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {/* Error display */}
              {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
            </div>
          </div>
        </div>
      </div>
      <UserSchemaModal
        isOpen={isSchemaModalOpen}
        onClose={() => setIsSchemaModalOpen(false)}
        connectionId={attachedDatabase?.id || null}
        connectionName={attachedDatabase?.name || ""}
      />
    </div>
  );
};

export default ChatPage;
