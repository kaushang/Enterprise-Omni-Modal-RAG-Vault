import React, { useState, useEffect, useRef } from "react";
import { Folder, Plus, MoreVertical, Check, Loader2 } from "lucide-react";
import type { CollectionResponse } from "../../types/document";

interface CollectionsSidebarProps {
  collections: CollectionResponse[];
  uncategorizedCount: number;
  totalDocuments: number;
  selectedFilter: {
    type: "all" | "uncategorized" | "collection";
    collectionId?: string;
  };
  onFilterChange: (filter: {
    type: "all" | "uncategorized" | "collection";
    collectionId?: string;
  }) => void;
  onCreateCollection: (name: string, description?: string) => Promise<void>;
  onRenameCollection: (collectionId: string, name: string) => Promise<void>;
  onDeleteCollection: (collectionId: string) => Promise<void>;
  isAdmin: boolean;
}

export const CollectionsSidebar: React.FC<CollectionsSidebarProps> = ({
  collections,
  uncategorizedCount,
  totalDocuments,
  selectedFilter,
  onFilterChange,
  onCreateCollection,
  onRenameCollection,
  onDeleteCollection,
  isAdmin,
}) => {
  // Creation state
  const [isCreating, setIsCreating] = useState(false);
  const [newItemName, setNewItemName] = useState("");
  const [creationError, setCreationError] = useState<string | null>(null);
  const [isSavingCreation, setIsSavingCreation] = useState(false);

  // Renaming state
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [isSavingRename, setIsSavingRename] = useState(false);
  const [renameError, setRenameError] = useState<string | null>(null);

  // Deletion state
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Dropdown menu state
  const [activeMenuId, setActiveMenuId] = useState<string | null>(null);

  // Refs for inputs and dropdowns
  const createInputRef = useRef<HTMLInputElement>(null);
  const renameInputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Focus create input
  useEffect(() => {
    if (isCreating) {
      createInputRef.current?.focus();
      setNewItemName("");
      setCreationError(null);
    }
  }, [isCreating]);

  // Focus rename input
  useEffect(() => {
    if (renamingId) {
      renameInputRef.current?.focus();
      setRenameError(null);
    }
  }, [renamingId]);

  // Close dropdown on click outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setActiveMenuId(null);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const handleCreateSubmit = async () => {
    const trimmed = newItemName.trim();
    if (!trimmed) {
      setIsCreating(false);
      return;
    }
    setIsSavingCreation(true);
    setCreationError(null);
    try {
      await onCreateCollection(trimmed);
      setIsCreating(false);
      setNewItemName("");
    } catch (err: any) {
      setCreationError(
        err.response?.data?.detail || err.message || "Failed to create",
      );
    } finally {
      setIsSavingCreation(false);
    }
  };

  const handleRenameSubmit = async (collectionId: string) => {
    const trimmed = renameValue.trim();
    if (!trimmed) {
      setRenamingId(null);
      return;
    }
    setIsSavingRename(true);
    setRenameError(null);
    try {
      await onRenameCollection(collectionId, trimmed);
      setRenamingId(null);
    } catch (err: any) {
      setRenameError(
        err.response?.data?.detail || err.message || "Failed to rename",
      );
    } finally {
      setIsSavingRename(false);
    }
  };

  const handleDeleteSubmit = async (collectionId: string) => {
    setIsDeleting(true);
    try {
      await onDeleteCollection(collectionId);
      setDeleteConfirmId(null);
      // If we deleted the currently selected collection, switch back to 'all'
      if (
        selectedFilter.type === "collection" &&
        selectedFilter.collectionId === collectionId
      ) {
        onFilterChange({ type: "all" });
      }
    } catch (err) {
      // Error handled by parent hook
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="w-56 shrink-0 flex flex-col border-slate-200 dark:border-slate-800 select-none sticky top-0 self-start">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 px-2">
        <span className="text-sm font-semibold text-slate-900 dark:text-slate-200">
          Collections
        </span>
        {isAdmin && (
          <button
            onClick={() => setIsCreating(true)}
            className="p-1 rounded-md text-slate-900 dark:text-slate-200 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            title="Create new collection"
          >
            <Plus className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Fixed Navigation Items */}
      <div className="space-y-1">
        <button
          onClick={() => onFilterChange({ type: "all" })}
          className={`w-full flex items-center p-2 text-sm font-medium rounded-xl transition-all gap-2 ${
            selectedFilter.type === "all"
              ? "bg-indigo-50 dark:bg-indigo-950/50 text-indigo-700 dark:text-indigo-400"
              : "text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800/50 hover:text-slate-900 dark:hover:text-white"
          }`}
        >
          <span>All Documents</span>
          <span className="inline-flex items-center justify-center min-w-4 h-4 px-1 text-[10px] font-semibold rounded-full bg-indigo-200 dark:bg-indigo-800/50 dark:text-slate-300">
            {totalDocuments}
          </span>
        </button>

        <button
          onClick={() => onFilterChange({ type: "uncategorized" })}
          className={`w-full flex items-center p-2 text-sm font-medium rounded-xl transition-all gap-2 ${
            selectedFilter.type === "uncategorized"
              ? "bg-indigo-50 dark:bg-indigo-950/50 text-indigo-700 dark:text-indigo-400"
              : "text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800/50 hover:text-slate-900 dark:hover:text-white"
          }`}
        >
          <span>Uncategorized</span>
          <span className="inline-flex items-center justify-center min-w-4 h-4 px-1 text-[10px] font-semibold rounded-full bg-indigo-200 dark:bg-indigo-800/50 dark:text-slate-300">
            {uncategorizedCount}
          </span>
        </button>
      </div>

      {/* Divider */}
      <div className="border-b border-slate-200 dark:border-slate-800 my-3" />

      {/* Dynamic Collection List */}
      <div className="flex-1 overflow-y-auto space-y-1 max-h-[400px]">
        {collections.map((col) => {
          const isSelected =
            selectedFilter.type === "collection" &&
            selectedFilter.collectionId === col.id;
          const isRenaming = renamingId === col.id;
          const isConfirmingDelete = deleteConfirmId === col.id;

          return (
            <div key={col.id} className="space-y-1">
              {/* Collection Item Row */}
              {!isRenaming && (
                <div
                  className={`group relative flex items-center justify-between rounded-xl cursor-pointer ${
                    isSelected
                      ? "bg-indigo-50 dark:bg-indigo-950/50 text-indigo-700 dark:text-indigo-400 font-semibold"
                      : "text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/40 hover:text-slate-900 dark:hover:text-white"
                  }`}
                >
                  <button
                    onClick={() =>
                      onFilterChange({
                        type: "collection",
                        collectionId: col.id,
                      })
                    }
                    className="flex-1 flex items-center gap-2 p-2 text-sm text-left overflow-hidden min-w-0"
                  >
                    <Folder className="w-4 h-4 shrink-0 text-slate-400 dark:text-slate-500" />
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <span className="truncate" title={col.name}>
                        {col.name}
                      </span>

                      <span className="inline-flex items-center justify-center px-0.5 min-w-4 h-4 text-[10px] font-semibold rounded-full bg-indigo-200 dark:bg-indigo-800/50 dark:text-slate-300">
                        {col.document_count}
                      </span>
                    </div>
                  </button>

                  <div className="flex items-center pr-2 shrink-0 gap-1.5">
                    {/* Admin Actions Button */}
                    {isAdmin && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setActiveMenuId(
                            activeMenuId === col.id ? null : col.id,
                          );
                        }}
                        className="p-1 rounded-full text-slate-500 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-350 group-hover:opacity-100 focus:opacity-100 transition-opacity hover:bg-slate-100 hover:dark:hover:bg-slate-800"
                      >
                        <MoreVertical className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>

                  {/* Actions Dropdown */}
                  {activeMenuId === col.id && (
                    <div
                      ref={dropdownRef}
                      className="absolute right-2 top-8 w-32 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-xl z-30 overflow-hidden py-1"
                    >
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setRenamingId(col.id);
                          setRenameValue(col.name);
                          setActiveMenuId(null);
                        }}
                        className="w-full text-left px-3 py-2 text-xs text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
                      >
                        Rename
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setDeleteConfirmId(col.id);
                          setActiveMenuId(null);
                        }}
                        className="w-full text-left px-3 py-2 text-xs text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                      >
                        Delete
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* Inline Renaming Input */}
              {isRenaming && (
                <div className="px-1.5 py-1">
                  <div className="flex items-center gap-1.5 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg px-2 py-1 focus-within:ring-2 focus-within:ring-indigo-400">
                    <input
                      ref={renameInputRef}
                      type="text"
                      value={renameValue}
                      onChange={(e) => setRenameValue(e.target.value)}
                      disabled={isSavingRename}
                      className="bg-transparent text-xs w-full focus:outline-none text-slate-800 dark:text-slate-200 py-0.5"
                      onKeyDown={(e) => {
                        if (e.key === "Enter") handleRenameSubmit(col.id);
                        if (e.key === "Escape") setRenamingId(null);
                      }}
                      onBlur={() => handleRenameSubmit(col.id)}
                    />
                    {isSavingRename ? (
                      <Loader2 className="w-3.5 h-3.5 text-indigo-500 animate-spin shrink-0" />
                    ) : (
                      <button
                        onMouseDown={(e) => {
                          e.preventDefault(); // prevent blur before click registers
                          handleRenameSubmit(col.id);
                        }}
                        className="text-emerald-600 dark:text-emerald-400 hover:bg-emerald-50 dark:hover:bg-emerald-950/30 p-0.5 rounded"
                      >
                        <Check className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                  {renameError && (
                    <p className="text-[10px] text-red-550 dark:text-red-400 mt-1 pl-1 truncate">
                      {renameError}
                    </p>
                  )}
                </div>
              )}

              {/* Inline Deletion Confirmation */}
              {isConfirmingDelete && (
                <div className="bg-red-50/40 dark:bg-red-950/10 border border-red-200/50 dark:border-red-900/20 rounded-xl p-2.5 mx-1 my-1 text-xs">
                  <p className="text-slate-600 dark:text-slate-400 mb-2 leading-relaxed">
                    Delete this collection? Documents inside this will be{" "}
                    <strong>uncategorized</strong>.
                  </p>
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => handleDeleteSubmit(col.id)}
                      disabled={isDeleting}
                      className="px-2 py-1 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold shrink-0"
                    >
                      {isDeleting ? "Deleting..." : "Delete"}
                    </button>
                    <button
                      onClick={() => setDeleteConfirmId(null)}
                      disabled={isDeleting}
                      className="px-2 py-1 border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg shrink-0"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })}

        {/* Inline Create Input */}
        {isCreating && (
          <div className="px-1.5 py-1">
            <div className="flex items-center gap-1.5 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg px-2 py-1 focus-within:ring-2 focus-within:ring-indigo-400">
              <input
                ref={createInputRef}
                type="text"
                value={newItemName}
                onChange={(e) => setNewItemName(e.target.value)}
                disabled={isSavingCreation}
                placeholder="Folder name..."
                className="bg-transparent text-xs w-full focus:outline-none text-slate-800 dark:text-slate-200 py-0.5"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleCreateSubmit();
                  if (e.key === "Escape") setIsCreating(false);
                }}
                onBlur={handleCreateSubmit}
              />
              {isSavingCreation ? (
                <Loader2 className="w-3.5 h-3.5 text-indigo-500 animate-spin shrink-0" />
              ) : (
                <button
                  onMouseDown={(e) => {
                    e.preventDefault();
                    handleCreateSubmit();
                  }}
                  className="text-emerald-600 dark:text-emerald-400 hover:bg-emerald-50 dark:hover:bg-emerald-950/30 p-0.5 rounded"
                >
                  <Check className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
            {creationError && (
              <p className="text-[10px] text-red-550 dark:text-red-400 mt-1 pl-1 truncate">
                {creationError}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
