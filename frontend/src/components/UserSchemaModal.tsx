import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { X, ChevronDown, ChevronRight, Database, Loader2, AlertCircle, Table } from "lucide-react";
import { databaseService } from "../services/databaseService";

interface UserSchemaModalProps {
  isOpen: boolean;
  onClose: () => void;
  connectionId: string | null;
  connectionName: string;
}

export const UserSchemaModal: React.FC<UserSchemaModalProps> = ({
  isOpen,
  onClose,
  connectionId,
  connectionName,
}) => {
  const [expandedTables, setExpandedTables] = useState<Record<string, boolean>>({});

  const {
    data: schemaData,
    isLoading,
    isError,
    error,
  } = useQuery({
    queryKey: ["user-database-schema", connectionId],
    queryFn: () => databaseService.getUserSchema(connectionId!),
    enabled: isOpen && !!connectionId,
    staleTime: 60000,
  });

  if (!isOpen) return null;

  const toggleTable = (tableName: string) => {
    setExpandedTables((prev) => ({
      ...prev,
      [tableName]: !prev[tableName],
    }));
  };

  const tables = schemaData?.tables || [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-fadeIn">
      <div
        className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-2xl w-full max-w-2xl max-h-[85vh] flex flex-col overflow-hidden text-slate-800 dark:text-slate-100"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-indigo-50 dark:bg-indigo-950/50 text-indigo-600 dark:text-indigo-400">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-base text-slate-900 dark:text-slate-100">
                {connectionName || "Database Schema"}
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Permitted Tables & Columns
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer"
            title="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 overflow-y-auto flex-1">
          {isLoading && (
            <div className="flex flex-col items-center justify-center py-16 text-slate-500 dark:text-slate-400 gap-3">
              <Loader2 className="w-7 h-7 animate-spin text-indigo-600 dark:text-indigo-400" />
              <span className="text-sm font-medium">Loading schema...</span>
            </div>
          )}

          {isError && (
            <div className="flex flex-col items-center justify-center py-12 text-rose-500 gap-2 text-center">
              <AlertCircle className="w-8 h-8" />
              <p className="font-semibold text-sm">Failed to load schema</p>
              <p className="text-xs text-slate-400 max-w-xs">
                {error instanceof Error ? error.message : "An unexpected error occurred while fetching database schema."}
              </p>
            </div>
          )}

          {!isLoading && !isError && tables.length === 0 && (
            <div className="flex flex-col items-center justify-center py-16 text-slate-400 dark:text-slate-500 text-center gap-2">
              <Table className="w-10 h-10 stroke-1 opacity-50" />
              <p className="text-sm font-medium text-slate-600 dark:text-slate-400">
                You don't have access to any tables in this database
              </p>
            </div>
          )}

          {!isLoading && !isError && tables.length > 0 && (
            <div className="flex flex-col gap-2.5">
              {tables.map((table: any) => {
                const tableName = table.name || table.table_name;
                const isExpanded = !!expandedTables[tableName];
                const columns = table.columns || [];

                return (
                  <div
                    key={tableName}
                    className="border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden bg-slate-50/40 dark:bg-slate-850/40 transition-colors"
                  >
                    <button
                      onClick={() => toggleTable(tableName)}
                      className="w-full flex items-center justify-between px-4 py-3 text-xs font-semibold text-slate-800 dark:text-slate-200 bg-slate-100/70 dark:bg-slate-800/60 hover:bg-slate-200/60 dark:hover:bg-slate-800 transition-colors text-left select-none cursor-pointer"
                    >
                      <div className="flex items-center gap-2.5">
                        <Table className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
                        <span className="font-mono font-medium text-sm">{tableName}</span>
                        <span className="text-[11px] font-normal text-slate-400 dark:text-slate-500 bg-slate-200/60 dark:bg-slate-700/50 px-2 py-0.5 rounded-full">
                          {columns.length} {columns.length === 1 ? "column" : "columns"}
                        </span>
                      </div>
                      {isExpanded ? (
                        <ChevronDown className="w-4 h-4 text-slate-400" />
                      ) : (
                        <ChevronRight className="w-4 h-4 text-slate-400" />
                      )}
                    </button>

                    {isExpanded && (
                      <div className="px-4 py-2 bg-white dark:bg-slate-900 border-t border-slate-200 dark:border-slate-800 divide-y divide-slate-100 dark:divide-slate-800/60">
                        {columns.length === 0 ? (
                          <div className="py-2 text-xs text-slate-400 italic">No permitted columns</div>
                        ) : (
                          columns.map((col: any) => {
                            const colName = col.name || col.column_name;
                            const colType = col.type || col.data_type || "unknown";

                            return (
                              <div
                                key={colName}
                                className="flex items-center justify-between py-2 text-xs"
                              >
                                <span className="font-mono text-slate-700 dark:text-slate-300 font-medium">
                                  {colName}
                                </span>
                                <span className="font-mono text-[11px] text-slate-400 dark:text-slate-500">
                                  {colType}
                                </span>
                              </div>
                            );
                          })
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
