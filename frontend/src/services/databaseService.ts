import api from "./api";

export interface DatabaseConnectionResponse {
  id: string;
  tenant_id: string;
  name: string;
  engine: string;
  host: string;
  port: number;
  database_name: string;
  username: string;
  ssl_mode: string | null;
  status: string;
  last_error: string | null;
  last_synced_at: string | null;
  created_at: string;
  updated_at: string;
  table_count: number;
}

export interface DatabaseConnectionCreatePayload {
  name: string;
  engine: string;
  host: string;
  port: number;
  database_name: string;
  username: string;
  password?: string;
  ssl_mode?: string;
}

export interface DatabaseAccessPolicyResponse {
  id: string;
  connection_id: string;
  role_id: string;
  role_name: string;
  granted_via: string;
  inherited_from_role_id: string | null;
  inherited_from_role_name: string | null;
  granted_via_department_id: string | null;
  granted_via_department_name: string | null;
  table_name: string | null;
  columns?: string[] | null;
  created_at: string;
}

export interface DatabaseAccessPolicyCreatePayload {
  role_id?: string;
  role_ids?: string[] | null;
  department_id?: string;
  department_ids?: string[] | null;
  table_name?: string | null;
  table_names?: string[] | null;
  columns?: string[] | null;
}

export const databaseService = {
  getDatabases: async (): Promise<DatabaseConnectionResponse[]> => {
    const response = await api.get("/api/v1/databases");
    return response.data;
  },

  getAuthorizedDatabases: async (): Promise<DatabaseConnectionResponse[]> => {
    const response = await api.get("/api/v1/databases/authorized");
    return response.data;
  },

  getDatabase: async (id: string): Promise<DatabaseConnectionResponse> => {
    const response = await api.get(`/api/v1/databases/${id}`);
    return response.data;
  },

  getDatabaseSchema: async (id: string): Promise<any> => {
    const response = await api.get(`/api/v1/databases/${id}/schema`);
    return response.data;
  },

  getUserSchema: async (id: string): Promise<any> => {
    const response = await api.get(`/api/v1/databases/${id}/user-schema`);
    return response.data;
  },


  createDatabase: async (
    data: DatabaseConnectionCreatePayload,
  ): Promise<DatabaseConnectionResponse> => {
    const response = await api.post("/api/v1/databases", data);
    return response.data;
  },

  updateDatabase: async (
    id: string,
    data: Partial<DatabaseConnectionCreatePayload>,
  ): Promise<DatabaseConnectionResponse> => {
    const response = await api.put(`/api/v1/databases/${id}`, data);
    return response.data;
  },

  deleteDatabase: async (id: string): Promise<void> => {
    await api.delete(`/api/v1/databases/${id}`);
  },

  testConnection: async (
    data: Omit<DatabaseConnectionCreatePayload, "name">,
  ): Promise<{ status: string; message: string }> => {
    const response = await api.post("/api/v1/databases/test-connection", data);
    return response.data;
  },

  refreshSchema: async (id: string): Promise<DatabaseConnectionResponse> => {
    const response = await api.post(`/api/v1/databases/${id}/refresh`);
    return response.data;
  },

  listAccessPolicies: async (
    id: string,
  ): Promise<DatabaseAccessPolicyResponse[]> => {
    const response = await api.get(`/api/v1/databases/${id}/access`);
    return response.data;
  },

  grantAccess: async (
    id: string,
    data: DatabaseAccessPolicyCreatePayload,
  ): Promise<DatabaseAccessPolicyResponse[]> => {
    const response = await api.post(`/api/v1/databases/${id}/access`, data);
    return response.data;
  },

  revokeAccess: async (id: string, policyId: string): Promise<void> => {
    await api.delete(`/api/v1/databases/${id}/access/${policyId}`);
  },
};
