const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface HealthResponse {
  status: string;
  indexed_documents_count?: number;
  documents_indexed?: number;
  total_chunks?: number;
  faiss_index_ready?: boolean;
}

export interface UploadResponse {
  document_id: string;
  filename: string;
  chunks_created?: number;
  status: string;
  message: string;
  job_id: string;
}

export interface StatusResponse {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  total_chunks: number;
  processed_chunks: number;
  progress?: number;
  doc_id?: string;
  filename?: string;
  file_type?: string;
  file_size?: number;
  message?: string;
  error?: string;
}

export interface AskRequest {
  question: string;
  top_k?: number;
  document_id?: string;
  doc_ids?: string[];
  max_context_chars?: number;
  filter_date_range?: [string, string];
}

export interface RetrievedChunk {
  position?: number;
  doc_id: string;
  text: string;
  headers?: Record<string, string | null>;
  page_num?: number;
  chunk_index?: number;
  block_type?: string;
  block_metadata?: Record<string, any>;
  confidence_score?: number;
  source_label?: string;
}

export interface AskResponse {
  answer: string;
  sources?: string[];
  source_chunks?: RetrievedChunk[];
  confidence?: number;
  confidence_score?: number;
  doc_ids_used?: string[];
  chunks_used?: any[];
  context_chars_used?: number;
  context_budget_remaining?: number;
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed with status ${res.status}`);
  }
  return res.json();
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Upload failed with status ${res.status}`);
  }
  return res.json();
}

export async function pollStatus(jobId: string): Promise<StatusResponse> {
  const res = await fetch(`${API_BASE_URL}/status/${jobId}`);
  if (!res.ok) {
    throw new Error(`Status check failed with status ${res.status}`);
  }
  return res.json();
}

export async function askQuestion(req: AskRequest): Promise<AskResponse> {
  const res = await fetch(`${API_BASE_URL}/ask`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Ask request failed with status ${res.status}`);
  }
  return res.json();
}
