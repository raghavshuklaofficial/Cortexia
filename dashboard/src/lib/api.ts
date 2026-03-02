/**
 * CORTEXIA API client — typed HTTP methods for backend communication.
 */

const API_BASE = "/api/v1";

interface RequestOptions {
  headers?: Record<string, string>;
  signal?: AbortSignal;
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  options?: RequestOptions
): Promise<T> {
  const headers: Record<string, string> = {
    ...(options?.headers || {}),
  };

  // Attach API key for authenticated requests
  const apiKey =
    localStorage.getItem("cortexia_api_key") ||
    (import.meta as unknown as { env: Record<string, string> }).env?.VITE_API_KEY;
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }

  if (body && !(body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: body instanceof FormData ? body : body ? JSON.stringify(body) : undefined,
    signal: options?.signal,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new ApiError(response.status, error.detail || "Request failed");
  }

  return response.json();
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

export const api = {
  get: <T>(path: string, options?: RequestOptions) =>
    request<T>("GET", path, undefined, options),

  post: <T>(path: string, body?: unknown, options?: RequestOptions) =>
    request<T>("POST", path, body, options),

  put: <T>(path: string, body?: unknown, options?: RequestOptions) =>
    request<T>("PUT", path, body, options),

  delete: <T>(path: string, options?: RequestOptions) =>
    request<T>("DELETE", path, undefined, options),

  upload: <T>(path: string, formData: FormData, options?: RequestOptions) =>
    request<T>("POST", path, formData, options),
};

// ─── Typed API Methods ─────────────────────────────────────────────────

export interface Identity {
  id: number;
  name: string;
  metadata: Record<string, unknown> | null;
  face_count: number;
  privacy_score: number;
  is_active: boolean;
  last_seen: string | null;
  created_at: string;
  updated_at: string;
}

export interface PaginationMeta {
  total: number;
  skip: number;
  limit: number;
  has_more: boolean;
}

export interface FaceAnalysis {
  bbox: { x1: number; y1: number; x2: number; y2: number };
  detection_confidence: number;
  trust_score: number;
  processing_time_ms: number;
  track_id: number | null;
  liveness: {
    verdict: string;
    confidence: number;
    method: string;
  } | null;
  recognition: {
    identity_id: number;
    identity_name: string;
    distance: number;
    confidence: number;
    is_known: boolean;
  } | null;
  attributes: {
    age: number | null;
    gender: string | null;
    gender_confidence: number | null;
    emotion: string | null;
    emotion_confidence: number | null;
  } | null;
}

export interface RecognitionResult {
  faces: FaceAnalysis[];
  face_count: number;
  known_count: number;
  spoof_count: number;
  total_processing_time_ms: number;
  frame_dimensions: { width: number; height: number };
}

export interface RecognitionEvent {
  id: number;
  identity_id: number | null;
  identity_name: string | null;
  timestamp: string;
  confidence: number;
  trust_score: number;
  is_spoof: boolean;
  is_known: boolean;
  source: string;
  attributes: Record<string, unknown> | null;
  bounding_box: Record<string, number> | null;
}

export interface OverviewStats {
  total_identities: number;
  total_events: number;
  known_events: number;
  unknown_events: number;
  spoof_events: number;
  unknown_ratio: number;
  avg_trust_score: number;
  avg_recognition_confidence: number;
}

export interface TimelinePoint {
  period: string;
  total: number;
  known: number;
  spoofs: number;
}

export interface DemographicsData {
  age_distribution: Record<string, number>;
  gender_distribution: Record<string, number>;
  emotion_distribution: Record<string, number>;
}

export interface ClusterInfo {
  id: number;
  member_count: number;
  created_at: string;
}

export interface HealthStatus {
  status: string;
  version: string;
  uptime_seconds: number;
}

export interface ReadinessStatus {
  status: string;
  database: string;
  redis: string;
  models_loaded: boolean;
}

export interface ApiResponse {
  success: boolean;
  message: string;
  data: Record<string, unknown> | null;
}

export interface ForensicAnalysisResult {
  face_detected: boolean;
  liveness: {
    verdict: string;
    confidence: number;
    method: string;
  } | null;
  face_quality_score: number;
  attributes: {
    age: number | null;
    gender: string | null;
    gender_confidence: number | null;
    emotion: string | null;
    emotion_confidence: number | null;
  } | null;
  trust_score: number;
  processing_time_ms: number;
}

// ─── Domain-specific API calls ──────────────────────────────────────────

export const cortexiaApi = {
  // Health
  health: () => api.get<HealthStatus>("/health"),
  ready: () => api.get<ReadinessStatus>("/ready"),

  // Identities
  listIdentities: (page = 1, size = 20, search?: string) => {
    const skip = (page - 1) * size;
    const params = new URLSearchParams({ skip: String(skip), limit: String(size) });
    if (search) params.set("search", search);
    return api.get<{ identities: Identity[]; pagination: PaginationMeta }>(
      `/identities?${params}`
    );
  },
  getIdentity: (id: number) =>
    api.get<ApiResponse>(`/identities/${id}`),
  createIdentity: (name: string, file?: File, metadata?: Record<string, unknown>) => {
    const form = new FormData();
    form.append("name", name);
    if (file) form.append("images", file);
    if (metadata) form.append("metadata", JSON.stringify(metadata));
    return api.upload<ApiResponse>("/identities", form);
  },
  deleteIdentity: (id: number, hard = false) =>
    api.delete<ApiResponse>(`/identities/${id}?hard=${hard}`),
  addFaces: (id: number, files: FileList) => {
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("images", f));
    return api.upload<ApiResponse>(
      `/identities/${id}/faces`,
      form
    );
  },

  // Recognition
  recognize: (file: File) => {
    const form = new FormData();
    form.append("image", file);
    return api.upload<RecognitionResult>("/recognize", form);
  },

  // Events
  listEvents: (params?: {
    identity_id?: number;
    source?: string;
    is_known?: boolean;
    is_spoof?: boolean;
    page?: number;
    size?: number;
  }) => {
    const search = new URLSearchParams();
    if (params) {
      const { page, size, ...rest } = params;
      // Convert page/size to skip/limit
      if (page !== undefined && size !== undefined) {
        search.set("skip", String((page - 1) * size));
        search.set("limit", String(size));
      } else if (size !== undefined) {
        search.set("limit", String(size));
      }
      Object.entries(rest).forEach(([k, v]) => {
        if (v !== undefined) search.set(k, String(v));
      });
    }
    return api.get<{ events: RecognitionEvent[]; pagination: PaginationMeta }>(
      `/events?${search}`
    );
  },

  // Analytics
  overview: () => api.get<OverviewStats>("/analytics/overview"),
  timeline: (days = 7, interval = "hour") =>
    api.get<TimelinePoint[]>(
      `/analytics/timeline?days=${days}&interval=${interval}`
    ),
  demographics: () =>
    api.get<DemographicsData>("/analytics/demographics"),

  // Clusters
  discoverClusters: (minSize = 5) =>
    api.post<ApiResponse>(`/clusters/discover?min_cluster_size=${minSize}`),
  listClusters: () =>
    api.get<ApiResponse>("/clusters"),

  // Forensics
  analyzeLiveness: (file: File) => {
    const form = new FormData();
    form.append("image", file);
    return api.upload<ForensicAnalysisResult>("/forensics/liveness", form);
  },
  forensicAnalyze: (file: File) => {
    const form = new FormData();
    form.append("image", file);
    return api.upload<ForensicAnalysisResult>("/forensics/analyze", form);
  },
};
