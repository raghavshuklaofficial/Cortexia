# CORTEXIA — Architecture & System Design

## Overview

CORTEXIA is a neural face intelligence platform built as a production-grade system. This document details the architectural decisions, system design patterns, and technical rationale behind each component.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     NGINX Edge Gateway                       │
│          (Rate limiting · WebSocket upgrade · TLS)           │
├──────────────┬──────────────────────────┬───────────────────┤
│              │                          │                   │
│   Dashboard  │      FastAPI Server      │   Celery Worker   │
│  (React SPA) │    (Trust Pipeline)      │  (Background)     │
│              │                          │                   │
│  ┌─────────┐ │  ┌──────────────────┐   │  ┌─────────────┐  │
│  │Live Feed│ │  │   /api/v1/*      │   │  │ Batch Recog │  │
│  │Identity │ │  │   /ws/streams    │   │  │ Re-cluster  │  │
│  │Analytics│ │  │   Health/Ready   │   │  │ Cleanup     │  │
│  │Forensics│ │  └────────┬─────────┘   │  │ Gallery     │  │
│  └─────────┘ │           │             │  └──────┬──────┘  │
│              │           │             │         │         │
├──────────────┴───────────┼─────────────┴─────────┼─────────┤
│                          │                       │         │
│  ┌──────────────┐   ┌────┴────────┐   ┌─────────┴───────┐ │
│  │  PostgreSQL   │   │   Redis     │   │  Model Cache    │ │
│  │  + pgvector   │   │  (Celery)   │   │  (InsightFace)  │ │
│  │  IVFFlat idx  │   │  (Cache)    │   │  (~500MB)       │ │
│  └──────────────┘   └─────────────┘   └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Trust Pipeline — The Core Innovation

Every face detected by CORTEXIA passes through a multi-stage **Trust Pipeline** that produces a calibrated trust score. This is the system's central architectural concept.

### Pipeline Stages

```
Frame → Detection → Alignment → Liveness → Embedding → Recognition → Attributes
                                                                         │
                                                              Trust Score ◄
```

1. **Detection (RetinaFace)**: Multi-scale face detection with 5-point landmark localization. Returns bounding boxes and confidence scores.

2. **Alignment (ArcFace Reference)**: Similarity transform to canonical 112×112 crop using 5 landmarks (2 eyes, nose, 2 mouth corners). This normalization is critical for embedding quality.

3. **Liveness (Multi-Spectral Ensemble)**: Four independent anti-spoofing analyses:
   - **Frequency Analysis (FFT)**: Detects screen/print artifacts via high-frequency energy ratios
   - **Color Analysis (YCrCb)**: Measures chroma distribution anomalies
   - **Texture Analysis (Laplacian + Sobel)**: Evaluates surface texture naturalness
   - **Moiré Detection (Autocorrelation)**: Finds periodic patterns from screen capture

   Weighted ensemble: `0.30·freq + 0.20·color + 0.30·texture + 0.20·moiré`

4. **Embedding (ArcFace buffalo_l)**: 512-dimensional L2-normalized feature vector. Uses InsightFace's buffalo_l model which achieves 99.83% on LFW benchmark.

5. **Recognition (Cosine + Platt)**: Compares embedding against enrolled gallery. Raw cosine similarity is calibrated via Platt Scaling:
   ```
   P(match) = 1 / (1 + exp(A·(1-sim) + B))
   where A = -15.0, B = 6.5 (fitted parameters)
   ```

6. **Attributes**: Age, gender (via InsightFace genderage module), and emotion (heuristic feature analysis).

### Trust Score Composition

```
trust = 0.20·detection_conf + 0.40·liveness_conf + 0.40·recognition_conf
if spoof_detected:
    trust *= 0.3  # Heavy penalty
```

## Database Design

### Entity Relationship

```
Identity (1) ──── (N) FaceEmbedding
    │
    └── (N) RecognitionEvent
    
Cluster (1) ──── (N) ClusterMember
```

### pgvector Integration

Face embeddings are stored as `Vector(512)` columns with an IVFFlat index using cosine distance (`vector_cosine_ops`). This enables:
- Sub-millisecond nearest-neighbor search on 100K+ embeddings
- Native PostgreSQL ACID guarantees for embedding storage
- Efficient batch operations via standard SQL

### Immutable Audit Trail

`RecognitionEvent` records are append-only (no UPDATE/DELETE in application code), creating a forensic audit trail of all recognition activity. Events include:
- Timestamp, source, identity match
- Trust score, spoof flag
- Full attributes JSON (age, gender, emotion)

## API Design

RESTful + WebSocket hybrid:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/identities` | CRUD | Identity management |
| `/api/v1/recognize` | POST | Single-image recognition |
| `/api/v1/streams/webcam` | WS | Real-time video analysis |
| `/api/v1/forensics/*` | POST | Deep liveness analysis |
| `/api/v1/clusters/*` | GET/POST | HDBSCAN clustering |
| `/api/v1/analytics/*` | GET | Metrics & demographics |
| `/api/v1/events` | GET | Audit log query |
| `/api/v1/health` | GET | Liveness probe |
| `/api/v1/ready` | GET | Readiness probe |

## Background Processing

Celery workers handle:
- **Batch recognition**: Process queued image batches
- **Re-clustering**: Periodic HDBSCAN on all embeddings (every 6h)
- **Data cleanup**: Remove events past retention period (daily, GDPR)
- **Gallery warming**: Pre-load identity gallery into memory (hourly)

## Security Considerations

1. **API Key Authentication**: Optional in dev, required in production
2. **Rate Limiting**: Nginx-level (30 req/s general, 5 req/s uploads)
3. **Soft Delete**: Identities are soft-deleted by default (GDPR right to rectification)
4. **Hard Delete**: Available for GDPR right to erasure
5. **Privacy Scores**: Per-identity privacy scoring
6. **No PII in Logs**: Structured logging without sensitive data

## Deployment Topology

Single `docker compose up` command deploys:
- 🗄️ PostgreSQL 16 + pgvector extension
- 📦 Redis 7 (Celery broker + cache)
- 🧠 CORTEXIA API (FastAPI + Uvicorn, 2 workers)
- ⚙️ CORTEXIA Worker (Celery + Beat scheduler)
- 🎨 Dashboard (React → nginx static)
- 🌐 Nginx (reverse proxy, rate limiting, WebSocket upgrade)

All services use health checks and restart policies for production reliability.
