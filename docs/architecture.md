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

## Security Architecture

### Authentication & Access Control

1. **API Key Authentication**: Required on every endpoint via `X-API-Key` header. The app refuses to start without `API_KEY` and `SECRET_KEY` set in production mode.
2. **WebSocket Authentication**: Token-based via query parameter (`?token=`), validated before the connection is accepted. Max 5 concurrent WebSocket connections to prevent resource exhaustion.
3. **Startup Validation**: The application checks for required secrets at boot and fails fast if they're missing.

### Network Security

4. **TLS/HTTPS**: Self-signed certificates for raw IP deploys (with documented upgrade path to Let's Encrypt). HTTP automatically redirects to HTTPS.
5. **Rate Limiting**: Nginx-level rate limiting across three zones:
   - General API: 30 requests/sec (burst 50)
   - Uploads: 5 requests/sec (burst 10)
   - WebSocket: 2 connections/sec (burst 5)
6. **Security Headers**: HSTS, Content-Security-Policy, Permissions-Policy, X-Frame-Options, X-Content-Type-Options, Referrer-Policy. Server tokens disabled.
7. **CORS**: Explicit method and header allowlists (no wildcards).

### Container Hardening

8. **Internal-only Services**: Only nginx is exposed externally (ports 80/443). PostgreSQL, Redis, API, and workers are on an internal Docker network with no direct internet access.
9. **Network Segmentation**: Two Docker networks — `frontend` (bridge, nginx only) and `backend` (internal, all services). Backend network blocks outbound internet.
10. **Resource Limits**: CPU and memory limits on every container to prevent runaway processes.
11. **Read-only Filesystems**: Nginx and dashboard containers run with `read_only: true` and `tmpfs` mounts for temp data.
12. **Privilege Restrictions**: All containers run with `no-new-privileges` security option.
13. **Redis Authentication**: Password-protected, not exposed outside the Docker network.

### Application-Level Security

14. **Upload Validation**: Magic-byte signature checking (JPEG, PNG, BMP, WebP), 10 MB size limit, minimum size enforcement. Applied on all upload endpoints.
15. **Docs Disabled in Production**: `/docs`, `/redoc`, and `/openapi.json` are blocked with 404 in production. Error details are hidden.
16. **Soft Delete**: Identities are soft-deleted by default (GDPR right to rectification).
17. **Hard Delete**: Available for GDPR right to erasure.
18. **No PII in Logs**: Structured logging without sensitive data. Log rotation on all containers (10 MB max, 5 files).

### VPS Hardening (deploy/ scripts)

19. **SSH**: Key-only authentication, custom port, root login disabled, 3 max auth tries, 5-minute idle timeout.
20. **Firewall (UFW)**: Default deny incoming. Only SSH, HTTP (redirect), and HTTPS allowed. Common botnet ports explicitly blocked (Telnet, TR-069, ADB).
21. **Fail2ban**: SSH brute force protection (24h ban after 3 attempts), nginx rate limit abuse detection (1h ban).
22. **CrowdSec**: Community IPS with shared global botnet blocklists. Nginx, SSH, and Linux collections installed.
23. **Automatic Updates**: Unattended-upgrades for security patches, weekly auto-clean.
24. **Kernel Hardening**: SYN cookies, ICMP redirect blocking, reverse path filtering, martian packet logging, source routing disabled.

## Deployment Topology

Single `docker compose up` command deploys 6 services across 2 isolated networks:

```
                         Internet
                            │
                     ┌──────┴──────┐
                     │   Nginx     │ ← Ports 80/443 (only public-facing service)
                     │  (frontend) │    TLS termination, rate limiting
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │      backend network
    ┌─────────┴──┐  ┌───────┴───────┐  ┌──┴──────────┐  (internal, no internet)
    │  Dashboard  │  │  FastAPI API   │  │   Celery    │
    │  React SPA  │  │  Trust Pipeline│  │   Workers   │
    └─────────────┘  └───────┬───────┘  └──┬──────────┘
                             │             │
                    ┌────────┴─────────────┴──────┐
                    │                              │
              ┌─────┴──────┐            ┌─────────┴──┐
              │ PostgreSQL  │            │   Redis     │
              │ + pgvector  │            │ (auth req.) │
              └─────────────┘            └─────────────┘
```

All services use health checks, restart policies, resource limits, and log rotation for production reliability.
