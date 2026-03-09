# Architecture Notes

## High-level overview

The system has 6 services running behind nginx, all in Docker:

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

Only nginx is exposed to the internet. Everything else talks over an internal docker network.

## Trust Pipeline

This is the main idea. Every face goes through a sequence of stages and gets a combined trust score at the end:

```
Frame → Detection → Alignment → Liveness → Embedding → Recognition → Attributes
                                                                         │
                                                              Trust Score ◄
```

### Stages

1. **Detection (RetinaFace)** -- 5-point landmark detection + bounding boxes. Pretty standard.

2. **Alignment** -- Similarity transform to 112x112 crop using the 5 landmarks. This is important for the embeddings to work well.

3. **Liveness check** -- This is the part I'm most unsure about. It's all heuristic, no neural network:
   - FFT frequency analysis (catches screen/print patterns)
   - YCrCb color distribution (printed photos look different)
   - Laplacian + Sobel texture check (live skin has more micro-texture)
   - Autocorrelation moiré detection (screen capture artifacts)

   Weights: `0.30*freq + 0.20*color + 0.30*texture + 0.20*moire`

   Works okay for obvious attacks but a proper CNN would be better for serious use (see TODO in antispoof.py).

4. **Embedding (ArcFace buffalo_l)** -- 512-d L2-normalized vector. InsightFace's pretrained model.

5. **Recognition** -- Cosine similarity against the enrolled gallery, then Platt scaling to get a calibrated probability:
   ```
   P(match) = 1 / (1 + exp(A*(1-sim) + B))
   # A = -15.0, B = 6.5 -- fitted manually, should be done properly on a validation set
   ```

6. **Attributes** -- Age, gender from InsightFace's genderage module. Emotion is heuristic (not great tbh).

### Trust score formula

```
trust = 0.20*detection_conf + 0.40*liveness_conf + 0.40*recognition_conf
if spoof_detected:
    trust *= 0.3  # penalize hard
```

## Database

```
Identity (1) ──── (N) FaceEmbedding
    │
    └── (N) RecognitionEvent
    
Cluster (1) ──── (N) ClusterMember
```

Embeddings are stored as `Vector(512)` in pgvector with an IVFFlat index (cosine distance). This gives pretty fast nearest-neighbor search and keeps everything in Postgres instead of needing a separate vector DB.

Recognition events are append-only -- no updates or deletes in the app code. Useful for audit trails.

## API endpoints

REST + one WebSocket for the live stream:

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/api/v1/identities` | CRUD | Manage enrolled people |
| `/api/v1/recognize` | POST | Recognize a face from an image upload |
| `/api/v1/streams/webcam` | WS | Live video from browser webcam |
| `/api/v1/forensics/*` | POST | Detailed liveness analysis |
| `/api/v1/clusters/*` | GET/POST | Run/view HDBSCAN clustering |
| `/api/v1/analytics/*` | GET | Stats and demographics |
| `/api/v1/events` | GET | Query the event log |
| `/api/v1/health` | GET | Health check |
| `/api/v1/ready` | GET | Readiness check |

## Background tasks (Celery)

- Batch recognition for queued images
- Re-clustering every 6 hours (HDBSCAN on all embeddings)
- Event cleanup (GDPR retention)
- Gallery pre-loading into memory (hourly)

## Security

### Auth

API key required on every request (`X-API-Key` header). The app won't start without `API_KEY` and `SECRET_KEY` set. WebSocket auth is via query param token, validated before the connection opens. Max 5 concurrent WS connections.

### Network

- Nginx does TLS termination (self-signed certs for now, easy to swap to Let's Encrypt)
- Rate limiting at nginx level: 30 req/s general, 5 req/s uploads, 2 conn/s websocket
- Standard security headers (HSTS, CSP, X-Frame-Options, etc)
- CORS with explicit allowlists, no wildcards

### Docker/infra

- Only nginx exposes ports. Postgres, Redis, API, workers are all internal.
- Two docker networks: frontend (nginx) and backend (everything). Nginx bridges both.
- CPU/memory limits on all containers
- Nginx and dashboard containers use read-only filesystems
- `no-new-privileges` on all containers
- Redis is password-protected

### App-level

- Upload validation checks magic bytes (not just extension), 10MB max
- Swagger/redoc disabled in production
- Soft delete by default (GDPR), with hard delete option
- No PII in logs, log rotation configured (10MB, 5 files)

### VPS hardening scripts (deploy/)

The scripts in `deploy/` are for hardening a fresh VPS. In order:
1. SSH hardening (key-only, custom port, no root)
2. UFW firewall (default deny, only SSH + HTTP/HTTPS open, botnet ports blocked)
3. Fail2ban (SSH brute force bans, nginx rate limit abuse)
4. CrowdSec (community IP blocklists)
5. Unattended security upgrades
6. Kernel hardening (SYN cookies, ICMP redirect blocking, reverse path filtering)

## Deployment

`docker compose up` brings up everything. Two networks keep things isolated:

```
                         Internet
                            │
                     ┌──────┴──────┐
                     │   Nginx     │  ports 80/443
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │      backend network
    ┌─────────┴──┐  ┌───────┴───────┐  ┌──┴──────────┐
    │  Dashboard  │  │  FastAPI API   │  │   Celery    │
    │  React SPA  │  │               │  │   Workers   │
    └─────────────┘  └───────┬───────┘  └──┬──────────┘
                             │             │
                    ┌────────┴─────────────┴──────┐
                    │                              │
              ┌─────┴──────┐            ┌─────────┴──┐
              │ PostgreSQL  │            │   Redis     │
              │ + pgvector  │            │             │
              └─────────────┘            └─────────────┘
```

The backend network isn't set to `internal` because containers need to download ML model weights on first startup. Might change this later and pre-bake models into the image instead.