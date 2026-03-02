<div align="center">

# 🧠 CORTEXIA

### Neural Face Intelligence Platform

*Production-grade face recognition system with real-time detection, anti-spoofing,*
*identity clustering, and a forensic audit trail — deployed with one command.*

[![CI](https://github.com/raghavshuklaofficial/cortexia/actions/workflows/ci.yml/badge.svg)](https://github.com/raghavshuklaofficial/cortexia/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React_18-61DAFB.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Quick Start](#-quick-start) · [Architecture](#-architecture) · [Features](#-features) · [API](#-api-reference) · [Dashboard](#-dashboard) · [Deployment](#-production-deployment) · [Contributing](#-contributing)

</div>

---

## Why CORTEXIA?

Most face recognition projects are toy scripts. CORTEXIA is what a **production system** looks like:

| Aspect | Typical Project | CORTEXIA |
|--------|----------------|----------|
| **Detection** | dlib HOG (2013) | RetinaFace (2019, SOTA) |
| **Embeddings** | 128-d face_recognition | 512-d ArcFace (99.83% LFW) |
| **Anti-Spoofing** | None | 4-method ensemble (FFT, color, texture, moiré) |
| **Confidence** | Raw distance thresholds | Platt-calibrated probabilities |
| **Clustering** | None | HDBSCAN zero-shot identity discovery |
| **Database** | Pickle file | PostgreSQL + pgvector (vector search) |
| **API** | None | FastAPI with WebSocket streaming |
| **Frontend** | OpenCV window | React + TypeScript dashboard |
| **Deployment** | `python script.py` | Docker Compose (6 services) |
| **Security** | None | TLS, API key auth, rate limiting, hardened containers |
| **Testing** | None | pytest + CI/CD pipeline |
| **Audit Trail** | None | Immutable forensic event log |

## 🏗 Architecture

```
                    ┌──────────────────┐
                    │   NGINX Gateway  │ ← TLS termination, rate limiting, WebSocket upgrade
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────┴──┐  ┌───────┴───────┐  ┌──┴──────────┐
    │  Dashboard  │  │  FastAPI API   │  │ Celery      │
    │  React SPA  │  │  Trust Pipeline│  │ Workers     │
    └─────────────┘  └───────┬───────┘  └──┬──────────┘
                             │             │
                    ┌────────┴─────────────┴──────┐
                    │                              │
              ┌─────┴──────┐            ┌─────────┴──┐
              │ PostgreSQL  │            │   Redis     │
              │ + pgvector  │            │             │
              └─────────────┘            └─────────────┘
```

### Trust Pipeline (Core Innovation)

Every face passes through a six-stage pipeline producing a **calibrated trust score**:

```
Frame → Detection → Alignment → Liveness → Embedding → Recognition → Attributes
         RetinaFace   ArcFace     4-method    512-d       Platt        Age/Gender
         (99.4%)      112×112     ensemble    L2-norm     Scaling      Emotion
```

**Trust Score** = `w₁·detection + w₂·liveness + w₃·recognition` (default 0.20/0.40/0.40, configurable via `TRUST_WEIGHT_*` env vars; penalized 0.3× for spoofs)

## ✨ Features

### 🔍 Face Detection & Recognition
- **RetinaFace** detector with automatic GPU/CPU selection
- **ArcFace buffalo_l** embedder — 512-dimensional, L2-normalized
- Real-time multi-face tracking (SORT-inspired algorithm)
- Gallery management with centroid-based identity matching

### 🛡 Anti-Spoofing (Liveness Detection)
- **Frequency Analysis**: FFT-based screen/print artifact detection
- **Color Analysis**: YCrCb chroma distribution anomaly detection
- **Texture Analysis**: Laplacian variance + Sobel gradient evaluation
- **Moiré Detection**: Autocorrelation-based periodic pattern finder
- Weighted ensemble with configurable thresholds

### 📊 Platt-Calibrated Confidence
Raw cosine similarity is unreliable as a probability. CORTEXIA uses **Platt Scaling** (sigmoid calibration) to convert distances into meaningful confidence scores:
```
P(match) = 1 / (1 + exp(-15.0·(1-sim) + 6.5))
```

### 🔬 Zero-Shot Identity Discovery
HDBSCAN density-based clustering discovers natural identity groupings without specifying cluster count. Runs periodically as a background job to surface unknown recurring faces.

### 📝 Forensic Audit Trail
Every recognition event is logged as an immutable record:
- Timestamp, source, matched identity
- Trust score, spoof flag, face attributes
- Queryable by time range, identity, source, verdict

### 🗄 Vector Database
pgvector extension enables:
- Sub-millisecond nearest-neighbor search on 100K+ embeddings
- IVFFlat indexing with cosine distance
- ACID-compliant face embedding storage

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- `.env` file with required secrets (see below)

### Setup & Run

```bash
git clone https://github.com/raghavshuklaofficial/cortexia.git
cd cortexia
cp .env.example .env
```

Edit `.env` and fill in the required secrets:

```bash
# generate these:
openssl rand -hex 32   # → SECRET_KEY
openssl rand -hex 32   # → API_KEY
openssl rand -hex 24   # → POSTGRES_PASSWORD
openssl rand -hex 16   # → REDIS_PASSWORD
```

For local development, set `APP_ENV=development` in `.env`.

Then start everything:

```bash
docker compose up --build
```

Open **https://localhost** for the dashboard (you'll see a cert warning since the default setup uses self-signed TLS).

### Services Started

| Service | Port | Description |
|---------|------|-------------|
| Nginx | 443 | HTTPS gateway with TLS |
| Nginx | 80 | Redirects to HTTPS |
| API | 8000 (internal) | FastAPI + Trust Pipeline |
| PostgreSQL | 5432 (internal) | Database + pgvector |
| Redis | 6379 (internal) | Celery broker + cache |

> Internal ports are not exposed to the host — all traffic goes through nginx.

### Seed Sample Data

```bash
docker compose exec api python scripts/seed_data.py
```

### CLI Usage

```bash
# Serve the API
cortexia serve --port 8000

# Enroll a face
cortexia enroll --name "Alice" --image photos/alice.jpg

# Recognize faces in an image
cortexia recognize --image test.jpg

# System information
cortexia info
```

## 📡 API Reference

All API endpoints require the `X-API-Key` header in production.

### Recognition

```bash
curl -X POST https://localhost/api/v1/recognize \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "image=@photo.jpg"
```

Response:
```json
{
  "status": "success",
  "data": {
    "faces": [
      {
        "bbox": {"x1": 120, "y1": 80, "x2": 280, "y2": 300},
        "detection_confidence": 0.98,
        "trust_score": 0.87,
        "liveness": {
          "verdict": "LIVE",
          "confidence": 0.94,
          "scores": {
            "frequency": 0.91,
            "color": 0.88,
            "texture": 0.96,
            "moire": 0.92
          }
        },
        "match": {
          "identity_id": "a1b2c3d4",
          "identity_name": "Alice",
          "confidence": 0.92,
          "is_known": true
        },
        "attributes": {
          "age": 28,
          "gender": "female",
          "dominant_emotion": "happy"
        }
      }
    ],
    "face_count": 1,
    "processing_time_ms": 45.2
  }
}
```

### Identity Management

```bash
# Create identity with face
curl -X POST https://localhost/api/v1/identities \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "name=Alice" \
  -F "images=@alice.jpg"

# List identities
curl https://localhost/api/v1/identities?skip=0&limit=20 \
  -H "X-API-Key: YOUR_API_KEY"

# Add more faces
curl -X POST https://localhost/api/v1/identities/{id}/faces \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg"
```

### WebSocket Streaming

```javascript
// token = your API key
const ws = new WebSocket("wss://localhost/api/v1/streams/webcam?token=YOUR_API_KEY");
ws.send(JSON.stringify({ type: "FRAME", image: base64Data }));
ws.onmessage = (event) => {
  const analysis = JSON.parse(event.data);
  // { type: "ANALYSIS", faces: [...], processing_time_ms: 35 }
};
```

### Forensic Analysis

```bash
# Liveness check
curl -X POST https://localhost/api/v1/forensics/liveness \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "image=@suspect.jpg"

# Full forensic analysis
curl -X POST https://localhost/api/v1/forensics/analyze \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "image=@evidence.jpg"
```

### Analytics

```bash
# Overview stats
curl https://localhost/api/v1/analytics/overview \
  -H "X-API-Key: YOUR_API_KEY"

# Timeline (last 30 days)
curl "https://localhost/api/v1/analytics/timeline?days=30" \
  -H "X-API-Key: YOUR_API_KEY"

# Demographics
curl https://localhost/api/v1/analytics/demographics \
  -H "X-API-Key: YOUR_API_KEY"
```

## 🎨 Dashboard

The React dashboard provides 7 views:

| Page | Description |
|------|-------------|
| **Live Feed** | WebSocket-connected webcam with real-time bounding boxes, trust scores, and liveness indicators |
| **Identity Gallery** | CRUD management for enrolled identities with face upload |
| **Recognition Log** | Filterable audit trail (by identity, source, known/unknown, spoof) |
| **Analytics** | Overview stats, recognition timeline chart, demographic breakdowns |
| **Clusters** | HDBSCAN zero-shot identity discovery with merge-to-identity |
| **Forensics** | Deep liveness analysis with component score breakdown |
| **Settings** | System status, health checks, architecture info |

## 🔒 Production Deployment

CORTEXIA ships with a full production hardening setup. See `deploy/` for the scripts.

### Security features built in

- **TLS/HTTPS** — self-signed certs for IP-based deploys, easy swap to Let's Encrypt
- **API key auth** — required on every endpoint, including WebSocket (via query param)
- **Rate limiting** — nginx-level: 30 req/s API, 5 req/s uploads, 2 req/s WebSocket
- **Security headers** — HSTS, CSP, X-Frame-Options, X-Content-Type-Options, Permissions-Policy
- **Docker hardening** — internal-only ports, read-only filesystems, resource limits, `no-new-privileges`, network segmentation
- **Redis auth** — password-protected, not exposed outside Docker network (This was done after i got affected by **Mozi Malware(Botnet Port Scanners)** 😅 previosuly i did not know about it)
- **Upload validation** — magic-byte checking, 10 MB size limit, format whitelist
- **No debug in prod** — `/docs`, `/redoc`, `/openapi.json` all disabled; error details hidden
- **Startup validation** — app refuses to start without `SECRET_KEY` and `API_KEY`

### Deploying to a VPS

1. **Harden the VPS** (SSH, firewall, IPS):
   ```bash
   sudo bash deploy/setup-vps.sh
   ```
   This sets up: SSH key-only auth on a custom port, UFW firewall (only 80/443/SSH), Fail2ban, CrowdSec community IPS, automatic security updates, and kernel hardening.

2. **Generate TLS certs**:
   ```bash
   bash scripts/generate-certs.sh YOUR_VPS_IP
   ```

3. **Configure and launch**:
   ```bash
   cp .env.example .env
   # fill in secrets (see .env.example for instructions)
   docker compose up -d
   ```

4. **Set up backups** (optional but recommended):
   ```bash
   # add to crontab — runs daily at 2 AM
   0 2 * * * /opt/cortexia/deploy/backup.sh
   ```

### Upgrading to Let's Encrypt

When you have a domain:
```bash
apt install certbot python3-certbot-nginx
certbot certonly --standalone -d yourdomain.com
# update the cert volume path in docker-compose.yml
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires DB)
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=cortexia --cov-report=html
```

## 🛠 Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Lint
ruff check cortexia/ tests/

# Format
ruff format cortexia/ tests/

# Type check
mypy cortexia/

# Or use Make
make lint
make test
make format
```

## 📁 Project Structure

```
cortexia/
├── cortexia/                   # Python package
│   ├── core/                   # ML engine
│   │   ├── detector.py         #   RetinaFace / MediaPipe detection
│   │   ├── embedder.py         #   ArcFace 512-d embedding extraction
│   │   ├── recognizer.py       #   Gallery matching + Platt calibration
│   │   ├── clusterer.py        #   HDBSCAN identity clustering
│   │   ├── tracker.py          #   Multi-face SORT tracking
│   │   ├── trust_pipeline.py   #   🔑 Orchestrator (the centerpiece)
│   │   ├── types.py            #   Core data structures
│   │   └── models/
│   │       ├── antispoof.py    #   4-method liveness ensemble
│   │       └── attributes.py   #   Age, gender, emotion prediction
│   ├── db/                     # Database layer
│   │   ├── models.py           #   SQLAlchemy ORM (pgvector)
│   │   ├── session.py          #   Async engine + session factory
│   │   ├── repositories/       #   Data access patterns
│   │   └── migrations/         #   Alembic (async)
│   ├── api/                    # FastAPI application
│   │   ├── main.py             #   App factory + lifespan
│   │   ├── schemas/            #   Pydantic v2 models
│   │   ├── deps.py             #   Dependency injection
│   │   ├── upload_utils.py     #   Image upload validation
│   │   └── routes/             #   8 route modules
│   ├── workers/                # Celery background tasks
│   ├── cli.py                  # Click CLI (serve, enroll, recognize)
│   └── config.py               # Pydantic Settings
├── dashboard/                  # React 18 + TypeScript + Vite
│   └── src/
│       ├── pages/              #   7 page components
│       ├── components/         #   UI components (shadcn-style)
│       └── lib/                #   API client, store, utils
├── docker/                     # Container configs
│   ├── Dockerfile.api          #   Multi-stage API build
│   ├── Dockerfile.worker       #   Celery worker
│   ├── Dockerfile.dashboard    #   Node build → nginx serve
│   └── nginx.conf              #   TLS + security headers + rate limiting
├── deploy/                     # VPS hardening & ops scripts
│   ├── setup-vps.sh            #   Master setup (runs everything below)
│   ├── 01-ssh-harden.sh        #   SSH key-only, custom port
│   ├── 02-firewall.sh          #   UFW rules
│   ├── 03-fail2ban.sh          #   Brute force protection
│   ├── 04-crowdsec.sh          #   Community IPS
│   ├── 05-auto-updates.sh      #   Automatic security patches
│   ├── 06-kernel-harden.sh     #   Sysctl hardening
│   ├── backup.sh               #   Postgres + Redis backup
│   └── health-monitor.sh       #   Health check + auto-restart
├── tests/                      # pytest suite
│   ├── unit/                   #   Core module tests
│   └── integration/            #   API endpoint tests
├── scripts/                    # Utility scripts
│   ├── generate-certs.sh       #   Self-signed TLS for raw IP deploys
│   ├── seed_data.py            #   Sample identity loader
│   └── setup_models.py         #   ML model downloader
├── docs/                       # Architecture documentation
├── docker-compose.yml          # One-command deployment
├── pyproject.toml              # Python project config
├── Makefile                    # Developer shortcuts
└── .github/workflows/          # CI/CD pipelines
```

## 🔧 Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Detection** | RetinaFace (InsightFace) | SOTA accuracy, 5-point landmarks |
| **Embeddings** | ArcFace buffalo_l | 512-d, 99.83% LFW accuracy |
| **Anti-Spoof** | Heuristic ensemble | Multi-spectral analysis (FFT, color, texture, moiré) |
| **Calibration** | Platt Scaling | Mathematically grounded confidence |
| **Clustering** | HDBSCAN | Density-based, no k parameter needed |
| **API** | FastAPI | Async, OpenAPI docs, WebSocket support |
| **ORM** | SQLAlchemy 2.0 (async) | Type-safe, modern async patterns |
| **Vector DB** | pgvector | SQL-native vector search, ACID |
| **Queue** | Celery + Redis | Reliable background processing |
| **Frontend** | React 18 + TypeScript | Type-safe, component-based UI |
| **Styling** | TailwindCSS + shadcn/ui | Consistent, accessible design system |
| **Charts** | Recharts | Composable, responsive charts |
| **State** | Zustand | Minimal, performant state management |
| **Deployment** | Docker Compose | Reproducible, one-command setup |
| **CI/CD** | GitHub Actions | Lint, test, build, publish |

## 👨‍💻 Author

**Raghav Shukla**
📌 [GitHub Profile](https://github.com/raghavshuklaofficial)

## 📄 License

MIT License — see [LICENSE](License) for details.

---

<div align="center">

*Named after the fusiform face area in the brain's temporal cortex —*
*the region responsible for human face recognition.*

</div>

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.
