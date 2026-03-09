# Cortexia

Face recognition system with anti-spoofing, clustering, and a full audit trail. Built to go beyond the usual `face_recognition` library demos.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

I built this because I was frustrated with how every face recognition project on GitHub is just a script that wraps `dlib` or `face_recognition` with no real backend, no anti-spoofing, and definitely no production setup. Cortexia is my attempt at building what an actual deployable face intelligence system would look like — proper DB, proper API, proper security.

**What it does:**
- Detects faces (RetinaFace), extracts ArcFace embeddings (512-d), matches against enrolled identities
- Anti-spoofing pipeline that catches photo/screen attacks using FFT, color, texture, and moiré analysis
- Confidence scores are Platt-calibrated (not just raw cosine thresholds)
- HDBSCAN clustering to discover unknown recurring faces automatically
- React dashboard for live feed, identity management, analytics
- Full Docker Compose deployment with TLS, rate limiting, hardened containers

## Architecture

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

### Trust Pipeline

The main thing that differentiates this from other projects. Every face goes through all stages sequentially:

```
Frame → Detection → Alignment → Liveness → Embedding → Recognition → Attributes
         RetinaFace   ArcFace     4-method    512-d       Platt        Age/Gender
         (99.4%)      112×112     ensemble    L2-norm     Scaling      Emotion
```

Trust score is a weighted combination: `0.20 * detection_conf + 0.40 * liveness_conf + 0.40 * recognition_conf`. If the face is detected as a spoof, the score gets multiplied by 0.3 (heavy penalty).

## How it works

### Face Detection & Recognition
- RetinaFace detector with auto GPU/CPU selection
- ArcFace buffalo_l embedder (512-d, L2-normalized)
- Multi-face tracking across video frames (SORT-inspired IoU + embedding matching)
- Gallery management with centroid-based identity matching

### Anti-Spoofing
I didn't want to train a whole neural net for this, so the liveness detector uses a combination of heuristic signals:
- **FFT analysis** — screens and printers leave frequency-domain artifacts
- **YCrCb color analysis** — printed photos have shifted chroma distributions
- **Texture (LBP variance + Sobel)** — live faces have micro-texture from pores etc
- **Moiré detection** — autocorrelation picks up the periodic patterns from screens

Each method scores 0-1, and they're combined with weights (0.30/0.20/0.30/0.20).

### Platt-Calibrated Confidence
Raw cosine similarity is terrible as a probability estimate. Instead of just thresholding at 0.45, I use Platt scaling to get actual calibrated probabilities:
```
P(match) = 1 / (1 + exp(-15.0 * (1 - sim) + 6.5))
```
The parameters (A=-15.0, B=6.5) were tuned empirically on ArcFace embedding distributions.

### Clustering (Zero-Shot Identity Discovery)
HDBSCAN runs periodically on unknown face embeddings to find recurring people who aren't enrolled. No need to specify cluster count — it figures it out from the density structure.

### Forensic Audit Trail
Every recognition event goes into an append-only table with timestamp, trust score, spoof flag, matched identity, and face attributes. Queryable by time range, identity, source.

### Vector Search
pgvector enables sub-millisecond nearest-neighbor search with IVFFlat indexing + cosine distance. Embeddings are stored as `Vector(512)` columns with full ACID guarantees.

## Quick Start

### What you need

- Docker & Docker Compose
- `.env` file with required secrets (see below)

### Setup

```bash
git clone https://github.com/raghavshuklaofficial/cortexia.git
cd cortexia
cp .env.example .env
```

Generate secrets and put them in `.env`:

```bash
openssl rand -hex 32   # SECRET_KEY
openssl rand -hex 32   # API_KEY
openssl rand -hex 24   # POSTGRES_PASSWORD
openssl rand -hex 16   # REDIS_PASSWORD
```

For local dev, set `APP_ENV=development` in `.env`.

Then:

```bash
docker compose up --build
```

Dashboard will be at **https://localhost** (you'll get a cert warning since it uses self-signed TLS by default).

### Services

| Service | Port | Description |
|---------|------|-------------|
| Nginx | 443 | HTTPS gateway with TLS |
| Nginx | 80 | Redirects to HTTPS |
| API | 8000 (internal) | FastAPI + Trust Pipeline |
| PostgreSQL | 5432 (internal) | Database + pgvector |
| Redis | 6379 (internal) | Celery broker + cache |

Internal ports only — everything goes through nginx.

### Seed sample data

```bash
docker compose exec api python scripts/seed_data.py
```

### CLI

```bash
cortexia serve --port 8000       # start API server
cortexia enroll --name "Alice" --image photos/alice.jpg
cortexia recognize --image test.jpg
cortexia info                    # system info
```

## API

All endpoints need `X-API-Key` header in production.

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
# Create
curl -X POST https://localhost/api/v1/identities \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "name=Alice" -F "images=@alice.jpg"

# List
curl https://localhost/api/v1/identities?skip=0&limit=20 \
  -H "X-API-Key: YOUR_API_KEY"

# Add more face images to existing identity
curl -X POST https://localhost/api/v1/identities/{id}/faces \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "images=@photo1.jpg" -F "images=@photo2.jpg"
```

### WebSocket (Live Feed)

```javascript
const ws = new WebSocket("wss://localhost/api/v1/streams/webcam?token=YOUR_API_KEY");
ws.send(JSON.stringify({ frame: base64JpegData }));
ws.onmessage = (e) => console.log(JSON.parse(e.data));
// returns { type: "ANALYSIS", faces: [...], processing_time_ms: 35 }
```

### Other endpoints

```bash
# Liveness check
curl -X POST https://localhost/api/v1/forensics/liveness \
  -H "X-API-Key: YOUR_API_KEY" -F "image=@photo.jpg"

# Analytics overview
curl https://localhost/api/v1/analytics/overview -H "X-API-Key: YOUR_API_KEY"

# Timeline
curl "https://localhost/api/v1/analytics/timeline?days=30" -H "X-API-Key: YOUR_API_KEY"
```

## Dashboard

7 pages in the React frontend:

| Page | Description |
|------|-------------|
| **Live Feed** | WebSocket-connected webcam with real-time bounding boxes, trust scores, and liveness indicators |
| **Identity Gallery** | CRUD management for enrolled identities with face upload |
| **Recognition Log** | Filterable audit trail (by identity, source, known/unknown, spoof) |
| **Analytics** | Overview stats, recognition timeline chart, demographic breakdowns |
| **Clusters** | HDBSCAN zero-shot identity discovery with merge-to-identity |
| **Forensics** | Deep liveness analysis with component score breakdown |
| **Settings** | System status, health checks |

## Production Deployment

### Security

The whole security setup exists because I deployed this on a raw VPS and immediately got hit by Mozi botnet port scanners 😅 — so yeah, everything is locked down now:

- TLS/HTTPS with self-signed certs (swap to Let's Encrypt when you have a domain)
- API key auth on every endpoint including WebSocket
- Nginx rate limiting (30 req/s API, 5 req/s uploads, 2 req/s WebSocket)
- Security headers (HSTS, CSP, X-Frame-Options, etc.)
- Docker hardening — internal-only ports, resource limits, `no-new-privileges`, read-only filesystems where possible
- Redis is password-protected and not exposed outside the Docker network
- Upload validation with magic-byte checking, 10 MB size limit
- `/docs` and `/redoc` disabled in production, error details hidden
- App won't start without `SECRET_KEY` and `API_KEY` set

### Deploy to a VPS

1. Harden the server:
   ```bash
   sudo bash deploy/setup-vps.sh
   ```
   Sets up SSH key-only auth, UFW firewall, Fail2ban, CrowdSec, auto security updates, kernel hardening.

2. TLS certs:
   ```bash
   bash scripts/generate-certs.sh YOUR_VPS_IP
   ```

3. Configure and launch:
   ```bash
   cp .env.example .env
   # fill in secrets (see .env.example for instructions)
   docker compose up -d
   ```

4. Backups (optional):
   ```bash
   # daily at 2 AM via crontab
   0 2 * * * /opt/cortexia/deploy/backup.sh
   ```

### Let's Encrypt

When you have a domain:
```bash
apt install certbot python3-certbot-nginx
certbot certonly --standalone -d yourdomain.com
# then update cert paths in docker-compose.yml
```

## Testing

```bash
pytest tests/ -v              # all tests
pytest tests/unit/ -v         # unit only
pytest tests/integration/ -v  # integration (needs DB)
pytest tests/ --cov=cortexia --cov-report=html  # with coverage
```

## Dev Setup

```bash
pip install -e ".[dev]"

ruff check cortexia/ tests/   # lint
ruff format cortexia/ tests/  # format
mypy cortexia/                # type check

# or just use make
make lint && make test && make format
```

## Project Structure

```
cortexia/
├── cortexia/               # python package
│   ├── core/               # ML pipeline (detector, embedder, recognizer, clusterer, tracker)
│   │   ├── trust_pipeline.py   # the main orchestrator
│   │   └── models/         # antispoof + attribute prediction
│   ├── db/                 # SQLAlchemy + pgvector + alembic migrations
│   ├── api/                # FastAPI routes, schemas, middleware
│   ├── workers/            # celery background tasks
│   ├── cli.py              # click CLI
│   └── config.py           # pydantic settings
├── dashboard/              # React 18 + TypeScript + Vite + Tailwind
├── docker/                 # Dockerfiles + nginx config
├── deploy/                 # VPS hardening scripts
├── tests/                  # pytest (unit + integration)
├── scripts/                # seed data, model download, cert generation
├── docker-compose.yml
└── pyproject.toml
```

## Tech Stack

- **ML**: RetinaFace + ArcFace (InsightFace), HDBSCAN, Platt scaling
- **Backend**: FastAPI, SQLAlchemy 2.0 (async), pgvector, Celery + Redis
- **Frontend**: React 18, TypeScript, TailwindCSS, Zustand, Recharts
- **Infra**: Docker Compose, Nginx (TLS + rate limiting), PostgreSQL 16

## Author

Raghav Shukla — [GitHub](https://github.com/raghavshuklaofficial)

## License

MIT — see [LICENSE](License).
