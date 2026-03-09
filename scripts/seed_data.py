#!/usr/bin/env python3
"""
Seed sample data for development/demo.

Creates sample identities from any images found in a sample_faces/ directory,
or generates synthetic placeholder data if no images are available.

Usage:
    python scripts/seed_data.py
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from random import choice, random, randint

import numpy as np


async def seed():
    from cortexia.config import get_settings
    from cortexia.db.session import init_db, async_session_factory
    from cortexia.db.models import Identity, FaceEmbedding, RecognitionEvent

    print("=" * 60)
    print("  Seed Data")
    print("=" * 60)

    settings = get_settings()
    await init_db()

    async with async_session_factory() as session:
        # Check if data already exists
        from sqlalchemy import select, func

        count = await session.scalar(select(func.count()).select_from(Identity))
        if count and count > 0:
            print(f"\n  Database already has {count} identities. Skipping seed.")
            print("  To re-seed, clear the database first.")
            return

        print("\n[1/3] Creating sample identities...")

        names = [
            "Ada Lovelace",
            "Alan Turing",
            "Grace Hopper",
            "Linus Torvalds",
            "Margaret Hamilton",
            "Dennis Ritchie",
            "Barbara Liskov",
            "Vint Cerf",
        ]

        identities = []
        for name in names:
            identity = Identity(
                id=str(uuid.uuid4()),
                name=name,
                metadata_json={"role": "scientist", "source": "seed_script"},
                privacy_score=round(random() * 0.3 + 0.7, 2),
            )
            session.add(identity)
            identities.append(identity)
            print(f"      ✓ {name}")

        await session.flush()

        print("\n[2/3] Generating synthetic embeddings...")

        for identity in identities:
            n_faces = randint(2, 5)
            base_emb = np.random.randn(512).astype(np.float32)
            base_emb /= np.linalg.norm(base_emb)

            for j in range(n_faces):
                # Small perturbation to simulate different photos of same person
                noise = np.random.randn(512).astype(np.float32) * 0.05
                emb = base_emb + noise
                emb /= np.linalg.norm(emb)

                face_emb = FaceEmbedding(
                    id=str(uuid.uuid4()),
                    identity_id=identity.id,
                    embedding=emb.tolist(),
                    source_image=f"seed_{identity.name.lower().replace(' ', '_')}_{j}.jpg",
                )
                session.add(face_emb)

            print(f"      ✓ {identity.name}: {n_faces} face embeddings")

        print("\n[3/3] Creating sample recognition events...")

        now = datetime.now(timezone.utc)
        sources = ["webcam", "api_upload", "batch_job", "security_cam"]

        for i in range(50):
            identity = choice(identities) if random() > 0.3 else None
            event = RecognitionEvent(
                id=str(uuid.uuid4()),
                identity_id=identity.id if identity else None,
                source=choice(sources),
                is_known=identity is not None,
                trust_score=round(random() * 0.4 + 0.6, 3),
                is_spoof=random() < 0.08,
                attributes_json={
                    "age": randint(20, 65),
                    "gender": choice(["male", "female"]),
                    "emotion": choice(["neutral", "happy", "surprised"]),
                },
                created_at=now - timedelta(hours=randint(0, 720)),
            )
            session.add(event)

        await session.commit()
        print(f"      ✓ 50 recognition events")

    print()
    print("=" * 60)
    print("  Seed complete!")
    print("=" * 60)


def main():
    asyncio.run(seed())


if __name__ == "__main__":
    main()
