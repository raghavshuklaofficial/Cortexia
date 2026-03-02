"""
Celery worker configuration and task definitions.
"""

from __future__ import annotations

from celery import Celery

from cortexia.config import get_settings

_settings = get_settings()

celery_app = Celery(
    "cortexia",
    broker=_settings.celery_broker_url,
    backend=_settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max per task
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
)
