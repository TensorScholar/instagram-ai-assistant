# Aura Platform - Resilience Patterns Documentation

This document outlines the resilience patterns implemented in the Aura platform to handle various failure scenarios and ensure system stability.

## Overview

The Aura platform implements multiple resilience patterns to handle:
- Database transaction failures
- LLM API timeouts and circuit breaker protection
- Poison pill message detection and DLQ routing
- Retry tracking and failure recovery

## Resilience Patterns

### 1. Transactional Decorator Pattern

**Purpose:** Ensure database transactions are properly rolled back on network failures.

**Implementation:** Enhanced `@transactional` decorator in `src/shared_lib/app/db/database.py`

**Key Features:**
- Catches `sqlalchemy.exc.OperationalError` and `DisconnectionError`
- Explicitly attempts `session.rollback()` on network failures
- Logs critical failures for monitoring
- Raises `DatabaseTransactionError` for proper error handling

**Usage:**
```python
@transactional
async def create_product_with_variants(session: AsyncSession, tenant_id: UUID, product_data: dict):
    product = await create_product(session, tenant_id, product_data)
    await create_variants(session, product.id, product_data['variants'])
    return product
```

**Monitoring Metrics:**
- `database_transaction_rollbacks_total{reason="network_error"}`

### 2. Circuit Breaker Pattern for Timeouts

**Purpose:** Prevent cascading failures from LLM API timeouts and connection issues.

**Implementation:** Enhanced circuit breaker in `src/shared_lib/app/ai/resilient_llm.py`

**Key Features:**
- Aggressive HTTP timeouts: 5s connect, 10s read
- Circuit breaker detects `httpx.TimeoutException` and `httpx.ConnectError`
- Automatic circuit opening after 3 failures
- 60-second recovery timeout

**Configuration:**
```python
self.http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(
        connect=5.0,  # 5 second connection timeout
        read=10.0,   # 10 second read timeout
        write=5.0,   # 5 second write timeout
        pool=5.0,    # 5 second pool timeout
    )
)

self.circuit_breaker = pybreaker.CircuitBreaker(
    fail_max=3,
    reset_timeout=60,
    expected_exception=(
        LLMError,
        httpx.TimeoutException,
        httpx.ConnectError,
        httpx.HTTPStatusError,
    ),
)
```

**Monitoring Metrics:**
- `llm_circuit_breaker_opens_total`
- `llm_api_request_duration_seconds`

### 3. Poison Pill Handling Pattern

**Purpose:** Detect and immediately route malformed messages to Dead-Letter Queue (DLQ).

**Implementation:** `PoisonPillDetector` in `src/shared_lib/app/utils/retry_tracker.py`

**Key Features:**
- Detects critical, non-transient errors:
  - `JSONDecodeError` - Malformed JSON
  - `UnicodeDecodeError` - Invalid encoding
  - `AttributeError` - Missing required attributes
  - `KeyError` - Missing required keys
  - `ValueError` - Invalid values
  - `TypeError` - Type mismatches
- Immediate DLQ routing without retries
- Human-readable error reasons

**Usage:**
```python
detector = PoisonPillDetector()
if detector.is_poison_pill(exception):
    reason = detector.get_poison_reason(exception)
    await route_to_dlq(message_id, reason)
```

**Monitoring Metrics:**
- `messages_sent_to_dlq_total{reason="poison_pill"}`

### 4. Retry Tracking Pattern

**Purpose:** Track message retry attempts and route to DLQ after maximum retries exceeded.

**Implementation:** `RetryTracker` in `src/shared_lib/app/utils/retry_tracker.py`

**Key Features:**
- Redis-based retry counter tracking
- Configurable maximum retry count (default: 3)
- Automatic TTL for retry counters (default: 1 hour)
- Atomic increment operations

**Usage:**
```python
tracker = RetryTracker(redis_client, max_retries=3)
should_dlq = await tracker.increment_and_check(message_id)
if should_dlq:
    await route_to_dlq(message_id, "max_retries_exceeded")
```

**Monitoring Metrics:**
- `messages_sent_to_dlq_total{reason="max_retries_exceeded"}`

### 5. Enhanced Celery Task Pattern

**Purpose:** Integrate poison pill detection and retry tracking into Celery tasks.

**Implementation:** `@resilient_task` decorator in `src/shared_lib/app/utils/resilient_tasks.py`

**Key Features:**
- Automatic poison pill detection
- Redis-based retry tracking
- Automatic DLQ routing
- Configurable retry limits and delays

**Usage:**
```python
@resilient_task(
    bind=True,
    max_retries=3,
    poison_pill_detection=True,
    dlq_routing=True
)
async def process_message(self, message_data: dict):
    # Task implementation
    pass
```

### 6. Celery Task Timeout Pattern

**Purpose:** Prevent worker thread exhaustion from hanging tasks.

**Implementation:** Enhanced Celery configuration in `src/intelligence_worker/app/celery_app.py`

**Key Features:**
- Soft time limit: 45 seconds (graceful shutdown)
- Hard time limit: 60 seconds (force kill)
- Automatic task termination
- Worker process recycling

**Configuration:**
```python
celery_app.conf.update(
    task_soft_time_limit=45,   # 45 seconds soft limit
    task_time_limit=60,        # 60 seconds hard limit
    worker_max_tasks_per_child=1000,  # Restart workers frequently
)
```

## Error Handling Hierarchy

1. **Poison Pill Detection** - Immediate DLQ routing for malformed messages
2. **Retry Tracking** - Track retry attempts and route to DLQ after max retries
3. **Circuit Breaker** - Prevent cascading failures from external services
4. **Transaction Rollback** - Ensure data consistency on network failures
5. **Task Timeouts** - Prevent worker thread exhaustion

## Monitoring and Alerting

### Key Metrics

- **Database Layer:**
  - `database_transaction_rollbacks_total{reason="network_error"}`
  - `sqlalchemy_pool_connections_active`

- **LLM Layer:**
  - `llm_circuit_breaker_opens_total`
  - `llm_api_request_duration_seconds`

- **Message Processing:**
  - `messages_sent_to_dlq_total{reason="poison_pill"}`
  - `messages_sent_to_dlq_total{reason="max_retries_exceeded"}`
  - `celery_tasks_pending{queue="realtime_queue"}`
  - `celery_tasks_pending{queue="bulk_queue"}`

### Alert Thresholds

- **Circuit Breaker Opens:** Alert when circuit breaker opens
- **DLQ Routing:** Alert when messages are routed to DLQ
- **Transaction Rollbacks:** Alert on high rollback rates
- **Task Timeouts:** Alert on high timeout rates

## Best Practices

### 1. Error Classification
- **Transient Errors:** Retry with exponential backoff
- **Poison Pills:** Immediate DLQ routing
- **Network Failures:** Explicit rollback and retry
- **Timeout Errors:** Circuit breaker protection

### 2. Monitoring Strategy
- Monitor all resilience pattern metrics
- Set up alerts for critical failure modes
- Track DLQ message patterns for system health
- Monitor circuit breaker states

### 3. Testing Strategy
- Unit tests for each resilience pattern
- Integration tests for end-to-end scenarios
- Chaos engineering tests for failure simulation
- Load tests for timeout and circuit breaker validation

## Configuration

### Environment Variables

```bash
# Database resilience
DB_POOL_SIZE=50
DB_MAX_OVERFLOW=100
DB_POOL_RECYCLE=1800

# Celery resilience
CELERY_TASK_SOFT_TIME_LIMIT=45
CELERY_TASK_HARD_TIME_LIMIT=60
CELERY_WORKER_MAX_TASKS_PER_CHILD=1000

# Retry tracking
RETRY_TRACKER_MAX_RETRIES=3
RETRY_TRACKER_TTL=3600

# Circuit breaker
CIRCUIT_BREAKER_FAIL_MAX=3
CIRCUIT_BREAKER_RESET_TIMEOUT=60
```

## Troubleshooting

### Common Issues

1. **High DLQ Routing Rate**
   - Check for malformed message patterns
   - Verify poison pill detection accuracy
   - Review retry limit configuration

2. **Circuit Breaker Frequently Open**
   - Check LLM API health and response times
   - Verify timeout configuration
   - Review retry and backoff settings

3. **Transaction Rollback Issues**
   - Check database connection health
   - Verify network stability
   - Review transaction isolation levels

4. **Task Timeout Issues**
   - Check worker resource usage
   - Verify task complexity and duration
   - Review timeout configuration

### Debug Commands

```bash
# Check DLQ messages
kubectl exec -it deployment/rabbitmq -n aura-platform -- rabbitmqctl list_queues name messages

# Check circuit breaker state
kubectl logs -f deployment/intelligence-worker -n aura-platform | grep "circuit breaker"

# Check retry tracking
kubectl exec -it deployment/redis -n aura-platform -- redis-cli keys "retry_count:*"

# Check transaction rollbacks
kubectl logs -f deployment/api-gateway -n aura-platform | grep "rollback"
```

## Conclusion

The Aura platform implements comprehensive resilience patterns to handle various failure scenarios. These patterns work together to ensure system stability, data consistency, and graceful degradation under stress. Regular monitoring and testing of these patterns is essential for maintaining production reliability.
