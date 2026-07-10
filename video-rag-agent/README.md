
## PostgreSQL Setup

A Postgres instance with the `pgvector` extension is available via Docker Compose.

Start the container:

```bash
docker-compose -f docker-files/pg-vector.yaml up -d
```

This creates a persistent volume at `~/.postgres_volume/`.

Stop the container:

```bash
docker-compose -f docker-files/pg-vector.yaml down
```

---

## Qdrant Setup

### Install docker
```bash
docker pull qdrant/qdrant
```

Start docker

```bash
docker run -d --name qdrant-vdb \
    -p 6333:6333 -p 6334:6334 \
    -v "$HOME/.qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```

> **Note:** The `--name` flag must come **before** the image name. Docker treats everything after the image name as a command to execute inside the container, not as a container flag.

Stop the container:
```bash
docker stop qdrant-vdb
```

Remove the container:
```bash
docker rm qdrant-vdb
```