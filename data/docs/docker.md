# Docker Documentation

Source: https://docs.docker.com/get-started/

## Basic Docker CLI Commands
Common commands to manage Docker containers and images:
- `docker build -t my-app:latest .` : Builds a Docker image from a Dockerfile in the current directory.
- `docker run -d -p 8080:80 --name webserver nginx` : Runs an nginx container in detached mode, forwarding host port 8080 to container port 80.
- `docker ps` : Lists currently running containers. Use `docker ps -a` to see stopped containers.
- `docker stop <container_id>` : Gracefully stops a running container.
- `docker rm <container_id>` : Removes a stopped container.
- `docker logs -f <container_id>` : Follows the live log output of a container.

## Writing a Dockerfile
A Dockerfile defines the instructions for assembling an image.
```dockerfile
# Step 1: Base image
FROM python:3.11-slim

# Step 2: Set working directory
WORKDIR /app

# Step 3: Copy dependency specifications and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy application source code
COPY . .

# Step 5: Expose port and define default entrypoint
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Multi-Stage Builds
Multi-stage builds allow you to drastically reduce final image size by separating build tools from runtime environments.
```dockerfile
# Build stage
FROM golang:1.21 AS builder
WORKDIR /src
COPY . .
RUN CGO_ENABLED=0 go build -o /bin/app

# Final production stage
FROM alpine:latest
WORKDIR /app
COPY --from=builder /bin/app /app/app
CMD ["/app/app"]
```
This produces minimal runtime images without build compilers or intermediate source files.

## Port Mapping and Networking
Containers run in isolated network namespaces by default.
- Host port to container port: `-p 8000:8000` maps incoming host port 8000 to port 8000 in the container.
- Custom bridge networks allow containers to communicate by container name:
  `docker network create my-net`
  `docker run --network my-net --name backend my-backend`
  `docker run --network my-net --name db postgres`

## Volumes and Persistent Storage
Containers have an ephemeral writable layer. To persist data beyond the lifecycle of a container, use Docker volumes.
- Named volume: `docker run -v pgdata:/var/lib/postgresql/data postgres`
- Bind mount: `docker run -v $(pwd):/app python:3.11` (maps local working directory to container `/app`).
Volumes are managed by Docker and stored in `/var/lib/docker/volumes/` on Linux.

## Docker Compose
Docker Compose defines and runs multi-container Docker applications via YAML.
```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - db
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
  db:
    image: postgres:15
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb

volumes:
  pgdata:
```
Run `docker compose up -d` to launch all services.
