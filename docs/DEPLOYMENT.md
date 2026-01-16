# Deployment Guide

This guide covers how to deploy the LLM API Proxy locally, using Docker, or on a free VPS.

## Option 1: Docker (Recommended)

Docker is the easiest way to run the proxy, ensuring all dependencies are isolated.

### Prerequisites
- Docker and Docker Compose installed.

### Steps
1.  **Configure Environment:**
    Ensure you have a `.env` file with your API keys.
    ```bash
    cp simple-env-template.env .env
    # Edit .env with your keys
    ```

2.  **Run with Compose:**
    ```bash
    docker-compose up -d
    ```

3.  **Verify:**
    Check the logs to ensure it started correctly:
    ```bash
    docker-compose logs -f
    ```
    Access the health endpoint: `http://localhost:8000/health`

### Updating
To update the proxy (e.g., after changing `requirements.txt` or code):
```bash
docker-compose down
docker-compose build
docker-compose up -d
```

---

## Option 2: Local Deployment (Python)

Suitable for development or running on a personal machine.

### Prerequisites
- Python 3.10+
- `pip`

### Steps
1.  **Create Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run:**
    ```bash
    python src/proxy_app/main.py
    ```

---

## Option 3: Free Hosting (Oracle Cloud Always Free)

The best "free forever" option for this proxy is **Oracle Cloud's Always Free Tier**, specifically the **ARM Ampere** instances.

### Why Oracle Cloud?
- **Specs:** 4 OCPUs (ARM), 24GB RAM (Generous!)
- **Storage:** 200GB Block Volume
- **Network:** Public IP included
- **Cost:** $0.00 / month

### Deployment Steps on Oracle VPS

1.  **Create Instance:**
    -   Sign up for Oracle Cloud Free Tier.
    -   Create a "VM.Standard.A1.Flex" instance (Ubuntu 22.04 or 24.04).
    -   Save your SSH key.

2.  **Network Setup (Ingress Rules):**
    -   In the Oracle Cloud Console, go to your VCN > Security Lists.
    -   Add an Ingress Rule to allow traffic on port **8000** (TCP) from `0.0.0.0/0`.
    -   *Note:* You may also need to open the port on the instance firewall (iptables/ufw).

3.  **Connect via SSH:**
    ```bash
    ssh -i path/to/key.key ubuntu@<public-ip>
    ```

4.  **Install Docker on VPS:**
    ```bash
    sudo apt update
    sudo apt install -y docker.io docker-compose
    sudo usermod -aG docker $USER
    # Log out and log back in for group change to take effect
    ```

5.  **Deploy Proxy:**
    Clone your repo (or copy files):
    ```bash
    git clone <your-repo-url> llm-proxy
    cd llm-proxy
    ```
    
    Create your `.env` file:
    ```bash
    nano .env
    # Paste your API keys and PROXY_API_KEY
    ```

    Start it up:
    ```bash
    docker-compose up -d
    ```

6.  **Access:**
    Your proxy is now available at `http://<public-ip>:8000/v1/chat/completions`.

---

## Option 4: Other Free PaaS Options

**Warning:** PaaS free tiers often have limitations (spin-down on inactivity, no persistent storage, short timeouts) that can affect an API proxy.

### Hugging Face Spaces (Docker)
- **Tier:** Free (CPU Basic)
- **Setup:** Create a new Space, select "Docker" SDK.
- **Config:** Add API keys as "Secrets" in Space settings.
- **Limitation:** Public spaces make your code visible (but secrets are hidden). Use a Private Space if possible.

### Render / Railway
- **Setup:** Connect GitHub repo.
- **Command:** `python src/proxy_app/main.py`
- **Limitation:** Free tiers often sleep after inactivity, causing latency on the first request.
