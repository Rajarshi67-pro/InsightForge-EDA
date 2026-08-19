# 🚀 DataLens AI — Production Hosting & Deployment Guide

This guide details how to deploy and host **DataLens AI** to the cloud or on a private server.

---

## 📋 Table of Contents
1. [Option 1: Deploy on Render.com (Recommended Free / Low-Cost)](#option-1-deploy-on-rendercom-recommended-free--low-cost)
2. [Option 2: Deploy on Railway.app (1-Click Git Deploy)](#option-2-deploy-on-railwayapp-1-click-git-deploy)
3. [Option 3: Deploy on Google Cloud Run (Serverless Container)](#option-3-deploy-on-google-cloud-run-serverless-container)
4. [Option 4: Deploy on Hugging Face Spaces (Free Docker Space)](#option-4-deploy-on-hugging-face-spaces-free-docker-space)
5. [Option 5: Self-Hosted VPS / Cloud Server (Docker Compose + Nginx)](#option-5-self-hosted-vps--cloud-server-docker-compose--nginx)

---

## Option 1: Deploy on Render.com (Recommended Free / Low-Cost)

[Render](https://render.com) provides native Python web service support and automatic SSL.

### Step-by-Step:
1. Push your repository to **GitHub** or **GitLab**:
   ```bash
   git init
   git add .
   git commit -m "feat: production release of DataLens AI"
   git remote add origin https://github.com/YOUR_USERNAME/DataLensAI.git
   git push -u origin main
   ```
2. Log in to [Render Dashboard](https://dashboard.render.com/) and click **New +** $\to$ **Web Service**.
3. Select your GitHub repository.
4. Fill in the deployment settings:
   * **Name**: `datalens-ai`
   * **Environment**: `Python 3`
   * **Region**: Choose closest to you (e.g., *Singapore*, *Oregon*, *Frankfurt*)
   * **Branch**: `main`
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT --workers 2`
5. *(Optional)* Under **Environment Variables**, add:
   * `GEMINI_API_KEY`: `your_gemini_api_key_here` (Optional for LLM features)
6. Click **Create Web Service**. Your app will be live at `https://datalens-ai.onrender.com` in ~2 minutes!

---

## Option 2: Deploy on Railway.app (1-Click Git Deploy)

[Railway](https://railway.app) automatically detects the [`Dockerfile`](file:///Users/souvik/Desktop/MY_CODES/Projects/DataLensAI/Dockerfile) and [`railway.json`](file:///Users/souvik/Desktop/MY_CODES/Projects/DataLensAI/railway.json).

### Step-by-Step:
1. Go to [Railway.app](https://railway.app) and sign in with GitHub.
2. Click **New Project** $\to$ **Deploy from GitHub repo**.
3. Select `DataLensAI`.
4. Railway will automatically build the Docker container and route traffic.
5. In **Settings** $\to$ **Networking**, click **Generate Domain** to get a public URL (e.g. `datalens-ai.up.railway.app`).

---

## Option 3: Deploy on Google Cloud Run (Serverless Container)

Google Cloud Run offers a generous free tier (2 million requests/month) with auto-scaling to zero.

### Step-by-Step:
1. Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) (`gcloud`).
2. Build and submit your container to Google Artifact Registry:
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/datalens-ai
   ```
3. Deploy the service to Cloud Run:
   ```bash
   gcloud run deploy datalens-ai \
     --image gcr.io/YOUR_PROJECT_ID/datalens-ai \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 1Gi \
     --port 8000
   ```
4. Cloud Run will output your live HTTPS URL.

---

## Option 4: Deploy on Hugging Face Spaces (Free Docker Space)

Hugging Face provides free compute for machine learning & data tools.

### Step-by-Step:
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Space Name: `DataLens-AI`
3. License: `MIT` or `Apache 2.0`
4. Space SDK: **Docker** $\to$ **Blank**.
5. Clone your Hugging Face space repo and push all files from this project into it:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/DataLens-AI
   git push space main
   ```
6. Hugging Face will automatically build your Docker container and provide a permanent public link.

---

## Option 5: Self-Hosted VPS / Cloud Server (Docker Compose + Nginx)

If deploying to DigitalOcean, AWS EC2, Hetzner, Linode, or any Ubuntu/Debian VPS:

### Step-by-Step:
1. SSH into your VPS:
   ```bash
   ssh root@YOUR_SERVER_IP
   ```
2. Install Docker & Docker Compose:
   ```bash
   apt-get update && apt-get install -y docker.io docker-compose
   ```
3. Clone your repo onto the server:
   ```bash
   git clone https://github.com/YOUR_USERNAME/DataLensAI.git
   cd DataLensAI
   ```
4. Start the container in the background:
   ```bash
   docker-compose up -d --build
   ```
5. *(Optional)* Configure **Nginx Reverse Proxy & SSL (Certbot)**:
   ```nginx
   server {
       server_name yourdomain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
           client_max_body_size 50M;
       }
   }
   ```
   Run `certbot --nginx -d yourdomain.com` for free automatic SSL.

---

## 🔍 Verification & Health Check

After hosting, you can verify your service status anytime:
```bash
curl https://YOUR_HOSTED_DOMAIN/healthz
```
Expected response:
```json
{
  "status": "healthy",
  "service": "DataLens AI",
  "version": "1.0.0"
}
```
