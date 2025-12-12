# 🚀 Deployment Guide

## Option 1: Render (Recommended - Free Tier Available)

### Deploy Backend

1. **Create account** at [render.com](https://render.com)

2. **New Web Service** → Connect your GitHub repo

3. **Configure:**
   - **Root Directory:** `backend`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Deploy** - Note your backend URL (e.g., `https://basket-trading-api.onrender.com`)

### Deploy Frontend

1. **New Static Site** → Connect same repo

2. **Configure:**
   - **Root Directory:** `frontend`
   - **Build Command:** `npm install && npm run build`
   - **Publish Directory:** `dist`

3. **Environment Variables:**
   ```
   VITE_API_URL=https://your-backend-url.onrender.com
   ```

4. **Deploy**

---

## Option 2: Vercel (Frontend) + Render (Backend)

### Backend on Render
Same as above.

### Frontend on Vercel

1. **Create account** at [vercel.com](https://vercel.com)

2. **Import Project** → Select your GitHub repo

3. **Configure:**
   - **Root Directory:** `frontend`
   - **Framework Preset:** Vite
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`

4. **Environment Variables:**
   ```
   VITE_API_URL=https://your-backend-url.onrender.com
   ```

5. **Deploy**

---

## Option 3: Railway (All-in-One)

1. **Create account** at [railway.app](https://railway.app)

2. **New Project** → Deploy from GitHub

3. **Add Backend Service:**
   - Select `backend` folder
   - Railway auto-detects Python
   - Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Add Frontend Service:**
   - Select `frontend` folder
   - Add env var: `VITE_API_URL=https://your-backend.railway.app`

5. **Generate Domain** for both services

---

## Required Frontend Changes for Production

Update `frontend/src/App.jsx` to use environment variable:

```javascript
// Change this line:
const API_BASE = '/api';

// To this:
const API_BASE = import.meta.env.VITE_API_URL 
  ? `${import.meta.env.VITE_API_URL}/api` 
  : '/api';
```

Or create `frontend/.env.production`:
```
VITE_API_URL=https://your-backend-url.onrender.com
```

---

## Update CORS for Production

In `backend/main.py`, update CORS origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-frontend-url.vercel.app",
        "https://your-frontend-url.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Environment Variables Summary

### Backend (Render/Railway)
| Variable | Value |
|----------|-------|
| `PYTHON_VERSION` | `3.11` |
| `PORT` | (auto-set by platform) |

### Frontend (Vercel/Render)
| Variable | Value |
|----------|-------|
| `VITE_API_URL` | `https://your-backend-url.onrender.com` |

---

## Free Tier Limitations

### Render Free Tier
- Backend sleeps after 15 min inactivity
- First request after sleep takes ~30 seconds
- 750 hours/month

### Vercel Free Tier
- Unlimited static sites
- 100GB bandwidth/month

### Railway Free Tier
- $5 credit/month
- Good for small projects

---

## Quick Deploy Checklist

- [ ] Push code to GitHub
- [ ] Deploy backend first, get URL
- [ ] Update frontend with backend URL
- [ ] Deploy frontend
- [ ] Update backend CORS with frontend URL
- [ ] Test the live app!

---

## Troubleshooting

### "CORS Error"
→ Add your frontend URL to backend CORS origins

### "API Not Found"
→ Check `VITE_API_URL` is set correctly (no trailing slash)

### "Backend Sleeping"
→ Free tier limitation, first request wakes it up (~30s)

### "Build Failed"
→ Check Node/Python versions match your local setup

