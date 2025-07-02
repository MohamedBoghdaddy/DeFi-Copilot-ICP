# DeFi Copilot 🧠💼

**A decentralized AI-powered financial advisor dApp built on the Internet Computer Protocol (ICP), combining on‑chain identity, machine learning forecasting, and portfolio simulation.**

---

## 🚀 Features

- **Wallet-Based Login**
  - Supports Plug Wallet and Internet Identity for seamless Web3 authentication.
- **AI Chatbot**
  - Powered by Google Gemini API for real-time financial advisory and Q&A.
- **Forecasting Engine**
  - Employs ARIMA, LSTM, and XGBoost models to predict prices of stocks, BTC, and gold.
- **Risk & Goal-Based Portfolio Recommender**
  - Suggests tailored portfolios based on user-defined risk tolerance and objectives.
- **On‑Chain Profile Storage**
  - Uses a Motoko ICP canister to store and retrieve user profile data.
- **DeFi Investment Simulator**
  - Features a mock DEX using test tokens for simulated trades.
- **CKBTC Integration (optional)**
  - Enables Chain-Key Bitcoin deposits and withdrawals.
- **Express.js API Backend**
  - Manages user auth, MongoDB storage, and web services.
- **FastAPI ML Service**
  - Handles forecasting, portfolio recommendation, and chatbot endpoint logic.

---

## 🏗️ Architecture Overview

React Frontend
↕ (REST via Axios)
Express.js Backend (Auth + MongoDB)
↕ (REST via Axios)
FastAPI ML Service (XGBoost, LSTM, Gemini)
↕ (dfinity/agent)
ICP Canister (On-chain profile via Motoko)


- **Frontend** (React + Tailwind): Dashboard, Chat, Profile UI  
- **Backend** (Node.js + Express): Authentication, user data, wallet operations  
- **Services** (FastAPI): Forecasting, ML recommendations, AI chatbot  
- **Canister** (Motoko): Blockchain-based user profile storage  
- **Scripts & Deployment**: Includes DFX + deployment logic  

---

## 🛠️ Local Development

### Prerequisites
- Node.js ≥ 18  
- Python ≥ 3.9  
- MongoDB (local or Atlas)  
- DFINITY `dfx` CLI  

### Setup Steps

1. **Frontend**
   ```bash
   cd frontend
   npm install
   npm run dev
Backend

cd backend
npm install
node server.js
AI Services (FastAPI)

cd services
pip install -r requirements.txt
uvicorn main:app --reload
ICP Canister

cd icp_canister
dfx start --background
dfx deploy
📊 Machine Learning Modules
Forecasting: ARIMA, LSTM, and XGBoost

Recommender: Clustering + Particle Swarm Optimization (PSO)

Chatbot: Google Gemini Integration

Simulator: Automated trade logic with EMA & RSI technical indicators

🛡️ License & Acknowledgements
Licensed under MIT – LICENSE

Thanks to:

DFINITY Foundation

Google Gemini API

DoraHacks WCHL 2025

Plug Wallet SDK 
github.com

📚 Contributing
Fork this repository

Create a feature branch (git checkout -b feature/my-feature)

Commit your changes (git commit -m "Add my feature")

Push the branch (git push origin feature/my-feature)

Open a Pull Request

📧 Contact
For questions or support, feel free to reach out or open an issue.

Built for the WCHL 2025 Qualification Round
