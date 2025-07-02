💸 DeFi Copilot – A Decentralized AI Financial Advisor on ICP
DeFi Copilot is an AI-powered, Web3-based financial advisory dApp built on the Internet Computer Protocol (ICP). It combines React, FastAPI, and machine learning with on-chain identity and wallet-based login to deliver personalized investment insights, real-time forecasting, and portfolio simulation.

📍 Built for the WCHL 2025 Qualification Round

🚀 Features
🔐 Wallet-Based Login (Plug Wallet / Internet Identity)

🧠 AI Chatbot (Gemini API powered)

📈 Forecasting Engine for stocks, BTC, gold using ARIMA, XGBoost, LSTM

🎯 Risk & Goal-Based Portfolio Recommender

🪙 On-Chain Profile Storage (via Motoko ICP Canister)

⚖️ DeFi Investment Simulator (mock DEX + test tokens)

🪙 Chain-Key Bitcoin (CKBTC) integration (optional)

🏗️ System Architecture

📁 Folder Structure

DeFi-Copilot-ICP/
├── frontend/         # React + Tailwind UI
│   └── src/pages/    # Dashboard, Chat, Profile
├── backend/          # Express.js API (auth, MongoDB)
│   └── server.js
├── services/      # FastAPI ML Service (XGBoost, Gemini, LSTM)
│   └── main.py
├── icp_canister/     # Motoko storage functions for profiles
│   └── main.mo
├── docs/             # Pitch deck, diagrams, openapi.yaml
├── scripts/          # Deployment scripts
└── README.md
🌐 Service Communication
Source → Target	Protocol	Description
React → Express	REST (Axios)	Auth, user data, wallet ops
React → FastAPI	REST (Axios)	ML forecasts, recommendations, chat
React → ICP Canister	@dfinity/agent	Store/load on-chain user profiles

📦 Local Development Setup
🔧 Prerequisites
Node.js 18+

Python 3.9+

MongoDB (local or Atlas)

dfx CLI

🔌 Frontend

cd frontend
npm install
npm run dev
🔄 Backend (Node + Express)

cd backend
npm install
node server.js
🧠 AI Services (FastAPI)

cd ai_services
pip install -r requirements.txt
uvicorn main:app --reload
🧠 Canister (ICP)

dfx start --background
dfx deploy
📊 Machine Learning Modules
Forecasting: ARIMA, LSTM, XGBoost

Recommendation: Clustering, PSO

Chatbot: Gemini API

Simulation: Trade bot logic (EMA, RSI)

📝 License
Licensed under the MIT License. See the LICENSE file for full text.

🙌 Acknowledgements
DFINITY Foundation

Google Gemini API

DoraHacks WCHL 2025

Plug Wallet SDK