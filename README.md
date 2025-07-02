# 🧠 DeFi Copilot – Decentralized AI Financial Advisor on ICP

**Built for the World Computer Hacker League (WCHL) 2025 – Qualification Round**  
Track: 🧠 **Decentralized AI** | Tags: ICP · AI · Motoko · Web3 · Chain Fusion · BTC

---

## 🔥 Project Summary

**DeFi Copilot** is a decentralized financial advisor built on the Internet Computer Protocol (ICP) that combines on-chain identity, AI-powered market analysis, and real-time forecasting for informed personal investment. It features a Gemini-powered AI chatbot, on-chain user profile storage, and a fullstack system integrating React, Express.js, FastAPI, and Motoko canisters.

---

## 🧩 Features

- 🔐 Wallet-based Login (Internet Identity, Plug)
- 🧠 **AI Chatbot** (Google Gemini API) for financial guidance
- 📊 **Forecasting Models**: ARIMA, XGBoost, LSTM
- 📁 User profiles stored on **ICP via Motoko canister**
- 💼 **Portfolio Recommendations** based on risk/goals
- 🏦 Mock DeFi Simulator using technical indicators (EMA, RSI)
- 🔄 Express + FastAPI microservice architecture
- 🌐 Frontend with React + Tailwind
- 🔗 ICP Mainnet deployable (with `dfx.json` & Canister IDs)
- 🧪 Open source licensed, testable, and extensible

---

## 🛠️ Architecture

React Frontend (Plug Wallet, Web UI)
↕ REST
Node.js Backend (Auth, MongoDB)
↕ REST
FastAPI Service (Gemini AI + Forecasting Engine)
↕ dfx agent
ICP Motoko Canister (on-chain profile storage)



---

## 🚀 Local Development Setup

### Prerequisites
- Node.js ≥ 18  
- Python ≥ 3.9  
- MongoDB  
- `dfx` CLI  
- Plug Wallet or Internet Identity for login

### Setup Steps

#### 1️⃣ Frontend
```
cd frontend
npm install
npm run dev
2️⃣ Backend
bash
Copy code
cd backend
npm install
node server.js
3️⃣ FastAPI ML Services
```
cd services
pip install -r requirements.txt
uvicorn main:app --reload
4️⃣ ICP Canister

cd icp_canister
dfx start --background
dfx deploy
🤖 AI Modules
Gemini API for financial Q&A

ARIMA, LSTM, XGBoost models for forecasting

PSO-based recommender for goal-based portfolios

Chat intent + sentiment parsing with fallback logic

🎥 Demo Video
🔗 Demo Walkthrough Video (10 mins)
🎯 Architecture Overview → Code Walkthrough → Live Demo
```
📸 Screenshots


🧱 Tech Stack
Frontend: React + Tailwind

Backend: Express.js + MongoDB

AI Services: FastAPI (Python), Gemini API

Blockchain: Motoko ICP Canister

Wallet: Plug + Internet Identity

Data: Real-time financial APIs + Gemini response modeling

📄 Submission Checklist (✅ Ready)
 GitHub repo with source code

 dfx.json included with working ICP deploy

 Architecture & project summary in README

 Gemini AI + Forecasting + Recommender integration

 Screenshots and architecture diagram

 Demo video (voice-over or subtitles)

 MIT License included

 On-chain ICP storage working with canister IDs

 Clear local setup instructions
---
```
🗺️ Future Roadmap
☁️ Deploy all services via Docker & ICP boundary nodes

🧬 On-chain encrypted financial history (anonymized learning)

🏛️ DAO-based governance on portfolio strategies

🔐 CKBTC support for real DeFi interactions

📱 Mobile-first responsive dApp UI

📜 License
This project is licensed under the MIT License.
---

🤝 Contributing
Fork this repo

Create a new branch (git checkout -b feature/my-feature)

Commit your changes (git commit -m "Add feature")

Push to the branch (git push origin feature/my-feature)

Open a Pull Request

🏆 About WCHL 2025
Built for the World Computer Hacker League (WCHL) – ICP Qualification Round.
Hack your way into Web3 history! 🧠🌐
```
Made with ❤️ by Mohamed Boghdaddy & Team for WCHL 2025
