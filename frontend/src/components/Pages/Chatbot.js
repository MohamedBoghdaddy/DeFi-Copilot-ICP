import React, { useState, useContext, useEffect } from "react";
import axios from "axios";
import "../styles/chhat.css";
import { DashboardContext } from "../../../context/DashboardContext";

const API_URL =
  process.env.REACT_APP_API_URL ||
  "http://127.0.0.1:8000" ||
  "http://localhost:8000";

const Chatbot = () => {
  const dashboardContext = useContext(DashboardContext);
  const profile = dashboardContext?.profile || {};
  const fetchProfile = dashboardContext?.actions?.fetchProfile;

  const [goal, setGoal] = useState("investment");
  const [messages, setMessages] = useState([
    {
      sender: "bot",
      text: "Hi! I'm your AI Financial Advisor. What would you like help with?",
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [inputText, setInputText] = useState("");

  // 🔁 Fetch profile if not already available
  useEffect(() => {
    if (!profile || Object.keys(profile).length === 0) {
      fetchProfile?.();
    }
  }, [fetchProfile, profile]);

  const sendProfile = async () => {
    if (!profile || Object.keys(profile).length === 0) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "⚠️ Please complete your profile first." },
      ]);
      return;
    }

    setLoading(true);
    setMessages((prev) => [
      ...prev,
      { sender: "user", text: `Give me a ${goal.replace("_", " ")}.` },
    ]);

    const endpoint = `${API_URL}/chatbot/generate/${
      goal === "life_management" ? "life" : "investment"
    }`;

    try {
      const res = await axios.post(endpoint, profile);
      const formatted = JSON.stringify(res.data, null, 2);
      setMessages((prev) => [...prev, { sender: "bot", text: formatted }]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "❌ Something went wrong while sending the profile.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const sendFreeText = async () => {
    if (!inputText.trim()) return;

    setMessages((prev) => [...prev, { sender: "user", text: inputText }]);
    setLoading(true);

    try {
      const res = await axios.post(`${API_URL}/chatbot/chat`, {
        message: inputText,
        profile: profile,
      });
      const formatted = res.data?.output || "🤖 No response.";
      setMessages((prev) => [...prev, { sender: "bot", text: formatted }]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "❌ Something went wrong while sending your message.",
        },
      ]);
    } finally {
      setLoading(false);
      setInputText("");
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">💬 AI Financial Advisor</div>

      <div className="chat-window">
        {messages.map((msg, index) => (
          <div key={index} className={`chat-bubble ${msg.sender}`}>
            {msg.text}
          </div>
        ))}
        {loading && <div className="chat-bubble bot">Typing...</div>}
      </div>

      <div className="chat-input-area">
        <select value={goal} onChange={(e) => setGoal(e.target.value)}>
          <option value="investment">Investment Plan</option>
          <option value="life_management">Life Management Plan</option>
        </select>
        <button onClick={sendProfile} disabled={loading}>
          Generate Plan
        </button>
      </div>

      <div className="chat-input-area">
        <input
          type="text"
          placeholder="Ask a question..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendFreeText()}
        />
        <button onClick={sendFreeText} disabled={loading}>
          Send
        </button>
      </div>
    </div>
  );
};

export default Chatbot;
