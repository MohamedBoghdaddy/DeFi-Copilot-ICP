import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faCommentDots,
  faPaperPlane,
  faTimes,
  faRobot,
  faCircle,
  faUser,
  faUserTie,
} from "@fortawesome/free-solid-svg-icons";
import "../styles/chat.css";
import logo from "../../../assets/latest_logo.png";

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [typingStage, setTypingStage] = useState(0);
  const [displayedResponse, setDisplayedResponse] = useState("");
  const [fullResponse, setFullResponse] = useState("");
  const [typingTimeout, setTypingTimeout] = useState(null);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const chatMessagesRef = useRef(null);

  // Scroll to bottom of messages container
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Focus input field when chat is opened
  useEffect(() => {
    if (isChatOpen) {
      setTimeout(() => {
        inputRef.current?.focus();
      }, 300);
    }
  }, [isChatOpen]);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages, displayedResponse]);

  // Handle typing effect
  useEffect(() => {
    if (typingStage === 1 && fullResponse) {
      // Calculate typing speed based on message length (faster for shorter messages)
      const messageLength = fullResponse.length;
      const baseDelay = Math.min(80, Math.max(20, 100 - messageLength / 10));

      // Break typing into chunks for more realistic effect
      const typingInterval = setInterval(() => {
        setDisplayedResponse((prev) => {
          if (prev.length < fullResponse.length) {
            // Add 1-4 characters at a time for more natural typing
            const chunkSize = Math.floor(Math.random() * 4) + 1;
            const nextChunk = fullResponse.substring(
              prev.length,
              prev.length + chunkSize
            );
            return prev + nextChunk;
          } else {
            clearInterval(typingInterval);

            // After finishing typing, add message to the chat and reset states
            setTimeout(() => {
              setMessages((prev) => [
                ...prev,
                {
                  text: fullResponse,
                  sender: "bot",
                  timestamp: new Date().toISOString(),
                },
              ]);

              setTypingStage(0);
              setFullResponse("");
              setDisplayedResponse("");
              setIsLoading(false);
            }, 300); // Small delay after typing completes

            return prev;
          }
        });
      }, baseDelay);

      return () => clearInterval(typingInterval);
    }
  }, [typingStage, fullResponse]);

  // Toggle chat visibility
  const toggleChat = () => {
    setIsChatOpen(!isChatOpen);
  };

  // Handle input change
  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  // Handle message submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!inputValue.trim()) return;

    // Add user message
    const userMessage = {
      text: inputValue.trim(),
      sender: "user",
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);

    try {
      // Make API request to backend - ensure we're using the correct endpoint
      const response = await axios.post("/api/chat", {
        message: userMessage.text,
        userId: "anonymous", // Add userId parameter for better context handling
        salary: 60000, // Default salary for financial context
      });

      // Extract bot response text
      const responseText =
        response.data.response ||
        "I'm sorry, I couldn't process your request at this time.";

      // Calculate a simulated "thinking" delay based on message complexity
      const messageLength = userMessage.text.length;
      const responseLength = responseText.length;
      const thinkingTime = Math.min(3000, Math.max(800, messageLength * 50));

      // Set the full response text but don't display immediately
      setFullResponse(responseText);

      // Simulate AI "thinking" with delay before typing starts
      clearTimeout(typingTimeout);
      const timeout = setTimeout(() => {
        setTypingStage(1); // Start typing animation
      }, thinkingTime);

      setTypingTimeout(timeout);
    } catch (error) {
      console.error("Error sending message:", error);

      // Add error message after a thinking delay
      setTimeout(() => {
        setMessages((prev) => [
          ...prev,
          {
            text: "I'm having trouble connecting to my services right now. Could you please try again in a moment?",
            sender: "bot",
            timestamp: new Date().toISOString(),
          },
        ]);
        setIsLoading(false);
      }, 1500);
    }
  };

  // Handle keyboard events (Enter to send, Shift+Enter for new line)
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="chatbot-wrapper">
      {/* Chat toggle button */}
      <button
        className={`chat-toggle-btn ${isChatOpen ? "active" : ""}`}
        onClick={toggleChat}
        aria-label={isChatOpen ? "Close chat" : "Open chat assistant"}
      >
        <FontAwesomeIcon icon={faCommentDots} />
        <span className="chat-toggle-text">Chat with Us</span>
      </button>

      {/* Chat container */}
      <div className={`chatbot-container ${isChatOpen ? "open" : "closed"}`}>
        {/* Chat header */}
        <div className="chat-header">
          <div className="chat-title">
            <div className="chat-avatar">
              <img src={logo} alt="Financial Assistant" />
            </div>
            <div>
              <h5>Financial Assistant</h5>
              <p className="chat-status">
                <FontAwesomeIcon icon={faCircle} className="status-icon" />{" "}
                Online
              </p>
            </div>
          </div>
          <button
            className="chat-close-btn"
            onClick={toggleChat}
            aria-label="Close chat"
          >
            <FontAwesomeIcon icon={faTimes} />
          </button>
        </div>

        {/* Chat messages */}
        <div className="chat-messages-container" ref={chatMessagesRef}>
          {messages.length === 0 ? (
            <div className="chat-welcome">
              <img
                src={logo}
                alt="Financial Assistant"
                className="welcome-image"
              />
              <h4>Welcome to Financial AI Advisor!</h4>
              <p>
                Ask me anything about financial planning, investments, market
                trends, or getting personalized financial advice.
              </p>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div
                key={index}
                className={`message-row ${
                  msg.sender === "user" ? "user-row" : "bot-row"
                }`}
              >
                <div className="message-avatar">
                  <FontAwesomeIcon
                    icon={msg.sender === "user" ? faUser : faUserTie}
                    size="sm"
                  />
                </div>
                <div
                  className={`message ${
                    msg.sender === "user" ? "user-message" : "bot-message"
                  }`}
                >
                  {msg.text}
                </div>
              </div>
            ))
          )}

          {/* Active typing indicator */}
          {isLoading && (
            <div className="message-row bot-row">
              <div className="message-avatar">
                <FontAwesomeIcon icon={faUserTie} size="sm" />
              </div>
              {typingStage === 0 ? (
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              ) : (
                <div className="message bot-message typing-message">
                  {displayedResponse}
                  <span className="typing-cursor"></span>
                </div>
              )}
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input form */}
        <form className="chat-form" onSubmit={handleSubmit}>
          <div className="chat-input-container">
            <textarea
              className="chat-input"
              placeholder="Type your message..."
              value={inputValue}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              ref={inputRef}
              rows="1"
              aria-label="Message input"
              disabled={isLoading}
            ></textarea>
            <button
              className="chat-send-btn"
              type="submit"
              disabled={!inputValue.trim() || isLoading}
              aria-label="Send message"
            >
              <FontAwesomeIcon icon={faPaperPlane} size="sm" />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Chatbot;
