import React, { useState, useEffect, useRef } from "react";
import { Container, Card, Form, Button, Spinner, Alert } from "react-bootstrap";
import { useAuthContext } from "../../../context/AuthContext";
import axios from "axios";
import "../styles/chat.css";

const Chat = () => {
  const { state } = useAuthContext();
  const { user } = state;
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [context, setContext] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const fetchChatHistory = async () => {
      try {
        const response = await axios.get("/api/chat/history", {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        });
        setMessages(response.data);
      } catch (error) {
        console.error("Error fetching chat history:", error);
      }
    };

    const fetchChatContext = async () => {
      try {
        const response = await axios.get(`/api/chat/context/${user._id}`, {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        });
        setContext(response.data);
      } catch (error) {
        console.error("Error fetching chat context:", error);
      }
    };

    if (user) {
      fetchChatHistory();
      fetchChatContext();
    }
  }, [user]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = {
      type: "user",
      content: input,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        "/api/chat",
        {
          message: input,
          userId: user._id,
        },
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        }
      );

      const aiMessage = {
        type: "ai",
        content: response.data.response,
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
      setError(
        "Sorry, I'm having trouble processing your request. Please try again."
      );
      setMessages((prev) => [
        ...prev,
        {
          type: "error",
          content:
            "Sorry, there was an error processing your message. Please try again.",
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatMessage = (content) => {
    // Convert markdown-style links to clickable links
    const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
    const formattedContent = content.replace(
      linkRegex,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );

    // Convert newlines to <br> tags
    return formattedContent.split("\n").map((line, i) => (
      <React.Fragment key={i}>
        {i > 0 && <br />}
        <span dangerouslySetInnerHTML={{ __html: line }} />
      </React.Fragment>
    ));
  };

  return (
    <Container className="chat-container">
      <Card className="chat-card">
        <Card.Header className="chat-header">
          <h3>AI Financial Advisor</h3>
          {context?.contextSummary && (
            <div className="context-summary">
              <small>{context.contextSummary}</small>
            </div>
          )}
        </Card.Header>
        <Card.Body className="chat-body">
          <div className="messages-container">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`message ${
                  message.type === "user" ? "user-message" : "ai-message"
                }`}
              >
                <div className="message-content">
                  {formatMessage(message.content)}
                </div>
                <div className="message-timestamp">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message ai-message">
                <div className="message-content">
                  <Spinner animation="border" size="sm" /> Thinking...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </Card.Body>
        <Card.Footer className="chat-footer">
          {error && (
            <Alert variant="danger" className="mb-3">
              {error}
            </Alert>
          )}
          <Form onSubmit={handleSubmit}>
            <div className="input-group">
              <Form.Control
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask me about investments, savings, or financial advice..."
                disabled={isLoading}
              />
              <Button
                type="submit"
                variant="primary"
                disabled={isLoading || !input.trim()}
              >
                Send
              </Button>
            </div>
          </Form>
        </Card.Footer>
      </Card>
    </Container>
  );
};

export default Chat;
