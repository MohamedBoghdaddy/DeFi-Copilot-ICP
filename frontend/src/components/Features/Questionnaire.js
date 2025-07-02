import React, { useState, useEffect } from "react";
import {
  Container,
  Card,
  Form,
  Button,
  Alert,
  ProgressBar,
} from "react-bootstrap";
import { useAuthContext } from "../../../context/AuthContext";
import axios from "axios";
import "../styles/questionnaire.css";

const Questionnaire = () => {
  const { state } = useAuthContext();
  const { user } = state;
  const [currentStep, setCurrentStep] = useState(1);
  const [formData, setFormData] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  const questions = {
    1: {
      question: "What is your monthly income?",
      type: "number",
      field: "monthlyIncome",
    },
    2: {
      question: "What are your monthly expenses?",
      type: "number",
      field: "monthlyExpenses",
    },
    3: {
      question: "What is your current savings balance?",
      type: "number",
      field: "currentSavings",
    },
    4: {
      question: "What is your target savings goal?",
      type: "number",
      field: "savingsGoal",
    },
    5: {
      question: "What is your risk tolerance?",
      type: "select",
      field: "riskTolerance",
      options: ["Conservative", "Moderate", "Aggressive"],
    },
    6: {
      question: "What is your investment timeline?",
      type: "select",
      field: "investmentTimeline",
      options: [
        "Short-term (1-3 years)",
        "Medium-term (3-5 years)",
        "Long-term (5+ years)",
      ],
    },
  };

  const totalSteps = Object.keys(questions).length;

  useEffect(() => {
    const fetchLatestQuestionnaire = async () => {
      try {
        const response = await axios.get("/api/questionnaire/latest", {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        });
        if (response.data) {
          setFormData(response.data);
          setSuccess(true);
        }
      } catch (error) {
        console.error("Error fetching questionnaire:", error);
      }
    };

    if (user) {
      fetchLatestQuestionnaire();
    }
  }, [user]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleNext = () => {
    if (currentStep < totalSteps) {
      setCurrentStep((prev) => prev + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 1) {
      setCurrentStep((prev) => prev - 1);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);

    try {
      await axios.post("/api/questionnaire/submit", formData, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("token")}`,
        },
      });
      setSuccess(true);
    } catch (error) {
      setError(
        error.response?.data?.message || "Error submitting questionnaire"
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  if (success) {
    return (
      <Container className="questionnaire-container">
        <Card className="questionnaire-card">
          <Card.Body>
            <Alert variant="success">
              <h4>Thank you for completing the questionnaire!</h4>
              <p>
                Your responses have been saved. We'll use this information to
                provide personalized financial advice.
              </p>
            </Alert>
            <Button
              variant="primary"
              onClick={() => {
                setSuccess(false);
                setCurrentStep(1);
              }}
            >
              Update Responses
            </Button>
          </Card.Body>
        </Card>
      </Container>
    );
  }

  return (
    <Container className="questionnaire-container">
      <Card className="questionnaire-card">
        <Card.Header>
          <h3>Financial Questionnaire</h3>
          <ProgressBar
            now={(currentStep / totalSteps) * 100}
            label={`${Math.round((currentStep / totalSteps) * 100)}%`}
          />
        </Card.Header>
        <Card.Body>
          <Form onSubmit={handleSubmit}>
            {error && <Alert variant="danger">{error}</Alert>}

            <div className="question-container">
              <h4>{questions[currentStep].question}</h4>

              {questions[currentStep].type === "number" ? (
                <Form.Group className="mb-3">
                  <Form.Control
                    type="number"
                    name={questions[currentStep].field}
                    value={formData[questions[currentStep].field] || ""}
                    onChange={handleInputChange}
                    required
                    min="0"
                    step="0.01"
                  />
                </Form.Group>
              ) : (
                <Form.Group className="mb-3">
                  <Form.Select
                    name={questions[currentStep].field}
                    value={formData[questions[currentStep].field] || ""}
                    onChange={handleInputChange}
                    required
                  >
                    <option value="">Select an option</option>
                    {questions[currentStep].options.map((option, index) => (
                      <option key={index} value={option}>
                        {option}
                      </option>
                    ))}
                  </Form.Select>
                </Form.Group>
              )}
            </div>

            <div className="button-group">
              {currentStep > 1 && (
                <Button
                  variant="secondary"
                  onClick={handlePrevious}
                  disabled={isSubmitting}
                >
                  Previous
                </Button>
              )}

              {currentStep < totalSteps ? (
                <Button
                  variant="primary"
                  onClick={handleNext}
                  disabled={isSubmitting}
                >
                  Next
                </Button>
              ) : (
                <Button variant="success" type="submit" disabled={isSubmitting}>
                  {isSubmitting ? "Submitting..." : "Submit"}
                </Button>
              )}
            </div>
          </Form>
        </Card.Body>
      </Card>
    </Container>
  );
};

export default Questionnaire;
