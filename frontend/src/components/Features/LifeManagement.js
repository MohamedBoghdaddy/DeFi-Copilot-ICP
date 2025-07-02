import React, { useState, useEffect } from "react";
import {
  Container,
  Card,
  Form,
  Button,
  Alert,
  ListGroup,
} from "react-bootstrap";
import { useAuthContext } from "../../../context/AuthContext";
import axios from "axios";
import "../styles/lifeManagement.css";

const LifeManagement = () => {
  const { state } = useAuthContext();
  const { user } = state;
  const [formData, setFormData] = useState({
    financialGoals: "",
    careerGoals: "",
    personalGoals: "",
    timeline: "1-3 years",
  });
  const [lifeGoals, setLifeGoals] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    const fetchLifeGoals = async () => {
      try {
        const response = await axios.get(`/api/life-management/${user._id}`, {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        });
        if (response.data) {
          setLifeGoals(response.data);
        }
      } catch (error) {
        console.error("Error fetching life goals:", error);
      }
    };

    if (user) {
      fetchLifeGoals();
    }
  }, [user]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        "/api/life-management",
        {
          ...formData,
          userId: user._id,
        },
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`,
          },
        }
      );

      setLifeGoals(response.data.lifeGoals);
      setSuccess(true);
      setFormData({
        financialGoals: "",
        careerGoals: "",
        personalGoals: "",
        timeline: "1-3 years",
      });
    } catch (error) {
      setError(error.response?.data?.message || "Error generating life goals");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container className="life-management-container">
      <Card className="life-management-card">
        <Card.Header>
          <h3>Life Goals Management</h3>
        </Card.Header>
        <Card.Body>
          {error && <Alert variant="danger">{error}</Alert>}
          {success && (
            <Alert
              variant="success"
              onClose={() => setSuccess(false)}
              dismissible
            >
              Your life goals have been generated successfully!
            </Alert>
          )}

          <Form onSubmit={handleSubmit}>
            <Form.Group className="mb-3">
              <Form.Label>Financial Goals</Form.Label>
              <Form.Control
                as="textarea"
                rows={3}
                name="financialGoals"
                value={formData.financialGoals}
                onChange={handleInputChange}
                placeholder="Describe your financial goals..."
                required
              />
            </Form.Group>

            <Form.Group className="mb-3">
              <Form.Label>Career Goals</Form.Label>
              <Form.Control
                as="textarea"
                rows={3}
                name="careerGoals"
                value={formData.careerGoals}
                onChange={handleInputChange}
                placeholder="Describe your career goals..."
                required
              />
            </Form.Group>

            <Form.Group className="mb-3">
              <Form.Label>Personal Goals</Form.Label>
              <Form.Control
                as="textarea"
                rows={3}
                name="personalGoals"
                value={formData.personalGoals}
                onChange={handleInputChange}
                placeholder="Describe your personal goals..."
                required
              />
            </Form.Group>

            <Form.Group className="mb-3">
              <Form.Label>Timeline</Form.Label>
              <Form.Select
                name="timeline"
                value={formData.timeline}
                onChange={handleInputChange}
                required
              >
                <option value="1-3 years">Short-term (1-3 years)</option>
                <option value="3-5 years">Medium-term (3-5 years)</option>
                <option value="5+ years">Long-term (5+ years)</option>
              </Form.Select>
            </Form.Group>

            <Button
              variant="primary"
              type="submit"
              disabled={isLoading}
              className="submit-button"
            >
              {isLoading ? "Generating Goals..." : "Generate Life Goals"}
            </Button>
          </Form>

          {lifeGoals && (
            <div className="goals-section">
              <h4>Your Generated Life Goals</h4>
              <ListGroup>
                {lifeGoals.split("\n").map(
                  (goal, index) =>
                    goal.trim() && (
                      <ListGroup.Item key={index} className="goal-item">
                        {goal}
                      </ListGroup.Item>
                    )
                )}
              </ListGroup>
            </div>
          )}
        </Card.Body>
      </Card>
    </Container>
  );
};

export default LifeManagement;
