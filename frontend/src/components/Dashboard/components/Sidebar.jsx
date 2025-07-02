import React, { useState, useEffect, useRef } from "react";
import AvatarEditor from "react-avatar-editor";
import { BsPersonLinesFill } from "react-icons/bs";

import { Link, useLocation } from "react-router-dom";
import {
  FaChartLine,
  FaUserCircle,
  FaCog,
  FaClipboardList,
  FaChartBar,
  FaWallet,
  FaUser,
  FaCogs,
  FaCalculator,
  FaBalanceScale,
} from "react-icons/fa";
import { Modal, Button } from "react-bootstrap";
import { useAuthContext } from "../../../../context/AuthContext";
import axios from "axios";
import "../../styles/Sidebar.css";

const Sidebar = () => {
  const { state } = useAuthContext();
  const { user, isAuthenticated } = state;
  const location = useLocation();

  const [profilePhoto, setProfilePhoto] = useState(user?.profilePhoto || null);
  const [image, setImage] = useState(null);
  const [scale, setScale] = useState(1.2);
  const [rotate, setRotate] = useState(0);
  const [isEditing, setIsEditing] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [error, setError] = useState("");

  const editorRef = useRef(null);

  useEffect(() => {
    const savedPhoto = localStorage.getItem("profilePhoto");
    if (savedPhoto) {
      setProfilePhoto(savedPhoto);
    } else if (user?.profilePhoto) {
      setProfilePhoto(user.profilePhoto);
    }
  }, [user]);

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImage(file);
      setError("");
    }
  };

  const handleSave = async () => {
    if (editorRef.current) {
      const canvas = editorRef.current.getImageScaledToCanvas();
      const dataUrl = canvas.toDataURL();

      try {
        const blob = await fetch(dataUrl).then((res) => res.blob());
        const formData = new FormData();
        formData.append("photoFile", blob, "profile-photo.png");

        const response = await axios.put(
          `http://localhost:4000/api/users/update/${user._id}`,
          formData,
          { headers: { "Content-Type": "multipart/form-data" } }
        );

        const updatedPhoto = response.data.user.profilePhoto;
        setProfilePhoto(updatedPhoto);
        localStorage.setItem("profilePhoto", updatedPhoto);
        setIsEditing(false);
      } catch (error) {
        console.error("Error uploading profile photo:", error);
        setError("Failed to upload profile photo.");
      }
    }
  };

  const linkClass = (path) =>
    `flex items-center gap-2 p-2 rounded-lg ${
      location.pathname === path
        ? "bg-teal-600 text-white"
        : "hover:text-teal-400"
    }`;

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h2>
          {isAuthenticated && user ? `Hello, ${user.username}` : "Welcome"}
        </h2>

        {isAuthenticated && (
          <div className="profile-photo-section">
            {profilePhoto ? (
              <img
                src={`http://localhost:4000/users/${profilePhoto}`}
                alt="Profile"
                className="profile-photo"
              />
            ) : (
              <FaUserCircle size={80} />
            )}
            <Button
              variant="secondary"
              onClick={() => setIsEditing(!isEditing)}
            >
              Edit Photo
            </Button>

            {isEditing && (
              <>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageChange}
                />
                {image && (
                  <>
                    <AvatarEditor
                      ref={editorRef}
                      image={image}
                      width={150}
                      height={150}
                      border={20}
                      borderRadius={100}
                      scale={scale}
                      rotate={rotate}
                    />
                    <div>
                      <input
                        type="range"
                        min="1"
                        max="3"
                        step="0.1"
                        value={scale}
                        onChange={(e) =>
                          setScale(parseFloat(e.target.value))
                        }
                      />
                      <Button onClick={() => setRotate((r) => r + 90)}>
                        Rotate
                      </Button>
                      <Button variant="success" onClick={handleSave}>
                        Save
                      </Button>
                    </div>
                  </>
                )}
              </>
            )}
            {error && <p className="error-message">{error}</p>}
          </div>
        )}
      </div>

      <Modal show={showPreview} onHide={() => setShowPreview(false)} centered>
        <Modal.Header closeButton>
          <Modal.Title>Profile Photo Preview</Modal.Title>
        </Modal.Header>
        <Modal.Body className="text-center">
          <img
            src={`http://localhost:4000/users/${profilePhoto}`}
            alt="Profile Preview"
            className="img-fluid"
            style={{ maxWidth: "100%", borderRadius: "50%" }}
          />
        </Modal.Body>
      </Modal>

      <ul className="sidebar-menu">
        <li>
          <Link to="/dashboard" className={linkClass("/dashboard")}>
            <FaChartBar /> Dashboard
          </Link>
        </li>
        <li>
          <Link to="/analytics">
            <FaChartLine /> Analytics
          </Link>
        </li>
        <li>
          <Link to="/InvestmentCard" className={linkClass("/InvestmentCard")}>
            <FaWallet /> Investments
          </Link>
        </li>
        <li>
          <Link to="/profile" className={linkClass("/profile")}>
            <FaUser /> Profile
          </Link>
        </li>
        <li>
          <Link to="/Settings" className={linkClass("/settings")}>
            <FaCogs /> Settings
          </Link>
        </li>
        <li>
          <Link to="/AIChat">
            <FaCog /> AIChat
          </Link>
        </li>
        <li>
          <Link to="/statistics">
            <FaChartLine /> Statistics
          </Link>
        </li>
        <li>
          <Link to="/currency-converter">
            <FaChartLine /> Currency Converter
          </Link>
        </li>
        
        
          <li>
    <Link to="/profile-card">
      <BsPersonLinesFill className="sidebar-icon" />
      View Profile
    </Link>
  </li>
  <li>
  <Link to="/expenses">
    <i className="sidebar-icon">💸</i>
    Expense Tracker
  </Link>
</li>
<li>
  <Link to="/finance">
    <i className="sidebar-icon">📊</i>
    Finance Tools
  </Link>
</li>
<li>
  <Link to="/chatbot">
    <i className="sidebar-icon">🤖</i>
    AI Chatbot
  </Link>
</li>
      </ul>
    </div>
  );
};

export default React.memo(Sidebar);
