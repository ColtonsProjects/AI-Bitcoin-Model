import React from 'react';
import './styles/tailwind.css';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import './styles/global.css';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';
import Login from './pages/Login_register'
function App() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/analytics" element={<Analytics />} />
                <Route path="/login" element={<Login />} />
            </Routes>
        </Router>
    );
}

export default App;
