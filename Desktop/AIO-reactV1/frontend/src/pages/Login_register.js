"use client"

import React, { useState } from "react"
import { Moon, Sun } from "lucide-react"

export default function Component() {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState("login")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [message, setMessage] = useState("")
  const [isDarkMode, setIsDarkMode] = useState(true)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setMessage("")

    //change these in developement 
    // const baseUrl = process.env.REACT_APP_MAIN_SERVER_URL;
    const baseUrl = "https://ai-automated-outreach-5c4ade0e7f79.herokuapp.com/api/login"; // Temporarily hardcode

    
    const url = activeTab === "login" ? `https://ai-automated-outreach-5c4ade0e7f79.herokuapp.com/api/login` : `https://ai-automated-outreach-5c4ade0e7f79.herokuapp.com/api/register`;
    console.log("--------------------------" + url + "---------------------")
    const body = activeTab === "register" ? { email, password, confirmPassword } : { email, password }
    console.log(url)
    if (activeTab === "register" && password !== confirmPassword) {
      setMessage("Passwords do not match")
      setLoading(false)
      return
    }

    try {
    let response;
    if(activeTab === "register"){
        response = await fetch("https://ai-automated-outreach-5c4ade0e7f79.herokuapp.com/api/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      })
    }
    else
    {
        response = await fetch("https://ai-automated-outreach-5c4ade0e7f79.herokuapp.com/api/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      })
    }
      
      const data = await response.json()
      if (!response.ok) throw new Error(data.error)

      setMessage(data.message || "Success!")
      if (activeTab === "login") {
        console.log("Token:", data.token)
      }
    } catch (error) {
      setMessage(error.message)
    } finally {
      setLoading(false)
    }
  }

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode)
  }

  return (
    <div className={`w-screen h-screen flex items-center justify-center`}>
      <div className={`w-full max-w-md rounded-lg shadow-lg overflow-hidden transition-colors duration-300 ease-in-out ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
        <div className="flex justify-between items-center px-6 py-4">
          <div className="flex justify-between relative w-5/6">
            <button
              className={`py-2 w-5/6 px-4 text-sm font-medium  transition-colors duration-300 ease-in-out ${
                activeTab === "login"
                  ? isDarkMode ? 'bg-transparent' : 'bg-customGray'
                  : isDarkMode ? 'text-gray-400 hover:text-gray-200 bg-transparent' : 'text-gray-500 hover:text-gray-700 bg-transparent'
              }`}
              onClick={() => setActiveTab("login")}
            >
              Login
            </button>
            <button
              className={`py-2 w-5/6 px-4 text-sm font-medium transition-colors duration-300 ease-in-out ${
                activeTab === "register"
                  ? isDarkMode ? 'bg-transparent' : 'bg-customGray'
                  : isDarkMode ? 'text-gray-400 hover:text-gray-200 bg-transparent' : 'text-gray-500 hover:text-gray-700 bg-customGray'
              }`}
              onClick={() => setActiveTab("register")}
            >
              Register
            </button>
            <div 
              className={`absolute bottom-0 h-0.5 bg-blue-500 transition-all duration-300 ease-in-out ${
                activeTab === "login" ? "left-0 w-1/2" : "left-1/2 w-1/2"
              }`}
            />
          </div>
          <button
            onClick={toggleDarkMode}
            className={`p-2 w-10 rounded-full transition-colors duration-300 ease-in-out ${isDarkMode ? 'bg-gray-700 text-yellow-300' : 'bg-gray-200 text-gray-800'}`}
            aria-label={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
          >
            {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
        </div>
        <div className="p-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="email" className={`block text-sm font-medium transition-colors duration-300 ease-in-out ${isDarkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                Email
              </label>
              <input
                id="email"
                type="email"
                required
                className={`mt-1 block w-full px-3 py-2 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-300 ease-in-out ${
                  isDarkMode
                    ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400'
                    : 'bg-gray-50 border-gray-300 text-gray-900 placeholder-gray-500'
                }`}
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <label htmlFor="password" className={`block text-sm font-medium transition-colors duration-300 ease-in-out ${isDarkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                Password
              </label>
              <input
                id="password"
                type="password"
                required
                className={`mt-1 block w-full px-3 py-2 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-300 ease-in-out ${
                  isDarkMode
                    ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400'
                    : 'bg-gray-50 border-gray-300 text-gray-900 placeholder-gray-500'
                }`}
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
            {activeTab === "register" && (
              <div className="space-y-2">
                <label htmlFor="confirm-password" className={`block text-sm font-medium transition-colors duration-300 ease-in-out ${isDarkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                  Confirm Password
                </label>
                <input
                  id="confirm-password"
                  type="password"
                  required
                  className={`mt-1 block w-full px-3 py-2 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-300 ease-in-out ${
                    isDarkMode
                      ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400'
                      : 'bg-gray-50 border-gray-300 text-gray-900 placeholder-gray-500'
                  }`}
                  placeholder="Confirm your password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                />
              </div>
            )}
            <button
              type="submit"
              className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-all duration-300 ease-in-out transform hover:scale-105 ${
                loading ? "opacity-50 cursor-not-allowed" : ""
              }`}
              disabled={loading}
            >
              {loading
                ? activeTab === "login" ? "Signing in..." : "Creating account..."
                : activeTab === "login" ? "Sign in" : "Create account"}
            </button>
          </form>
          {message && (
            <p className={`mt-4 text-sm transition-colors duration-300 ease-in-out ${
              message.includes("Success")
                ? isDarkMode ? 'text-green-400' : 'text-green-600'
                : isDarkMode ? 'text-red-400' : 'text-red-600'
            }`}>
              {message}
            </p>
          )}
        </div>
      </div>
      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .space-y-2 {
          animation: fadeIn 0.3s ease-out;
        }
      `}</style>
    </div>
  )
}