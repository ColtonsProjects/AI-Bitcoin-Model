// src/pages/Dashboard.js
import React, { useState, useEffect } from 'react';
import { Bar } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';
import CampaignManagerRedesign from '../components/CampaignManager';
import {
    fetchCampaigns,
    createCampaign,
    updateCampaign,
    deleteCampaign,
    startCampaign,
} from '../services/campaignService';
import '../styles/dashboard.css';
import { useNavigate } from 'react-router-dom';


ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const Dashboard = () => {
    const [campaigns, setCampaigns] = useState([]);
    const [chartData, setChartData] = useState(null);
    const [editingCampaign, setEditingCampaign] = useState(null);
    const navigate = useNavigate();

    useEffect(() => {
        const loadCampaigns = async () => {
            try {
                const data = await fetchCampaigns();
                setCampaigns(data || []); // Ensure campaigns is always an array
                prepareChartData(data || []);
            } catch (error) {
                console.error('Error loading campaigns:', error);
            }
        };
        loadCampaigns();
    }, []);

    const prepareChartData = (campaignsData) => {
        if (!campaignsData || campaignsData.length === 0) return; // Ensure campaignsData is defined and non-empty

        const labels = campaignsData.map((c) => c?.name || "Untitled Campaign");
        
        const totalCalls = campaignsData.map((c) => (c?.customers ? c.customers.length : 0));
        const warmLeads = campaignsData.map((c) => {
            if (!c?.customers) return 0;
            return c.customers.filter((customer) => customer.ranking >= 3).length;
        });
        const conversions = campaignsData.map((c) => {
            if (!c?.customers) return 0;
            return c.customers.filter((customer) => customer.ranking === 5).length;
        });

        setChartData({
            labels,
            datasets: [
                {
                    label: 'Total Calls',
                    data: totalCalls,
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                },
                {
                    label: 'Warm Leads',
                    data: warmLeads,
                    backgroundColor: 'rgba(255, 159, 64, 0.6)',
                },
                {
                    label: 'Conversions',
                    data: conversions,
                    backgroundColor: 'rgba(153, 102, 255, 0.6)',
                },
            ],
        });
    };

    const handleCampaignSubmit = async (newCampaign) => {
        try {
            if (editingCampaign) {
                // Update the campaign if editing
                const updatedCampaign = await updateCampaign(editingCampaign._id, newCampaign);
                setCampaigns((prev) =>
                    prev.map((campaign) =>
                        campaign._id === editingCampaign._id ? updatedCampaign : campaign
                    )
                );
                setEditingCampaign(null); // Clear editing state
            } else {
                // Create a new campaign if not editing
                const createdCampaign = await createCampaign(newCampaign);
                setCampaigns((prev) => [...prev, createdCampaign]);
            }
            prepareChartData(campaigns || []);
            window.location.reload(); // Refresh page after submission
        } catch (error) {
            console.error('Error creating or updating campaign:', error);
        }
    };

    const handleCampaignEdit = (campaign) => {
        setEditingCampaign(campaign);
    };

    const handleCampaignDelete = async (campaignId) => {
        try {
            await deleteCampaign(campaignId);
            setCampaigns((prev) => prev.filter((campaign) => campaign?._id !== campaignId));
            prepareChartData(campaigns || []);
            window.location.reload();
        } catch (error) {
            console.error('Error deleting campaign:', error);
        }
    };

    const handleStartCampaign = async (campaignId) => {
        try {
            await startCampaign(campaignId);
            alert('Campaign started successfully!');
        } catch (error) {
            console.error('Error starting campaign:', error);
            alert('An error occurred while starting the campaign');
        }
    };

    return (
        <div>
            <h1>AI Outbound Calling Dashboard</h1>
            
            <div className="card">
                <h2>Campaign Performance</h2>
                {chartData && (
                    <Bar
                        data={chartData}
                        options={{
                            responsive: true,
                            plugins: {
                                legend: { position: 'top' },
                                title: {
                                    display: true,
                                    text: 'Campaign Performance Metrics',
                                },
                            },
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    title: {
                                        display: true,
                                        text: 'Count',
                                    },
                                },
                                x: {
                                    title: {
                                        display: true,
                                        text: 'Campaign Names',
                                    },
                                },
                            },
                        }}
                    />
                )}
            <button
                onClick={() => navigate('/analytics')}
                className="call-analytics-button"
                style={{
                    marginTop: '20px',
                    padding: '10px 20px',
                    backgroundColor: '#e94560',
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                    cursor: 'pointer',
                    transition: '0.3s ease',
                }}
                onMouseEnter={(e) => (e.target.style.backgroundColor = '#ff6363')}
                onMouseLeave={(e) => (e.target.style.backgroundColor = '#e94560')}
            >
                View Call Analytics
            </button>
            </div>

            <CampaignManagerRedesign
                onCampaignSubmit={handleCampaignSubmit}
                editingCampaign={editingCampaign}
            />

            <h2>Active Campaigns</h2>
            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Opening Line</th>
                        <th>Objectives</th>
                        <th>Number of Contacts</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {campaigns && campaigns.length > 0 ? (
                        campaigns.map((campaign) => (
                            <tr key={campaign?._id}>
                                <td>{campaign?.name}</td>
                                <td>{campaign?.openingLine}</td>
                                <td>
                                    <ul>
                                        {(campaign?.objectives || []).map((obj, i) => (
                                            <li key={i}>{obj}</li>
                                        ))}
                                    </ul>
                                </td>
                                <td>{campaign?.phoneNumbers?.length || 0}</td>
                                <td>
                                    <button onClick={() => handleStartCampaign(campaign?._id)}>
                                        Start Campaign
                                    </button>
                                    <button onClick={() => handleCampaignEdit(campaign)}>Edit</button>
                                    <button onClick={() => handleCampaignDelete(campaign?._id)}>
                                        Delete
                                    </button>
                                </td>
                            </tr>
                        ))
                    ) : (
                        <tr>
                            <td colSpan="5">No campaigns available</td>
                        </tr>
                    )}
                </tbody>
            </table>
            
        </div>

        
    );
};

export default Dashboard;
