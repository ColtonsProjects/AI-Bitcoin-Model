import React, { useEffect, useState } from 'react';
import '../styles/analytics.css';

const MAIN_SERVER_URL = 'https://ai-automated-outreach-5c4ade0e7f79.herokuapp.com/api/campaigns';





const Analytics = () => {
    const [campaigns, setCampaigns] = useState([]);
    const [selectedCampaign, setSelectedCampaign] = useState(null);

    // Fetch campaign data from the backend API
    const fetchCampaignData = async () => {
        try {
            const response = await fetch(MAIN_SERVER_URL);
            if (!response.ok) throw new Error('Failed to fetch campaign data');
            const data = await response.json();

            const formattedCampaigns = data.map(campaign => ({
                id: campaign._id,
                name: campaign.name,
                leads: campaign.customers ? campaign.customers.length : 0,
                contacts: campaign.customers ? campaign.customers.map(customer => ({
                    name: customer.name || 'Unknown',
                    email: customer.email || '',
                    phone: customer.phoneNumber,
                    rating: customer.ranking,
                    callOverview: customer.callSummary,
                    clientInfo: customer.extraInfo,
                    time: customer.duration
                })) : []
            }));

            setCampaigns(formattedCampaigns);
        } catch (error) {
            console.error('Error fetching campaign data:', error);
        }
    };

    useEffect(() => {
        fetchCampaignData();
    }, []);

    const handleCampaignClick = (campaign) => {
        setSelectedCampaign(selectedCampaign === campaign ? null : campaign);
    };

    const formatDuration = (seconds) => {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    };

    return (
        <div>
            <h1>Campaign Analytics</h1>
            <div className="campaigns-container">
                {campaigns.map(campaign => (
                    <div 
                        key={campaign.id} 
                        className="card campaign-card" 
                        onClick={() => handleCampaignClick(campaign)}
                    >
                        <h2>{campaign.name}</h2>
                        <p>Average Call Time: {formatDuration(
                            campaign.contacts.reduce((total, contact) => total + (contact.time || 0), 0) / campaign.contacts.length || 0
                        )}</p>
                        <p>Success Rate: {Math.round(
                            (campaign.contacts.filter(contact => contact.rating >= 3).length / campaign.contacts.length) * 100
                        ) || 0}%</p>
                        <p>Number of Leads: {campaign.contacts.filter(contact => contact.rating >= 3).length}</p>

                        {selectedCampaign === campaign && (
                            <div className="campaign-details">
                                <h3>Contacts</h3>
                                <ul className="lead-list">
                                    {campaign.contacts.sort((a, b) => b.rating - a.rating).map((contact, index) => (
                                        <li
                                            key={index}
                                            className={`lead-item ${contact.rating >= 3 ? 'gold' : ''} ${contact.rating === 5 ? 'five-star' : ''}`}
                                        >
                                            <div className="lead-info">
                                                <p><strong>{contact.name}</strong></p>
                                                <p>{contact.email} | {contact.phone}</p>
                                            </div>
                                            <div className="lead-rating-overview">
                                                <div className="lead-rating">
                                                    {'★'.repeat(contact.rating)}{'☆'.repeat(5 - contact.rating)}
                                                </div>
                                                <div className="lead-overview">
                                                    <strong>Call Overview:</strong> {contact.callOverview}
                                                </div>
                                                <div className="lead-client-info">
                                                    <strong>Client Info:</strong> {contact.clientInfo}
                                                </div>
                                            </div>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Analytics;
