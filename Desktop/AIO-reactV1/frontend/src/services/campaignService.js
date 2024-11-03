const MAIN_SERVER_URL = 'https://ai-automated-outreach-5c4ade0e7f79.herokuapp.com/api/campaigns';


// Fetch all campaigns
export const fetchCampaigns = async () => {
    const response = await fetch(MAIN_SERVER_URL);
    if (!response.ok) throw new Error('Failed to fetch campaigns');
    return response.json();
};

// Create a new campaign
export const createCampaign = async (campaignData) => {
    const response = await fetch(MAIN_SERVER_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(campaignData),
    });
    if (!response.ok) throw new Error('Failed to create campaign');
    return response.json();
};

// Update an existing campaign
export const updateCampaign = async (campaignId, campaignData) => {
    const response = await fetch(`${MAIN_SERVER_URL}/${campaignId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(campaignData),
    });
    if (!response.ok) throw new Error('Failed to update campaign');
    return response.json();
};

// Delete a campaign
export const deleteCampaign = async (campaignId) => {
    const response = await fetch(`${MAIN_SERVER_URL}/${campaignId}`, {
        method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to delete campaign');
    return response.json();
};

// Start a campaign
export const startCampaign = async (campaignId) => {
    const response = await fetch(`${MAIN_SERVER_URL}/${campaignId}/start`, {
        method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to start campaign');
    return response.json();
};
