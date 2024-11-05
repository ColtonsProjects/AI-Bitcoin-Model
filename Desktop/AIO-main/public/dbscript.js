console.log('dbscript.js loaded');

let campaigns = [];
let chart;
let isEditing = false;
let editingCampaignId = null;

function addObjective() {
    const objectives = document.getElementById('objectives');
    const objectiveContainer = document.createElement('div');
    objectiveContainer.className = 'objective-container';
    
    const newObjective = document.createElement('input');
    newObjective.type = 'text';
    newObjective.placeholder = `Objective ${objectives.children.length + 1}`;
    
    const deleteButton = document.createElement('button');
    deleteButton.textContent = 'Delete';
    deleteButton.className = 'delete-objective';
    deleteButton.onclick = function() {
        objectives.removeChild(objectiveContainer);
    };
    
    objectiveContainer.appendChild(newObjective);
    objectiveContainer.appendChild(deleteButton);
    objectives.appendChild(objectiveContainer);
}


// --------------------------------- CREATE/ UPDATE CAMPAIGN ---------------------------------  

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded');

    const createCampaignBtn = document.getElementById('createCampaignBtn');
    const saveCampaignBtn = document.getElementById('saveCampaignBtn');
    console.log('Create Campaign button:', createCampaignBtn);

    // Fetch campaigns when the page loads
    fetchCampaigns();

    async function handleCampaign(event) {
        event.preventDefault();

        const name = document.getElementById('campaignName').value;
        const openingLine = document.getElementById('openingLine').value;
        const objectiveInputs = document.querySelectorAll('#objectives input');
        const objectives = Array.from(objectiveInputs).map(input => input.value).filter(Boolean);
        const phoneNumbers = document.getElementById('phoneNumbers').value.split('\n').filter(Boolean);
        
        const campaign = { name, openingLine, objectives, phoneNumbers };

        console.log('Campaign data:', campaign);

        try {
            let url = `https://ai-automated-outreach-5c4ade0e7f79.herokuapp.com/api/campaigns`;
            let method = 'POST';

            if (isEditing) {
                url = `https://ai-automated-outreach-5c4ade0e7f79.herokuapp.com/api/campaigns/${editingCampaignId}`;
                method = 'PUT';
            }

            const response = await fetch(url, {
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(campaign),
            });

            let data;
            try {
                data = await response.json();
            } catch (e) {
                console.error('Error parsing JSON:', e);
                data = null;
            }

            if (!response.ok) {
                console.error('Server responded with error:', response.status, data);
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            console.log('Campaign operation successful:', data);
            
            // Fetch all campaigns to refresh the list
            await fetchCampaigns();
            
            clearForm();
        } catch (error) {
            console.error('Error:', error);
            // Even if there's an error, update the UI if we're editing
            if (isEditing) {
                await fetchCampaigns();
                clearForm();
            }
        }
    }

    function clearForm() {
        document.getElementById('campaignName').value = '';
        document.getElementById('openingLine').value = '';
        document.getElementById('objectives').innerHTML = '';
        document.getElementById('phoneNumbers').value = '';
        addObjective(); // Add one empty objective
        isEditing = false;
        editingCampaignId = null;
        createCampaignBtn.style.display = 'inline-block';
        saveCampaignBtn.style.display = 'none';
    }

    createCampaignBtn.addEventListener('click', handleCampaign);
    saveCampaignBtn.addEventListener('click', handleCampaign);
    console.log('Event listeners added to buttons');
});

// -------------------------------------------------------------------------------------







function updateCampaignsTable() {
    const tbody = document.querySelector('#campaignsTable tbody');
    tbody.innerHTML = '';
    campaigns.forEach(campaign => {
        const row = tbody.insertRow();
        row.insertCell().textContent = campaign.name || 'N/A';
        row.insertCell().textContent = campaign.openingLine || 'N/A';
        row.insertCell().innerHTML = campaign.objectives && campaign.objectives.length ? 
            `<ul>${campaign.objectives.map(obj => `<li>${obj}</li>`).join('')}</ul>` : 'N/A';
        row.insertCell().textContent = campaign.phoneNumbers ? campaign.phoneNumbers.length : 'N/A';
        const actionCell = row.insertCell();
        
        const startButton = document.createElement('button');
        startButton.textContent = 'Start Calling';
        startButton.onclick = () => startCampaign(campaign._id);

        const editButton = document.createElement('button');
        editButton.textContent = 'Edit';
        editButton.className = 'edit-campaign';
        editButton.onclick = () => editCampaign(campaign._id);
        
        const deleteButton = document.createElement('button');
        deleteButton.textContent = 'Delete';
        deleteButton.className = 'delete-campaign';
        deleteButton.onclick = () => deleteCampaign(campaign._id);
        
        actionCell.appendChild(startButton);
        actionCell.appendChild(editButton);
        actionCell.appendChild(deleteButton);
    });
}



function editCampaign(id) {
    console.log('Editing campaign with id:', id);
    const campaign = campaigns.find(c => c._id === id);
    if (campaign) {
        document.getElementById('campaignName').value = campaign.name;
        document.getElementById('openingLine').value = campaign.openingLine;
        document.getElementById('phoneNumbers').value = campaign.phoneNumbers.join('\n');
        const objectives = document.getElementById('objectives');
        objectives.innerHTML = '';
        campaign.objectives.forEach(obj => {
            const objectiveContainer = document.createElement('div');
            objectiveContainer.className = 'objective-container';
            
            const input = document.createElement('input');
            input.type = 'text';
            input.value = obj;
            
            const deleteButton = document.createElement('button');
            deleteButton.textContent = 'Delete';
            deleteButton.className = 'delete-objective';
            deleteButton.onclick = function() {
                objectives.removeChild(objectiveContainer);
            };
            
            objectiveContainer.appendChild(input);
            objectiveContainer.appendChild(deleteButton);
            objectives.appendChild(objectiveContainer);
        });
        isEditing = true;
        editingCampaignId = id;
        document.getElementById('createCampaignBtn').style.display = 'none';
        document.getElementById('saveCampaignBtn').style.display = 'inline-block';
    } else {
        console.error('Campaign not found with id:', id);
    }
}

async function deleteCampaign(id) {
    // Show a confirmation dialog
    const isConfirmed = confirm("Are you sure you want to delete this campaign?");
    
    // If the user didn't confirm, exit the function
    if (!isConfirmed) {
        return;
    }

    try {
        const response = await fetch(`https://ai-automated-outreach-5c4ade0e7f79.herokuapp.com/api/campaigns/${id}`, {
            method: 'DELETE',
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        await fetchCampaigns(); // Refresh the campaign list
        alert("Campaign deleted successfully.");
    } catch (error) {
        console.error('Error deleting campaign:', error);
        alert("An error occurred while deleting the campaign.");
    }
}

function updateChart() {
    const ctx = document.getElementById('chart').getContext('2d');
    const data = {
        labels: campaigns.map(c => c.name),
        datasets: [
            {
                label: 'Total Calls',
                data: campaigns.map(c => c.customers ? c.customers.length : 0),
                backgroundColor: 'rgba(75, 192, 192, 0.6)',
            },
            {
                label: 'Warm Leads',
                data: campaigns.map(c => {
                    if (!c.customers) return 0;
                    return c.customers.filter(customer => customer.ranking >= 3).length;
                }),
                backgroundColor: 'rgba(255, 159, 64, 0.6)',
            },
            {
                label: 'Conversions',
                data: campaigns.map(c => {
                    if (!c.customers) return 0;
                    return c.customers.filter(customer => customer.ranking === 5).length;
                }),
                backgroundColor: 'rgba(153, 102, 255, 0.6)',
            }
        ]
    };

    if (chart) {
        chart.destroy();
    }

    chart = new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Handle CSV file upload
document.getElementById('csvFile').addEventListener('change', function(e) {
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.onload = function(e) {
        const content = e.target.result;
        const numbers = content.split('\n').filter(Boolean);
        document.getElementById('phoneNumbers').value = numbers.join('\n');
        alert(`CSV content loaded: ${numbers.length} numbers added`);
    };
    reader.readAsText(file);
});

// Initialize with one empty objective and chart
addObjective();
updateChart();

async function fetchCampaigns() {
    try {
        const response = await fetch('https://ai-automated-outreach-5c4ade0e7f79.herokuapp.com/api/campaigns');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        campaigns = data;
        updateCampaignsTable();
        updateChart();
    } catch (error) {
        console.error('Error fetching campaigns:', error);
    }
}

async function startCampaign(id) {
    try {
        const response = await fetch(`https://ai-automated-outreach-5c4ade0e7f79.herokuapp.com/api/campaigns/${id}/start`, {
            method: 'POST',
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('Campaign started:', result);
        alert('Campaign started successfully!');
        
        // Optionally, update the UI to reflect that the campaign has started
        await fetchCampaigns();
    } catch (error) {
        console.error('Error starting campaign:', error);
        alert('An error occurred while starting the campaign');
    }
}
