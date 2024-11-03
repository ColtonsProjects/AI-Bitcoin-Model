// CampaignManager.js

import React, { useState, useEffect } from 'react';

const CampaignManager = ({ onCampaignSubmit, editingCampaign }) => {
    const [campaignForm, setCampaignForm] = useState({
        name: '',
        companyInfo: '',
        openingLine: '',
        objectives: [''], // Ensure the first objective is always present
        phoneNumbers: [], // Ensure phoneNumbers starts as an array
    });

    useEffect(() => {
        if (editingCampaign) {
            setCampaignForm({
                ...editingCampaign,
                phoneNumbers: Array.isArray(editingCampaign.phoneNumbers)
                    ? editingCampaign.phoneNumbers
                    : (editingCampaign.phoneNumbers || '').split('\n'), // Ensure it's an array
            });
        }
    }, [editingCampaign]);

    const handleInputChange = (event) => {
        const { name, value } = event.target;
        setCampaignForm((prev) => ({
            ...prev,
            [name]: name === 'phoneNumbers' ? value.split('\n') : value, // Convert line-separated text to array for phone numbers
        }));
    };

    const handleObjectiveChange = (index, value) => {
        setCampaignForm((prev) => {
            const objectives = [...prev.objectives];
            objectives[index] = value;
            return { ...prev, objectives };
        });
    };

    const addObjective = () => {
        setCampaignForm((prev) => ({
            ...prev,
            objectives: [...prev.objectives, ''],
        }));
    };

    const removeObjective = (index) => {
        setCampaignForm((prev) => {
            const objectives = [...prev.objectives];
            objectives.splice(index, 1);
            return { ...prev, objectives };
        });
    };

    const clearForm = () => {
        setCampaignForm({
            name: '',
            companyInfo: '',
            openingLine: '',
            objectives: [''],
            phoneNumbers: [],
        });
    };

    return (
        <form onSubmit={(e) => {
            e.preventDefault();
            onCampaignSubmit(campaignForm);
            clearForm();
        }} className="campaign-form">
            <h2>{editingCampaign ? "Edit Campaign" : "Create Campaign"}</h2>

            <label htmlFor="campaignTitle">Campaign Title</label>
            <input
                id="campaignTitle"
                type="text"
                name="name"
                value={campaignForm.name}
                onChange={handleInputChange}
                placeholder="Campaign Title"
                required
            />

            <label>Company Information</label>
            <textarea
                name="companyInfo"
                value={campaignForm.companyInfo}
                onChange={handleInputChange}
                placeholder="Company Information"
                rows="3"
            />

            <label>Opening Line</label>
            <textarea
                name="openingLine"
                value={campaignForm.openingLine}
                onChange={handleInputChange}
                placeholder="Opening Line"
                rows="3"
            />

            <label>Objectives</label>
            {campaignForm.objectives.map((objective, index) => (
                <div key={index} className="objective-input">
                    <input
                        type="text"
                        value={objective}
                        onChange={(e) => handleObjectiveChange(index, e.target.value)}
                        placeholder={`Objective ${index + 1}`}
                        required={index === 0} // The first objective is required
                    />
                    {index > 0 && (
                        <button type="button" onClick={() => removeObjective(index)} className="remove-objective">
                            Remove
                        </button>
                    )}
                </div>
            ))}
            <button type="button" onClick={addObjective} className="add-objective">
                Add Objective
            </button>

            <label>Phone Numbers</label>
            <textarea
                name="phoneNumbers"
                value={campaignForm.phoneNumbers.join('\n')} // Display as new-line-separated text
                onChange={handleInputChange}
                placeholder="Enter phone numbers, one per line"
                rows="4"
            />

            <button type="submit" className="submit-campaign">
                {editingCampaign ? "Save Changes" : "Submit Campaign"}
            </button>
        </form>
    );
};

export default CampaignManager;
