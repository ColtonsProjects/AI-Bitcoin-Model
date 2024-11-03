// models/Campaign.js
import mongoose from 'mongoose';

const campaignSchema = new mongoose.Schema({
    name: { type: String, required: true },
    openingLine: String,
    objectives: [String],
    phoneNumbers: [String],
    customers: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Customer' }],
});

export default mongoose.model('Campaign', campaignSchema);
