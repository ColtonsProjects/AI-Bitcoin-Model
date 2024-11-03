// models/Customer.js
import mongoose from 'mongoose';

const customerSchema = new mongoose.Schema({
    name: String,
    email: String,
    phoneNumber: String,
    ranking: { type: Number, min: 1, max: 5 },
    callSummary: String,
    extraInfo: String,
    duration: Number,
});

export default mongoose.model('Customer', customerSchema);
