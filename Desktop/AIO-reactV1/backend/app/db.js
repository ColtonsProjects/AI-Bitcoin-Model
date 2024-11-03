import dotenv from 'dotenv';
import { MongoClient, ObjectId } from 'mongodb';
import twilio from 'twilio';



dotenv.config();

const { MONGO_URI, DB_NAME, MAIN_SERVER_URL, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, REACT_APP_MAIN_SERVER_URL } = process.env;
const client = twilio(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN);



let db;

// Connect to MongoDB
async function connectToDatabase() {

    if (!MONGO_URI) {
        throw new Error('MONGO_URI is not defined in the environment variables');
    }

    try {
        const mongoClient = await MongoClient.connect(MONGO_URI, {
            serverSelectionTimeoutMS: 30000, // Keep this if you want a longer timeout
        });
        db = mongoClient.db(DB_NAME);
        console.log('---------------------\nSuccessfully connected to MongoDB');
        return db;
    } catch (error) {
        console.error('Failed to connect to MongoDB:', error);
        throw error;
    }
}

// Function to run database operations
async function runDbOperation(operation) {
    if (!db) {
        await connectToDatabase();
    }
    return operation(db);
}

// ------------------ CAMPAIGN OPERATIONS ---------------------
async function createCampaign(campaignData) {
    const database = await connectToDatabase();
    const result = await database.collection('campaigns').insertOne(campaignData);
    return result.insertedId;
}

async function getCampaigns() {
    const database = await connectToDatabase();
    return await database.collection('campaigns').find({}).toArray();
}

async function updateCampaign(id, campaignData) {
    console.log('Updating campaign with id:', id);
    console.log('Update data:', campaignData);

    const database = await connectToDatabase();
    
    // Convert string id to MongoDB ObjectId
    const objectId = new ObjectId(id);
    
    // Remove id from campaignData if it exists
    const { id: _, ...updateData } = campaignData;
    
    const result = await database.collection('campaigns').findOneAndUpdate(
        { _id: objectId },
        { $set: updateData },
        { returnDocument: 'after' } // This option returns the updated document
    );
    
    console.log('Result from findOneAndUpdate:', result);

    if (!result) {
        console.error('Campaign not found for id:', id);
        throw new Error('Campaign not found');
    }
    
    return result;
}

async function deleteCampaign(id) {
    console.log('Attempting to delete campaign with id:', id);
    const database = await connectToDatabase();
    
    // Convert string id to MongoDB ObjectId
    const objectId = new ObjectId(id);
    
    const result = await database.collection('campaigns').deleteOne({ _id: objectId });
    
    console.log('Delete result:', result);
    
    if (result.deletedCount === 0) {
        throw new Error('No campaign found with the given id');
    }
    
    return result.deletedCount;
}
// ------------------------------------------------------------




// Function to initiate outbound call + PROMPT
async function initiateOutboundCall(phoneNumber, campaignData) {
    console.log(`Initiating call to ${phoneNumber}\n---------------------`);

    // CUSTOM PROMPT FOR ALL CALLS
    const customSystemMessage = `
        You are an AI assistant for a company named ${campaignData.name}.
        This is some information about the company: ${campaignData.companyInfo}
        Your opening line is: "${campaignData.openingLine}"
        Your objectives are: ${campaignData.objectives.join(', ')}
        RULES:
        - You must first respond by stating the opening line.
        - ALWAYS wait for a full/complete response or answer from the user before proceeding to speak or to the next objective
        - Keep your responses concise and to the point.
        - You must try your best to achieve the objectives listed above, in order.
        - You must respond in a conversational manner, and answer all questions, but do not get off track.
    `;

    try {
        const encodedMessage = encodeURIComponent(Buffer.from(customSystemMessage).toString('base64'));

        const call = await client.calls.create({
            url: `${REACT_APP_MAIN_SERVER_URL}/twiml?message=${encodedMessage}`,
            to: phoneNumber,
            from: TWILIO_PHONE_NUMBER,
            record: true,
            recordingStatusCallback: `${REACT_APP_MAIN_SERVER_URL}/recording-status?campaignId=${campaignData._id}&phoneNumber=${encodeURIComponent(phoneNumber)}`,
            recordingStatusCallbackEvent: ['completed']
        });

        return call.sid;
    } catch (error) {
        console.error('Error initiating outbound call:', error);
        throw error;
    }
}

// Function to start a campaign
async function startingCampaign(id) {
    const database = await connectToDatabase();
    const campaign = await database.collection('campaigns').findOne({ _id: new ObjectId(id) });

    if (!campaign) {
        throw new Error('Campaign not found');
    }

    const callResults = [];
    for (const phoneNumber of campaign.phoneNumbers) {
        try {
            const callSid = await initiateOutboundCall(phoneNumber, campaign);
            callResults.push({ phoneNumber, callSid, status: 'initiated' });
        } catch (error) {
            console.error(`Error calling ${phoneNumber}:`, error);
            callResults.push({ phoneNumber, status: 'failed', error: error.message });
        }
        // Add a delay between calls to avoid overwhelming the system
        await new Promise(resolve => setTimeout(resolve, 5000));
    }

    // Update campaign status and call results
    await database.collection('campaigns').updateOne(
        { _id: new ObjectId(id) },
        { 
            $set: { 
                status: 'active',
                callResults: callResults
            } 
        }
    );

    return { message: 'Campaign started successfully', callResults };
}

export {
    connectToDatabase,
    runDbOperation,
    createCampaign,
    getCampaigns,
    updateCampaign,
    deleteCampaign,
    initiateOutboundCall,
    startingCampaign,
};
