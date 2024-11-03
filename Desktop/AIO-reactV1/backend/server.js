// server.js
import Fastify from 'fastify';
import dotenv from 'dotenv';
import fastifyFormBody from '@fastify/formbody';
import fastifyWs from '@fastify/websocket';
import fastifyStatic from '@fastify/static';
import path from 'path';
import { connectToDatabase } from './app/db.js';
import router from './api/router.js';
import { analyzeCallTranscript } from './app/crm.js';

// Load environment variables
dotenv.config();

const fastify = Fastify({ logger: true });
const PORT = process.env.PORT || 5050;
const __dirname = path.dirname(new URL(import.meta.url).pathname);

// Register plugins
fastify.register(fastifyFormBody);
fastify.register(fastifyWs);

// Database Connection
connectToDatabase().catch((err) => {
    console.error('Failed to connect to MongoDB:', err);
    process.exit(1);
});

// Register API routes with a prefix
fastify.register(router, { prefix: '/api' });

// Serve React frontend in production mode
if (process.env.NODE_ENV === 'production') {
    fastify.register(fastifyStatic, {
        root: path.join(__dirname, '../frontend/build'),
        wildcard: false,
    });

    // Catch-all route to serve the React app for non-API requests
    fastify.get('/*', (request, reply) => {
        reply.sendFile('index.html');
    });
}

// Twilio Recording Endpoint
fastify.post('/recording-status', async (request, reply) => {
    const recordingUrl = request.body.RecordingUrl;
    const callSid = request.body.CallSid;
    const campaignId = request.query.campaignId;
    const phoneNumber = request.query.phoneNumber;

    const dirs = createRecordingDirectories();
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const fileName = `twilio_${callSid}_${timestamp}.mp3`;
    const filePath = path.join(dirs, fileName);

    await downloadRecording(recordingUrl, filePath);
    const transcription = await transcribeAudio(filePath);

    if (campaignId) {
        analyzeCallTranscript(campaignId, transcription, phoneNumber);
    }

    reply.send({ received: true, transcription });
});

// WebSocket for Twilio <-> OpenAI Real-Time Interaction
fastify.register(async (fastify) => {
    fastify.get('/media-stream', { websocket: true }, (connection, req) => {
        const customSystemMessage = req.query.message || 'default message';
        handleWebSocket(customSystemMessage)(connection, req);
    });
});

function createRecordingDirectories() {
    const baseDir = path.join(__dirname, 'Recordings', 'Outbound');
    if (!fs.existsSync(baseDir)) fs.mkdirSync(baseDir, { recursive: true });
    return baseDir;
}

async function downloadRecording(url, filePath) {
    const response = await fetch(url, {
        headers: {
            'Authorization': 'Basic ' + Buffer.from(`${process.env.TWILIO_ACCOUNT_SID}:${process.env.TWILIO_AUTH_TOKEN}`).toString('base64')
        }
    });
    const buffer = await response.buffer();
    fs.writeFileSync(filePath, buffer);
}

async function transcribeAudio(filePath) {
    const transcription = await openai.audio.transcriptions.create({
        file: fs.createReadStream(filePath),
        model: "whisper-1",
    });
    return transcription.text;
}

function handleWebSocket(customSystemMessage) {
    return (connection, req) => {
        const openAiWs = new WebSocket('wss://api.openai.com/v1/realtime', {
            headers: { Authorization: `Bearer ${process.env.OPENAI_API_KEY}` },
        });
        
        openAiWs.on('open', () => openAiWs.send(JSON.stringify({ type: 'session.update', session: { instructions: customSystemMessage } })));
        openAiWs.on('message', (data) => connection.send(data));
        connection.on('message', (message) => openAiWs.send(message));
        
        connection.on('close', () => openAiWs.close());
    };
}

// Start the server
fastify.listen({ port: PORT, host: '0.0.0.0' }, (err, address) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`Server is listening on ${address}`);
});

export default fastify;
