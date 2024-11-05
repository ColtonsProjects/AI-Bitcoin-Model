// main.js
import Fastify from 'fastify';
import dotenv from 'dotenv';
import fastifyFormBody from '@fastify/formbody';
import fastifyWs from '@fastify/websocket';
import fastifyStatic from '@fastify/static';
import fastifyCors from '@fastify/cors';
import path from 'path';
import WebSocket from 'ws';
import twilio from 'twilio';
import fs from 'fs';
import fetch from 'node-fetch';
import OpenAI from 'openai';
import { analyzeCallTranscript } from './app/crm.js';
import router from './api/router.js'; // Importing the router

// Load environment variables from .env file
dotenv.config();

const { OPENAI_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER } = process.env;

if (!OPENAI_API_KEY) {
    console.error('Missing OpenAI API key. Please set it in the .env file.');
    process.exit(1);
}

const client = twilio(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN);
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

const fastify = Fastify();
const PORT = process.env.PORT || 5050;
const __dirname = path.dirname(new URL(import.meta.url).pathname);

let systemMessage = `
  You are an AI assistant 

  RULES:
  - You must start by introducing yourself
  - You must respond in a conversational manner, and answer all questions.
`;
const VOICE = 'alloy';

fastify.register(fastifyFormBody);
fastify.register(fastifyWs);




// ------------------ ADAM'S RETARD CODE ---------------------------------

// Serve static files
fastify.register(fastifyStatic, {
    root: path.join(__dirname, 'public'),
    prefix: '/static',// CHANGE LATER
});

// Register the router with a prefix
fastify.register(router, { prefix: '/api' }); // Registering the router







// ------------------------ RECORD THE CONVO ----------------------------

function createRecordingDirectories() {
    const baseDir = path.join(process.cwd(), 'Recordings');
    const dirs = {
        outbound: path.join(baseDir, 'Outbound')
    };
    
    Object.values(dirs).forEach(dir => {
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
    });
    
    return dirs;
}

async function downloadRecording(url, filePath) {
    try {
        const response = await fetch(url, {
            headers: {
                'Authorization': 'Basic ' + Buffer.from(`${TWILIO_ACCOUNT_SID}:${TWILIO_AUTH_TOKEN}`).toString('base64')
            }
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const buffer = await response.buffer();
        fs.writeFileSync(filePath, buffer);
        console.log(`Recording downloaded\n---------------------`);
    } catch (error) {
        console.error('Error downloading recording:', error);
    }
}

async function transcribeAudio(filePath) {
    try {
        const transcription = await openai.audio.transcriptions.create({
            file: fs.createReadStream(filePath),
            model: "whisper-1",
        });
        return transcription.text;
    } catch (error) {
        console.error('Error transcribing audio:', error);
    }
}

fastify.post('/recording-status', async (request, reply) => {
    const recordingUrl = request.body.RecordingUrl;
    const callSid = request.body.CallSid;
    const campaignId = request.query.campaignId; 
    const phoneNumber = request.query.phoneNumber;

    console.log(`---------------------\nRecording completed`);
    // console.log(`Recording URL: ${recordingUrl}`);

    const dirs = createRecordingDirectories();
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const fileName = `twilio_${callSid}_${timestamp}.mp3`;
    const filePath = path.join(dirs.outbound, fileName);

    await downloadRecording(recordingUrl, filePath);

    // Transcribe the downloaded audio
    const transcription = await transcribeAudio(filePath);

    // Save the transcription to a text file
    const transcriptionFileName = `${path.parse(fileName).name}.txt`;
    const transcriptionFilePath = path.join(dirs.outbound, transcriptionFileName);
    fs.writeFileSync(transcriptionFilePath, transcription);
    console.log(`---------------------\nTranscription saved`);

    if (campaignId) {
        // Use the campaignId extracted from the query parameter
        analyzeCallTranscript(campaignId, transcriptionFilePath, filePath, phoneNumber);
    } else {
        console.warn('Campaign ID not provided in the request');
    }

    reply.send({ received: true, transcription });
});






// ------------------------ RUNS THE WEBSOCKET ----------------------------

//new call / decode prompt
fastify.all('/twiml', async (request, reply) => {

    if (request.query.message) {
        try {
            systemMessage = Buffer.from(request.query.message, 'base64').toString('utf-8');
        } catch (error) {
            console.error('Error decoding custom system message:', error);
        }
    }

    const twimlResponse = `<?xml version="1.0" encoding="UTF-8"?>
                          <Response>
                              <Connect>
                                  <Stream url="wss://${request.headers.host}/media-stream?message=${encodeURIComponent(systemMessage)}"/>
                              </Connect>
                          </Response>`;

    reply.type('text/xml').send(twimlResponse);
});

//connection functionality
function handleWebSocket(customSystemMessage = SYSTEM_MESSAGE) {
    return (connection, req) => {
        console.log('--------------------- \n Twilio media stream connected.');

        // Initialize OpenAI WebSocket connection
        const openAiWs = new WebSocket('wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01', {
            headers: {
                Authorization: `Bearer ${OPENAI_API_KEY}`,
                'OpenAI-Beta': 'realtime=v1',
            },
        });

        let streamSid = null;

        // Function to send session update to OpenAI
        const sendSessionUpdate = () => {
            const sessionUpdate = {
                type: 'session.update',
                session: {
                    turn_detection: { type: 'server_vad' },
                    input_audio_format: 'g711_ulaw',
                    output_audio_format: 'g711_ulaw',
                    voice: VOICE,
                    instructions: customSystemMessage,
                    modalities: ['text', 'audio'],
                    temperature: 0.8,
                },
            };
            openAiWs.send(JSON.stringify(sessionUpdate));
        };

        // OpenAI WebSocket connection
        openAiWs.on('open', () => {
            console.log('Connected to OpenAI Realtime API \n---------------------');
            setTimeout(sendSessionUpdate, 250); // Delay to ensure connection stability
        });

        // Handle incoming messages from OpenAI
        openAiWs.on('message', (data) => {
            try {
                const response = JSON.parse(data);
                if (response.type === 'response.audio.delta' && response.delta) {
                    const audioDelta = {
                        event: 'media',
                        streamSid: streamSid,
                        media: { payload: Buffer.from(response.delta, 'base64').toString('base64') },
                    };
                    // Send audio back to Twilio
                    connection.send(JSON.stringify(audioDelta));
                }
            } catch (error) {
                console.error('Error processing OpenAI message:', error);
            }
        });

        // Handle incoming messages from Twilio
        connection.on('message', (message) => {
            try {
                const data = JSON.parse(message);
                if (data.event === 'start') {
                    streamSid = data.start.streamSid;

                } else if (data.event === 'media') {
                    // Pass media (audio) data from Twilio to OpenAI
                    const audioData = {
                        type: 'input_audio_buffer.append',
                        audio: data.media.payload,
                    };
                    if (openAiWs.readyState === WebSocket.OPEN) {
                        openAiWs.send(JSON.stringify(audioData));
                    }
                }
            } catch (error) {
                console.error('Error parsing Twilio message:', error);
            }
        });

        // Handle WebSocket close events
        connection.on('close', () => {
            console.log('--------------------- \nTwilio media stream disconnected.');
            if (openAiWs.readyState === WebSocket.OPEN) openAiWs.close();
        });

        openAiWs.on('close', () => {
            console.log('OpenAI WebSocket connection closed.\n---------------------');
        });

        openAiWs.on('error', (error) => {
            console.error('OpenAI WebSocket error:', error);
        });
    };
}

//new connection
fastify.register(async (fastify) => {
    fastify.get('/media-stream', { websocket: true }, (connection, req) => {
        const customSystemMessage = req.query.message
            ? decodeURIComponent(req.query.message)
            : systemMessage;
        handleWebSocket(customSystemMessage)(connection, req);
    });
});

//Send the user to dashboard
fastify.get('/', async (request, reply) => { // THIS IS AN ESSENTIAL FUNCTION THAT MAKES NO SENSE
    reply.type('text/html').send(fs.readFileSync(path.join(__dirname, 'public', 'dashboard.html')));
});

// Start the server
fastify.listen({ port: PORT, host: '0.0.0.0' }, (err, address) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`Server is listening on ${address}`);
});
