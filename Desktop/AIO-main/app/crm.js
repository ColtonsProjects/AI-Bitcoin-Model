import OpenAI from 'openai';
import fs from 'fs/promises';
import { runDbOperation } from './db.js';
import { ObjectId } from 'mongodb';
import { getAudioDurationInSeconds } from 'get-audio-duration';

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});





export async function analyzeCallTranscript(campaignId, transcriptPath, audioPath, phoneNumber) {
    try {
        console.log('Analyzing call transcript...');
        const transcript = await fs.readFile(transcriptPath, 'utf-8');
        console.log(transcript);

        // Get the duration of the audio file
        const duration = await getAudioDurationInSeconds(audioPath);

        const analysis = await openai.chat.completions.create({
            model: "gpt-4o",
            messages: [
                { role: "system", content: "You are an AI assistant that analyzes call transcripts and provides insights. Respond with a JSON object." },
                { role: "user", content: `Analyze this call transcript and provide brief information about the call in a JSON object with the following fields:
                - name: Name of the customer on the phone
                - ranking: A lead warmth rating from 1-5 (5 being the warmest).
                - callSummary: A short 1-2 sentence, brief overview of how the call went.
                - extraInfo: Short Key information gained about the client.
                - email: The client's email if mentioned, otherwise null.

                Transcript: ${transcript}` }
            ],
            response_format: { type: "json_object" }
        });

        const result = JSON.parse(analysis.choices[0].message.content);

        // Update the campaign in the database with the duration included
        await runDbOperation(async (db) => {
            await db.collection('campaigns').updateOne(
                { _id: new ObjectId(campaignId) },
                {
                    $push: {
                        customers: {
                            phoneNumber: phoneNumber,
                            ranking: result.ranking,
                            callSummary: result.callSummary,
                            extraInfo: result.extraInfo,
                            email: result.email,
                            calledAt: new Date(),
                            duration: Math.round(duration) 
                        }
                    }
                }
            );
        });

        // Delete the audio and transcript files
        await fs.unlink(audioPath);
        await fs.unlink(transcriptPath);

        console.log('Call analysis completed and stored in the campaign.\n---------------------');
    } catch (error) {
        console.error('Error analyzing call transcript:', error);
    }
}

