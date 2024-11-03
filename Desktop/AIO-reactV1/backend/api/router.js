// router.js
import { connectToDatabase, createCampaign, getCampaigns, updateCampaign, deleteCampaign, startingCampaign } from '../app/db.js';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';

const JWT_SECRET = process.env.JWT_SECRET; // Replace with a secure secret

export default async function (fastify, opts) {
    // User Registration
    fastify.post('/register', async (request, reply) => {
        try {
            const { email, password } = request.body;
            const db = await connectToDatabase();
            const users = db.collection('Clients');

            // Check if user already exists
            const existingUser = await users.findOne({ email });
            if (existingUser) {
                return reply.status(400).send({ error: 'User already exists' });
            }

            // Hash password
            const hashedPassword = await bcrypt.hash(password, 10);

            // Save user
            const result = await users.insertOne({ email, password: hashedPassword });
            reply.status(201).send({ message: 'User registered successfully' });
        } catch (error) {
            reply.status(500).send({ error: 'Error registering user' });
        }
    });

    // User Login
    fastify.post('/login', async (request, reply) => {
        try {
            const { email, password } = request.body;
            const db = await connectToDatabase();
            const users = db.collection('Clients');

            // Find user
            const user = await users.findOne({ email });
            if (!user) {
                return reply.status(400).send({ error: 'User not found' });
            }

            // Check password
            const isPasswordValid = await bcrypt.compare(password, user.password);
            if (!isPasswordValid) {
                return reply.status(400).send({ error: 'Invalid credentials' });
            }

            // Generate JWT
            const token = jwt.sign({ email: user.email }, JWT_SECRET, { expiresIn: '1h' });
            reply.send({ message: 'Login successful', token });
        } catch (error) {
            reply.status(500).send({ error: 'Error logging in' });
        }
    });

    // CREATE CAMPAIGN
    fastify.post('/campaigns', async (request, reply) => {
        try {
            const id = await createCampaign(request.body);
            reply.status(201).send({ id });
        } catch (error) {
            reply.status(500).send({ error: error.message });
        }
    });

    // GET CAMPAIGNS
    fastify.get('/campaigns', async (request, reply) => {
        try {
            const campaigns = await getCampaigns();
            reply.send(campaigns);
        } catch (error) {
            console.error('Error fetching campaigns:', error);
            reply.status(500).send({ error: error.message });
        }
    });

    // UPDATE CAMPAIGN
    fastify.put('/campaigns/:id', async (request, reply) => {
        try {
            console.log('Campaign ID:', request.params.id);
            console.log('Request Body Before Modification:', request.body);

            // Remove _id if it exists in the body to prevent modification of the immutable field
            const updateData = { ...request.body };
            delete updateData._id;

            console.log('Request Body After Modification:', updateData);

            const updatedCampaign = await updateCampaign(request.params.id, updateData);

            if (!updatedCampaign) {
                throw new Error('Campaign not found');
            }

            console.log('Updated Campaign:', updatedCampaign);
            reply.send(updatedCampaign);
        } catch (error) {
            console.error('Error updating campaign:', error.message);
            reply.status(error.message === 'Campaign not found' ? 404 : 500).send({ error: error.message });
        }
    });

    // DELETE CAMPAIGN
    fastify.delete('/campaigns/:id', async (request, reply) => {
        try {
            const count = await deleteCampaign(request.params.id);
            reply.send({ deletedCount: count });
        } catch (error) {
            reply.status(500).send({ error: error.message });
        }
    });

    // START CAMPAIGN
    fastify.post('/campaigns/:id/start', async (request, reply) => {
        try {
            const result = await startingCampaign(request.params.id);
            reply.send({ message: 'Campaign started successfully', result });
        } catch (error) {
            console.error('Error starting campaign:', error);
            reply.status(500).send({ error: 'An error occurred while starting the campaign' });
        }
    });
}
