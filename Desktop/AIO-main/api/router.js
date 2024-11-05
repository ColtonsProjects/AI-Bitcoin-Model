// router.js
import { connectToDatabase, createCampaign, getCampaigns, updateCampaign, deleteCampaign, startingCampaign } from '../app/db.js';

export default async function (fastify, opts) {
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
            const updatedCampaign = await updateCampaign(request.params.id, request.body);
            reply.send(updatedCampaign);
        } catch (error) {
            console.error('Error updating campaign:', error);
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
