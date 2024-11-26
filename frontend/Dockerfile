# Use a Node.js base image
FROM node:16

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Copy the frontend folder into the container
COPY . .

# Install dependencies
RUN npm install

# Expose the port Vite uses (e.g., 5173)
EXPOSE 5173

# Run the frontend development server
CMD ["npm", "run", "dev"]
