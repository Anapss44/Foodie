# Zumattuu Foodie App

This project now includes a simple Node.js backend for:
- item availability checks
- order submission
- delivery partner location and tracking

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```
2. Start the server:
   ```bash
   npm start
   ```
3. Open the app in your browser:
   ```
   http://localhost:3000
   ```

## Backend endpoints

- `GET /api/availability/:itemId`
- `POST /api/order`
- `GET /api/delivery/:orderId`
- `GET /api/orders`

The server also serves `frontend.html` at `/`.

## Python chatbot backend

A Python chatbot backend is available at `http://localhost:5000/api/chat`.

1. Install Python dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Start the chatbot backend:
   ```bash
   python chatbot.py
   ```
3. Keep `npm start` running for the app backend, then use the chat UI as before.
