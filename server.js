const express = require('express');
const path = require('path');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

const DELIVERY_DEST = { lat: 24.8075, lng: 93.9394 };
const deliveryPartners = [
  { id: 1, name: 'Rohit', vehicle: 'Bike', location: { lat: 24.784, lng: 93.951 } },
  { id: 2, name: 'Meena', vehicle: 'Scooter', location: { lat: 24.799, lng: 93.917 } },
  { id: 3, name: 'Kiran', vehicle: 'Bike', location: { lat: 24.821, lng: 93.925 } }
];

const orders = [];

function getItemAvailability(itemId) {
  const seed = itemId % 7;
  const available = seed !== 0;
  const stock = available ? 5 + (itemId % 4) * 3 : 0;
  return { itemId, available, stock, etaMinutes: available ? 8 + (itemId % 12) : null };
}

function getPartnerStatus(orderId) {
  const order = orders.find(o => o.id === orderId);
  if (!order) return null;
  const timestamp = Date.now();
  const progress = Math.min(1, (timestamp - order.createdAt) / 1000 / 45);
  const partner = deliveryPartners.find(p => p.id === order.partnerId) || deliveryPartners[0];
  const currentLat = partner.location.lat + (DELIVERY_DEST.lat - partner.location.lat) * progress;
  const currentLng = partner.location.lng + (DELIVERY_DEST.lng - partner.location.lng) * progress;
  const eta = Math.max(2, Math.round((1 - progress) * 25));
  const status = progress < 0.25 ? 'Preparing' : progress < 0.75 ? 'On the way' : 'Almost there';
  return {
    orderId,
    status,
    eta,
    partner: {
      name: partner.name,
      vehicle: partner.vehicle,
      location: { lat: currentLat, lng: currentLng }
    }
  };
}

app.get('/api/availability/:itemId', (req, res) => {
  const itemId = parseInt(req.params.itemId, 10);
  if (Number.isNaN(itemId)) return res.status(400).json({ error: 'Invalid item id' });
  res.json(getItemAvailability(itemId));
});

app.post('/api/order', (req, res) => {
  const { restaurant, items, total, paymentMethod } = req.body;
  if (!restaurant || !Array.isArray(items) || items.length === 0 || total == null) {
    return res.status(400).json({ error: 'Invalid order payload' });
  }

  const partnerId = deliveryPartners[(orders.length + 1) % deliveryPartners.length].id;
  const orderId = orders.length + 1;
  const order = {
    id: orderId,
    restId: req.body.restId || null,
    restaurant,
    items,
    total,
    paymentMethod,
    status: paymentMethod === 'COD' ? 'Pending payment' : 'Paid',
    partnerId,
    createdAt: Date.now(),
    date: new Date().toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })
  };
  orders.push(order);

  res.json({ orderId, partnerId, status: order.status, eta: 25 });
});

app.get('/api/delivery/:orderId', (req, res) => {
  const orderId = parseInt(req.params.orderId, 10);
  if (Number.isNaN(orderId)) return res.status(400).json({ error: 'Invalid order id' });
  const status = getPartnerStatus(orderId);
  if (!status) return res.status(404).json({ error: 'Order not found' });
  res.json(status);
});

app.get('/api/orders', (req, res) => {
  res.json(orders.map(o => ({
    id: o.id,
    restId: o.restId,
    restaurant: o.restaurant,
    items: o.items,
    total: o.total,
    status: o.status,
    paymentMethod: o.paymentMethod,
    createdAt: o.createdAt,
    date: o.date
  })));
});

app.get('/', (req, res) => {
  res.sendFile(path.resolve(__dirname, 'frontend.html'));
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server started at http://localhost:${port}`);
});
