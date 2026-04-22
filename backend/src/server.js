const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const connectDB = require('./config/database');

// Load environment variables
dotenv.config();

// Initialize Express
const app = express();

// Connect to MongoDB
connectDB();

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Increased for base64 images
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Request logging middleware
app.use((req, res, next) => {
  console.log(`${req.method} ${req.path}`);
  next();
});

// Routes
app.get('/', (req, res) => {
  res.json({
    success: true,
    message: 'Coconut Health Monitor API',
    version: '1.0.0',
  });
});

app.get('/health', (req, res) => {
  res.json({
    success: true,
    message: 'Server is healthy',
    timestamp: new Date().toISOString(),
  });
});

// API Routes
app.use('/api/auth', require('./routes/auth'));
app.use('/api/users', require('./routes/users'));
app.use('/api/scans', require('./routes/scans'));
app.use('/api/notifications', require('./routes/notifications'));
app.use('/api/plantations', require('./routes/plantations'));
app.use('/api/trees', require('./routes/trees'));
app.use('/api/iot', require('./routes/iot'));

// 404 Handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: 'Route not found',
  });
});

// Error Handler
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    success: false,
    message: 'Internal Server Error',
    error: process.env.NODE_ENV === 'development' ? err.message : undefined,
  });
});

// Start server
const PORT = process.env.PORT || 5000;
app.listen(PORT, '0.0.0.0', () => {
  console.log(`\n🚀 Server running on port ${PORT}`);
  console.log(`📍 http://localhost:${PORT}`);
  console.log(`📡 http://0.0.0.0:${PORT} (accessible from all devices)`);
  console.log(`🌍 Environment: ${process.env.NODE_ENV || 'development'}\n`);
});

// UDP Discovery — ESP32 automatically finds this server (local network only)
if (process.env.NODE_ENV !== 'production') {
const dgram = require('dgram');
const os = require('os');
const udpServer = dgram.createSocket('udp4');

udpServer.on('message', (msg, rinfo) => {
  if (msg.toString().trim() === 'COCONUT_DISCOVER') {
    // Get local IP on the same network
    const interfaces = os.networkInterfaces();
    let localIP = '127.0.0.1';
    for (const iface of Object.values(interfaces)) {
      for (const addr of iface) {
        if (addr.family === 'IPv4' && !addr.internal) {
          // Prefer the IP on the same subnet as the requester
          const reqParts = rinfo.address.split('.').slice(0, 3).join('.');
          const myParts = addr.address.split('.').slice(0, 3).join('.');
          if (reqParts === myParts) {
            localIP = addr.address;
          }
        }
      }
    }
    // If no subnet match, use first non-internal IPv4
    if (localIP === '127.0.0.1') {
      for (const iface of Object.values(interfaces)) {
        for (const addr of iface) {
          if (addr.family === 'IPv4' && !addr.internal) {
            localIP = addr.address;
            break;
          }
        }
        if (localIP !== '127.0.0.1') break;
      }
    }

    const response = `COCONUT_SERVER:${localIP}:${PORT}`;
    udpServer.send(response, rinfo.port, rinfo.address);
    console.log(`📡 Discovery: ESP32 found us! Sent ${localIP}:${PORT} to ${rinfo.address}`);
  }
});

udpServer.bind(5001, '0.0.0.0', () => {
  console.log('📡 UDP Discovery listening on port 5001');
});
} // end if not production
