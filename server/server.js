require('dotenv').config();
const express = require('express');
const session = require('express-session');
const cors = require('cors');
const path = require('path');
const axios = require('axios');
const { initDatabase, dbHelpers, db } = require('./database');
const authHelpers = require('./auth');

const app = express();
const PORT = process.env.PORT || 3001;

// Update CORS for production
const allowedOrigins = process.env.NODE_ENV === 'production'
  ? ['https://farm-frontend-jb39.onrender.com', 'https://farm-frontend-jb39.onrender.com/']
  : ['http://localhost:8080', 'http://localhost:5173'];

app.use(cors({
  origin: function (origin, callback) {
    // Log the origin for debugging
    console.log('CORS request from origin:', origin);

    // Temporarily allow all origins for debugging
    if (process.env.NODE_ENV === 'production') {
      // In production, allow all origins temporarily
      callback(null, true);
    } else {
      // In development, use strict CORS
      if (!origin) return callback(null, true);

      if (allowedOrigins.indexOf(origin) !== -1) {
        callback(null, true);
      } else {
        console.log('CORS blocked origin:', origin);
        callback(new Error('Not allowed by CORS'));
      }
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'Cookie']
}));

// Middleware (CORS already configured above)

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Session configuration
app.use(session({
  secret: 'farmiq-secret-key-change-in-production',
  resave: false,
  saveUninitialized: false,
  cookie: {
    secure: false, // Set to true in production with HTTPS
    httpOnly: true,
    maxAge: 24 * 60 * 60 * 1000 // 24 hours
  }
}));

// Authentication middleware
const requireAuth = (req, res, next) => {
  if (req.session.userId) {
    next();
  } else {
    res.status(401).json({ message: 'Authentication required' });
  }
};

// Role-based middleware
const requireRole = (roles) => {
  return (req, res, next) => {
    if (req.session.userId && req.session.role && roles.includes(req.session.role)) {
      next();
    } else {
      res.status(403).json({ message: 'Insufficient permissions' });
    }
  };
};

// API Routes

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'OK',
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV || 'development',
    cors_origin: req.headers.origin || 'no-origin'
  });
});

// Simple test endpoint
app.get('/api/test', (req, res) => {
  res.json({
    message: 'Backend is working!',
    timestamp: new Date().toISOString(),
    origin: req.headers.origin
  });
});

// Authentication routes

// Register
app.post('/api/auth/register', async (req, res) => {
  try {
    const { role, full_name, email, phone, password, language_pref, location, crops_grown, available_quantity, expected_price } = req.body;

    const result = await authHelpers.register({
      role: role || 'farmer',
      full_name,
      email,
      phone,
      password,
      language_pref,
      location,
      crops_grown,
      available_quantity,
      expected_price
    });

    res.status(201).json({ ok: true, userId: result.userId });
  } catch (error) {
    if (error.message && error.message.includes('Email already exists')) {
      res.status(409).json({ message: error.message });
    } else {
      console.error('Registration error:', error);
      res.status(400).json({ message: error.message });
    }
  }
});

// Login
app.post('/api/auth/login', async (req, res) => {
  try {
    const { role, email, password } = req.body;  // email only, no username

    const user = await authHelpers.login(role, email, password);

    // Set session
    req.session.userId = user.id;
    req.session.role = user.role;
    req.session.email = user.email;

    // Determine redirect URL based on role
    let redirectUrl;
    switch (user.role) {
      case 'farmer':
        redirectUrl = '/farmer/dashboard';
        break;
      case 'vendor':
        redirectUrl = '/vendor/dashboard';
        break;
      case 'admin':
        redirectUrl = '/admin/dashboard';
        break;
      default:
        redirectUrl = '/login';
    }

    res.json({
      success: true,
      user: {
        id: user.id,
        role: user.role,
        email: user.email,
        phone: user.phone
      },
      redirectUrl
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(401).json({ message: error.message });
  }
});

// Get current session
app.get('/api/auth/session', async (req, res) => {
  try {
    if (req.session.userId) {
      const user = await authHelpers.getUserById(req.session.userId);
      if (user) {
        res.json({
          authenticated: true,
          user: {
            id: user.id,
            role: user.role,
            email: user.email,
            phone: user.phone
          }
        });
      } else {
        // User not found, clear session
        req.session.destroy();
        res.json({ authenticated: false });
      }
    } else {
      res.json({ authenticated: false });
    }
  } catch (error) {
    console.error('Session check error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Get user details by ID (for profile page)
app.get('/api/auth/user/:id', requireAuth, async (req, res) => {
  try {
    const userId = parseInt(req.params.id);

    // Ensure user can only access their own data
    if (req.session.userId !== userId) {
      return res.status(403).json({ message: 'Access denied' });
    }

    const user = await authHelpers.getUserById(userId);
    if (user) {
      res.json(user);
    } else {
      res.status(404).json({ message: 'User not found' });
    }
  } catch (error) {
    console.error('Get user details error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Logout
app.post('/api/auth/logout', (req, res) => {
  req.session.destroy((err) => {
    if (err) {
      console.error('Logout error:', err);
      res.status(500).json({ message: 'Could not log out' });
    } else {
      res.json({ ok: true });
    }
  });
});

// Get current user's profile
app.get('/api/me/profile', requireAuth, async (req, res) => {
  try {
    const userId = req.session.userId;
    const profile = await authHelpers.getUserProfile(userId);

    if (profile) {
      res.json(profile);
    } else {
      res.status(404).json({ message: 'Profile not found' });
    }
  } catch (error) {
    console.error('Get profile error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Update current user's profile (no aadhar, village, district, state)
app.put('/api/me/profile', requireAuth, async (req, res) => {
  try {
    const userId = req.session.userId;
    const { full_name, phone_number, language_pref } = req.body;  // only allowed fields

    const { dbHelpers } = require('./database');
    await dbHelpers.updateProfile(userId, {
      full_name,
      phone_number,
      language_pref
    });

    res.json({ ok: true, message: 'Profile updated successfully' });
  } catch (error) {
    console.error('Update profile error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Protected routes for testing
app.get('/api/farmer/dashboard', requireAuth, requireRole(['farmer']), (req, res) => {
  res.json({ message: 'Farmer dashboard data' });
});

app.get('/api/vendor/dashboard', requireAuth, requireRole(['vendor']), (req, res) => {
  res.json({ message: 'Vendor dashboard data' });
});

app.get('/api/admin/dashboard', requireAuth, requireRole(['admin']), (req, res) => {
  res.json({ message: 'Admin dashboard data' });
});

// ========== NGO SCHEMES ROUTES ==========

// Get all NGO schemes (all authenticated users can read)
app.get('/api/ngo-schemes', requireAuth, async (req, res) => {
  try {
    const schemes = await dbHelpers.getNgoSchemes();
    res.json(schemes);
  } catch (error) {
    console.error('Get NGO schemes error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// ========== GOVERNMENT SCHEMES ELIGIBILITY FILTER ==========

// Filter government schemes based on eligibility criteria
app.post('/api/government-schemes/filter', requireAuth, async (req, res) => {
  try {
    const { state, land, category, age, crop } = req.body;

    console.log('🌾 Government Schemes Filter Request:', {
      userId: req.session.userId,
      userRole: req.session.role,
      filters: { state, land, category, age, crop }
    });

    // Validate input
    if (!state && land === undefined && !category && age === undefined) {
      return res.status(400).json({
        message: 'At least one filter criteria (state, land, category, or age) is required'
      });
    }

    // Call the eligibility helper function
    const eligibleSchemes = await dbHelpers.getEligibleSchemes({
      state,
      land: land !== undefined ? parseFloat(land) : undefined,
      category,
      age: age !== undefined ? parseInt(age) : undefined
    });

    console.log(`✅ Returning ${eligibleSchemes.length} eligible schemes to farmer`);

    res.json(eligibleSchemes);
  } catch (error) {
    console.error('❌ Government schemes filter error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Get NGO scheme by ID (all authenticated users can read)
app.get('/api/ngo-schemes/:id', requireAuth, async (req, res) => {
  try {
    const schemeId = parseInt(req.params.id);
    const scheme = await dbHelpers.getNgoSchemeById(schemeId);

    if (scheme) {
      res.json(scheme);
    } else {
      res.status(404).json({ message: 'NGO scheme not found' });
    }
  } catch (error) {
    console.error('Get NGO scheme error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Create NGO scheme (admin only)
app.post('/api/ngo-schemes', requireAuth, requireRole(['admin']), async (req, res) => {
  try {
    const result = await dbHelpers.createNgoScheme(req.body);
    res.status(201).json({ id: result.id, message: 'NGO scheme created successfully' });
  } catch (error) {
    console.error('Create NGO scheme error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Update NGO scheme (admin only)
app.put('/api/ngo-schemes/:id', requireAuth, requireRole(['admin']), async (req, res) => {
  try {
    const schemeId = parseInt(req.params.id);
    await dbHelpers.updateNgoScheme(schemeId, req.body);
    res.json({ message: 'NGO scheme updated successfully' });
  } catch (error) {
    console.error('Update NGO scheme error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Delete NGO scheme (admin only)
app.delete('/api/ngo-schemes/:id', requireAuth, requireRole(['admin']), async (req, res) => {
  try {
    const schemeId = parseInt(req.params.id);
    await dbHelpers.deleteNgoScheme(schemeId);
    res.json({ message: 'NGO scheme deleted successfully' });
  } catch (error) {
    console.error('Delete NGO scheme error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// ========== SOIL LAB ROUTES ==========

// Get all soil labs (all authenticated users can read)
app.get('/api/soil-labs', requireAuth, async (req, res) => {
  try {
    const labs = await dbHelpers.getSoilLabs();
    res.json(labs);
  } catch (error) {
    console.error('Get soil labs error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Get soil lab by ID (all authenticated users can read)
app.get('/api/soil-labs/:id', requireAuth, async (req, res) => {
  try {
    const labId = parseInt(req.params.id);
    const lab = await dbHelpers.getSoilLabById(labId);

    if (lab) {
      res.json(lab);
    } else {
      res.status(404).json({ message: 'Soil lab not found' });
    }
  } catch (error) {
    console.error('Get soil lab error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Create soil lab (admin only)
app.post('/api/soil-labs', requireAuth, requireRole(['admin']), async (req, res) => {
  try {
    const result = await dbHelpers.createSoilLab(req.body);
    res.status(201).json({ id: result.id, message: 'Soil lab created successfully' });
  } catch (error) {
    console.error('Create soil lab error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Update soil lab (admin only)
app.put('/api/soil-labs/:id', requireAuth, requireRole(['admin']), async (req, res) => {
  try {
    const labId = parseInt(req.params.id);
    await dbHelpers.updateSoilLab(labId, req.body);
    res.json({ message: 'Soil lab updated successfully' });
  } catch (error) {
    console.error('Update soil lab error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Delete soil lab (admin only)
app.delete('/api/soil-labs/:id', requireAuth, requireRole(['admin']), async (req, res) => {
  try {
    const labId = parseInt(req.params.id);
    await dbHelpers.deleteSoilLab(labId);
    res.json({ message: 'Soil lab deleted successfully' });
  } catch (error) {
    console.error('Delete soil lab error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// ========== CROP HISTORY ROUTES ==========

// Get crops (farmers get own, admins get all, vendors denied)
app.get('/api/crops', requireAuth, async (req, res) => {
  try {
    const userRole = req.session.role;
    const userId = req.session.userId;

    // Vendors not allowed to access crop history
    if (userRole === 'vendor') {
      return res.status(403).json({ message: 'Vendors cannot access crop history' });
    }

    let crops;
    if (userRole === 'admin') {
      // Admins see all crops
      crops = await dbHelpers.getAllCrops();
    } else {
      // Farmers see only their crops
      crops = await dbHelpers.getCropsByUserId(userId);
    }

    res.json(crops);
  } catch (error) {
    console.error('Get crops error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Create crop (farmers and admins only, user_id from session)
app.post('/api/crops', requireAuth, async (req, res) => {
  try {
    const userRole = req.session.role;
    const userId = req.session.userId;

    // Only farmers and admins can create crops
    if (userRole === 'vendor') {
      return res.status(403).json({ message: 'Vendors cannot create crop records' });
    }

    // Validate required fields
    const { crop_name } = req.body;
    if (!crop_name) {
      return res.status(400).json({ message: 'Crop name is required' });
    }

    // SECURITY: user_id from session, NEVER from client
    // Even if client sends user_id, it's ignored
    const result = await dbHelpers.createCrop(userId, req.body);
    res.status(201).json({ id: result.id, message: 'Crop record created successfully' });
  } catch (error) {
    console.error('Create crop error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Update crop (owner or admin only)
app.put('/api/crops/:id', requireAuth, async (req, res) => {
  try {
    const cropId = parseInt(req.params.id);
    const userRole = req.session.role;
    const userId = req.session.userId;

    // Vendors not allowed
    if (userRole === 'vendor') {
      return res.status(403).json({ message: 'Vendors cannot modify crop records' });
    }

    // Check ownership (unless admin)
    const crop = await dbHelpers.getCropById(cropId);
    if (!crop) {
      return res.status(404).json({ message: 'Crop record not found' });
    }

    // Only admin or owner can update
    if (userRole !== 'admin' && crop.user_id !== userId) {
      return res.status(403).json({ message: 'You can only update your own crop records' });
    }

    await dbHelpers.updateCrop(cropId, req.body);
    res.json({ message: 'Crop record updated successfully' });
  } catch (error) {
    console.error('Update crop error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Delete crop (owner or admin only)
app.delete('/api/crops/:id', requireAuth, async (req, res) => {
  try {
    const cropId = parseInt(req.params.id);
    const userRole = req.session.role;
    const userId = req.session.userId;

    // Vendors not allowed
    if (userRole === 'vendor') {
      return res.status(403).json({ message: 'Vendors cannot delete crop records' });
    }

    // Check ownership (unless admin)
    const crop = await dbHelpers.getCropById(cropId);
    if (!crop) {
      return res.status(404).json({ message: 'Crop record not found' });
    }

    // Only admin or owner can delete
    if (userRole !== 'admin' && crop.user_id !== userId) {
      return res.status(403).json({ message: 'You can only delete your own crop records' });
    }

    await dbHelpers.deleteCrop(cropId);
    res.status(204).send();
  } catch (error) {
    console.error('Delete crop error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// ========== IOT SENSOR BOOKING API ==========

// GET /api/iot/status/:user_id - Get IoT status for a user
app.get('/api/iot/status/:user_id', requireAuth, async (req, res) => {
  try {
    const userId = parseInt(req.params.user_id);

    console.log(`🔍 IoT Status Request for user_id: ${userId}`);

    // Special case: user_id === 1 always returns 'active'
    if (userId === 1) {
      console.log('📌 Special case: user_id=1, returning active status');
      return res.json({
        user_id: userId,
        status: 'active',
        updated_at: new Date().toISOString(),
        note: 'Auto-activated for user ID 1'
      });
    }

    // Get or create status from database
    let status = await dbHelpers.getIotStatusByUserId(userId);

    if (!status) {
      // Create default 'inactive' status if doesn't exist
      console.log('📝 Creating default inactive status for user');
      status = await dbHelpers.upsertIotStatus(userId, 'inactive');
    }

    res.json(status);
  } catch (error) {
    console.error('❌ Error getting IoT status:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// GET /api/iot/request/:user_id - Get booking request for a user
app.get('/api/iot/request/:user_id', requireAuth, async (req, res) => {
  try {
    const userId = parseInt(req.params.user_id);

    console.log(`🔍 IoT Booking Request for user_id: ${userId}`);

    const booking = await dbHelpers.getIotReadingByUserId(userId);

    if (!booking) {
      return res.status(404).json({ message: 'No booking request found' });
    }

    res.json(booking);
  } catch (error) {
    console.error('❌ Error getting IoT booking request:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// POST /api/iot/request - Create new booking request
app.post('/api/iot/request', requireAuth, async (req, res) => {
  try {
    const userId = req.session.userId; // Get user_id from session, NOT from request body

    console.log(`📝 Creating IoT booking for user_id: ${userId}`);

    // Validate required fields
    const { name, phone_number, location, state, district, preferred_visit_date } = req.body;

    if (!name || !phone_number) {
      return res.status(400).json({ message: 'Name and phone number are required' });
    }

    // Validate phone number length (7-20 characters)
    if (phone_number.length < 7 || phone_number.length > 20) {
      return res.status(400).json({ message: 'Phone number must be between 7 and 20 characters' });
    }

    // Validate date format if provided
    if (preferred_visit_date) {
      const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
      if (!dateRegex.test(preferred_visit_date)) {
        return res.status(400).json({ message: 'Preferred visit date must be in YYYY-MM-DD format' });
      }

      // Try to parse the date to ensure it's valid
      const parsedDate = new Date(preferred_visit_date);
      if (isNaN(parsedDate.getTime())) {
        return res.status(400).json({ message: 'Invalid date provided' });
      }
    }

    // Check if user already has a booking
    const existingBooking = await dbHelpers.getIotReadingByUserId(userId);
    if (existingBooking) {
      console.log('⚠️ User already has an existing booking');
      return res.status(409).json({
        message: 'You already have an existing booking request',
        existing: existingBooking
      });
    }

    // Create the booking
    const newBooking = await dbHelpers.createIotReading(userId, req.body);

    console.log(`✅ Booking created successfully: ID=${newBooking.id}`);

    res.status(201).json(newBooking);
  } catch (error) {
    console.error('❌ Error creating IoT booking:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// PUT /api/iot/status/:user_id - Update IoT status (admin only)
app.put('/api/iot/status/:user_id', requireAuth, requireRole(['admin']), async (req, res) => {
  try {
    const userId = parseInt(req.params.user_id);
    const { status } = req.body;

    console.log(`🔧 Admin updating IoT status for user_id: ${userId} to: ${status}`);

    // Validate status value
    const validStatuses = ['inactive', 'active', 'booked'];
    if (!status || !validStatuses.includes(status)) {
      return res.status(400).json({
        message: `Invalid status. Must be one of: ${validStatuses.join(', ')}`
      });
    }

    // Update or create the status
    const updatedStatus = await dbHelpers.upsertIotStatus(userId, status);

    console.log(`✅ Status updated successfully for user_id: ${userId}`);

    res.json(updatedStatus);
  } catch (error) {
    console.error('❌ Error updating IoT status:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// GET /api/iot/readings/:user_id - Get sensor readings
app.get('/api/iot/readings/:user_id', requireAuth, async (req, res) => {
  try {
    const userId = parseInt(req.params.user_id);
    const limit = parseInt(req.query.limit) || 24;

    console.log(`📊 IoT Readings Request for user_id: ${userId}, limit: ${limit}`);

    // Special case: user_id === 1 always allowed to fetch readings
    if (userId !== 1) {
      // Check if user's device is active (only for non-user_id=1)
      const status = await dbHelpers.getIotStatusByUserId(userId);

      if (!status || status.status !== 'active') {
        console.log(`⚠️ Device not active for user_id: ${userId}, status: ${status?.status || 'none'}`);
        return res.status(403).json({
          message: 'Device not active. Sensor readings are only available when device status is "active".',
          current_status: status?.status || 'inactive'
        });
      }
    } else {
      console.log('📌 Special case: user_id=1, skipping status check - fetching data directly');
    }

    // Fetch real sensor data from ThingSpeak
    const readings = await dbHelpers.fetchThingSpeakReadings(limit);

    console.log(`✅ Returning ${readings.length} readings from ThingSpeak`);

    res.json(readings);
  } catch (error) {
    console.error('❌ Error getting IoT readings:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// ========== BLYNK IOT INTEGRATION ==========

// Blynk configuration
const BLYNK_AUTH_TOKEN = 'gHsyyhQpXdDMG9jfeg9UfnIdrMZMrRKH';
const BLYNK_DEVICE = 'ESP32ledINTEGRATED';
const BLYNK_BASE_URL = 'https://blynk.cloud/external/api';
const BLYNK_DATASTREAM_PIN = 'd13'; // Digital pin 13 for LED control (matches Blynk datastream configuration)

// GET /api/blynk/led/status - Get LED state from Blynk
app.get('/api/blynk/led/status', requireAuth, async (req, res) => {
  try {
    console.log('🔍 Fetching LED status from Blynk...');

    const blynkUrl = `${BLYNK_BASE_URL}/get?token=${BLYNK_AUTH_TOKEN}&${BLYNK_DATASTREAM_PIN}`;

    const response = await axios.get(blynkUrl, {
      timeout: 5000,
      headers: {
        'Accept': 'application/json'
      }
    });

    // Blynk returns the value directly (0 or 1)
    const ledState = parseInt(response.data) === 1;

    console.log(`✅ LED state from Blynk: ${ledState ? 'ON' : 'OFF'} (value: ${response.data})`);

    res.json({
      state: ledState,
      value: parseInt(response.data),
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('❌ Error fetching LED status from Blynk:', error.message);

    if (error.response) {
      console.error('Blynk API error:', error.response.status, error.response.data);
      return res.status(502).json({
        message: 'Failed to fetch LED status from Blynk',
        blynk_error: error.response.data
      });
    }

    if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
      return res.status(504).json({
        message: 'Timeout connecting to Blynk cloud'
      });
    }

    res.status(500).json({
      message: 'Internal server error while fetching LED status',
      error: error.message
    });
  }
});

// POST /api/blynk/led/control - Control LED via Blynk
app.post('/api/blynk/led/control', requireAuth, async (req, res) => {
  try {
    const { state } = req.body;

    // Validate state
    if (typeof state !== 'boolean') {
      return res.status(400).json({
        message: 'State must be a boolean value (true/false)'
      });
    }

    const ledValue = state ? 1 : 0;

    console.log(`🔧 Setting LED to: ${state ? 'ON' : 'OFF'} (value: ${ledValue})`);

    const blynkUrl = `${BLYNK_BASE_URL}/update?token=${BLYNK_AUTH_TOKEN}&${BLYNK_DATASTREAM_PIN}=${ledValue}`;

    const response = await axios.get(blynkUrl, {
      timeout: 5000,
      headers: {
        'Accept': 'application/json'
      }
    });

    console.log(`✅ LED successfully set to ${state ? 'ON' : 'OFF'}`);
    console.log('Blynk response:', response.data);

    res.json({
      success: true,
      state: state,
      value: ledValue,
      timestamp: new Date().toISOString(),
      blynk_response: response.data
    });
  } catch (error) {
    console.error('❌ Error controlling LED via Blynk:', error.message);

    if (error.response) {
      console.error('Blynk API error:', error.response.status, error.response.data);
      return res.status(502).json({
        message: 'Failed to control LED via Blynk',
        blynk_error: error.response.data
      });
    }

    if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
      return res.status(504).json({
        message: 'Timeout connecting to Blynk cloud'
      });
    }

    res.status(500).json({
      message: 'Internal server error while controlling LED',
      error: error.message
    });
  }
});


// ========== PROFILES / FARMER SEARCH ROUTES ==========

// Get profiles (for vendor farmer search)
app.get('/api/profiles', requireAuth, async (req, res) => {
  try {
    const { q } = req.query;
    console.log(`🔍 Profiles search request: q="${q}"`);

    const profiles = await dbHelpers.getProfiles({ q });
    console.log(`✅ Found ${profiles.length} profiles`);

    res.json(profiles);
  } catch (error) {
    console.error('❌ Get profiles error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// ========== EXPERTS ROUTES ==========

// Get experts with search and filter
app.get('/api/experts', requireAuth, async (req, res) => {
  try {
    const { q, specialization, limit, offset } = req.query;
    const rows = await dbHelpers.getExperts({
      q,
      specialization,
      limit: limit || 100,
      offset: offset || 0
    });

    // return parsed specialization as array
    const mapped = rows.map(r => ({
      ...r,
      specializations: r.specializations ? JSON.parse(r.specializations) : []
    }));

    res.json({ data: mapped, count: mapped.length });
  } catch (err) {
    console.error('GET /api/experts error', err);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// ========== QR CODE ROUTES ==========

// Parse QR code text
app.post('/api/qr/parse', requireAuth, async (req, res) => {
  try {
    const { qr_text } = req.body;

    if (!qr_text) {
      return res.status(400).json({ message: 'QR text is required' });
    }

    console.log('📱 QR scanned:', qr_text.substring(0, 100));

    // Return the QR text back for display
    // Frontend will handle JSON pretty-printing if needed
    res.json({ qr_text });
  } catch (error) {
    console.error('❌ QR parse error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// ========== MARKET PRICES API PROXY ==========

// In-memory cache for market prices
const MARKET_CACHE = new Map();
const MARKET_CACHE_TTL_MS = (parseInt(process.env.DATA_GOV_API_CACHE_TTL_SEC) || 300) * 1000;

function getMarketCacheKey(query) {
  return `${query.state || ''}|${query.district || ''}|${query.commodity || ''}|${query.offset || 0}|${query.limit || 50}`;
}

// GET /api/market-prices - Proxy to data.gov.in API
app.get('/api/market-prices', async (req, res) => {
  try {
    const { state, district, commodity, offset = '0', limit = '50' } = req.query;

    // Validate and parse parameters
    const parsedOffset = Math.max(0, parseInt(offset) || 0);
    const maxLimit = parseInt(process.env.DATA_GOV_API_MAX_LIMIT) || 1000;
    const parsedLimit = Math.min(Math.max(1, parseInt(limit) || 50), maxLimit);

    // Create cache key
    const cacheKey = getMarketCacheKey({ state, district, commodity, offset: parsedOffset, limit: parsedLimit });

    // Check cache
    const cached = MARKET_CACHE.get(cacheKey);
    if (cached && (Date.now() - cached.timestamp) < MARKET_CACHE_TTL_MS) {
      console.log(`market-proxy: CACHE HIT for key=${cacheKey}`);
      res.set('X-Cache', 'HIT');
      return res.json(cached.data);
    }

    // Build upstream URL
    const params = new URLSearchParams();
    params.append('api-key', process.env.DATA_GOV_API_KEY);
    params.append('format', 'json');
    params.append('offset', String(parsedOffset));
    params.append('limit', String(parsedLimit));

    // Only add filters if they are NOT "All" values
    if (state && state !== 'all' && state !== 'All States' && state !== 'All State') {
      params.append('filters[state]', state);
    }
    if (district && district !== 'all' && district !== 'All Districts') {
      params.append('filters[district]', district);
    }
    if (commodity && commodity !== 'all' && commodity !== 'All Crops' && commodity !== 'All commodities' && commodity !== 'All Commodity') {
      params.append('filters[commodity]', commodity);
    }

    const upstreamUrl = `https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?${params.toString()}`;

    // Log request (with redacted API key)
    const logUrl = upstreamUrl.replace(process.env.DATA_GOV_API_KEY, '[REDACTED]');
    console.log(`market-proxy: Fetching upstream url=${logUrl}`);

    const startTime = Date.now();

    // Make upstream request
    const timeout = parseInt(process.env.DATA_GOV_API_TIMEOUT_MS) || 15000;
    const upstreamResponse = await axios.get(upstreamUrl, {
      timeout,
      headers: {
        'Accept': 'application/json'
      }
    });

    const elapsed = Date.now() - startTime;
    console.log(`market-proxy: upstream ${upstreamResponse.status} in ${elapsed}ms`);

    // Parse and normalize response
    const rawData = upstreamResponse.data;
    let records = [];

    // data.gov.in returns data in records array
    if (rawData && Array.isArray(rawData.records)) {
      records = rawData.records;
    } else if (Array.isArray(rawData)) {
      records = rawData;
    }

    // Normalize data
    const normalizedData = records.map(item => {
      // Helper to parse price strings (remove commas, parse to int)
      const parsePrice = (priceStr) => {
        if (!priceStr) return null;
        const cleaned = String(priceStr).replace(/,/g, '').trim();
        const parsed = parseInt(cleaned, 10);
        return isNaN(parsed) ? null : parsed;
      };

      return {
        state: item.state || null,
        district: item.district || null,
        market: item.market || item.district || null,
        commodity: item.commodity || null,
        variety: item.variety || null,
        min_price: parsePrice(item.min_price),
        max_price: parsePrice(item.max_price),
        modal_price: parsePrice(item.modal_price),
        arrival_date: item.arrival_date || null
      };
    });

    // Build response
    const response = {
      meta: {
        offset: parsedOffset,
        limit: parsedLimit,
        count: normalizedData.length
      },
      data: normalizedData
    };

    // Cache the response
    MARKET_CACHE.set(cacheKey, {
      timestamp: Date.now(),
      data: response
    });
    console.log(`market-proxy: CACHE MISS - cached for key=${cacheKey}`);

    res.set('X-Cache', 'MISS');
    return res.json(response);

  } catch (error) {
    console.error('market-proxy ERROR:', error.message);

    // Handle specific error cases
    if (error.response) {
      const status = error.response.status;
      const upstreamData = error.response.data;

      console.error(`market-proxy: upstream error ${status}`, upstreamData);

      // Rate limit error
      if (status === 429) {
        const retryAfter = error.response.headers['retry-after'] || 60;
        res.set('Retry-After', String(retryAfter));
        return res.status(503).json({
          message: 'Upstream rate limit exceeded, please try again later',
          retry_after: retryAfter
        });
      }

      // Upstream server error
      if (status >= 500) {
        return res.status(502).json({
          message: 'Upstream server error - market data temporarily unavailable'
        });
      }

      // Other upstream errors (4xx)
      return res.status(502).json({
        message: 'Failed to fetch market data from upstream API'
      });
    }

    // Timeout or network error
    if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
      return res.status(504).json({
        message: 'Request timeout - upstream API not responding'
      });
    }

    // Generic error
    console.error('market-proxy: unexpected error', error);
    return res.status(500).json({
      message: 'Internal server error fetching market data'
    });
  }
});

// ========== FORUM ROUTES ==========

// Get all forum posts 
app.get('/api/forum/posts', async (req, res) => {
  try {
    const { category } = req.query;

    let query = `
      SELECT id, user_id, category, community, question, extracted_keywords, 
             status, created_at, upvotes, reply_count
      FROM forum_posts
    `;

    const params = [];
    if (category && category !== 'All') {
      query += ` WHERE category = ?`;
      params.push(category);
    }

    query += ` ORDER BY created_at DESC`;

    db.all(query, params, (err, posts) => {
      if (err) {
        console.error('Get forum posts error:', err);
        return res.status(500).json({ message: 'Database error', error: err.message });
      }

      if (!posts || posts.length === 0) {
        console.log('No posts found');
        return res.json([]);
      }

      console.log(`Found ${posts.length} posts`);

      // Get all replies
      db.all(`SELECT id, post_id, reply_text, replied_by, created_at, upvotes FROM forum_replies ORDER BY created_at ASC`, [], (err, replies) => {
        if (err) {
          console.error('Get replies error:', err);
          return res.status(500).json({ message: 'Database error', error: err.message });
        }

        console.log(`Found ${replies ? replies.length : 0} replies`);

        // Attach replies to posts
        const postsWithReplies = posts.map(post => ({
          ...post,
          replies: replies ? replies.filter(r => r.post_id === post.id) : []
        }));

        console.log(`✓ Returning ${postsWithReplies.length} posts with replies`);
        res.json(postsWithReplies);
      });
    });
  } catch (error) {
    console.error('Get forum posts error:', error);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// Get replies for a specific post
app.get('/api/forum/posts/:post_id/replies', async (req, res) => {
  try {
    const { post_id } = req.params;

    db.all(
      `SELECT id, post_id, reply_text, replied_by, upvotes, created_at 
       FROM forum_replies 
       WHERE post_id = ? 
       ORDER BY created_at ASC`,
      [post_id],
      (err, replies) => {
        if (err) {
          console.error('Get forum replies error:', err);
          return res.status(500).json({ message: 'Internal server error' });
        }
        console.log(`✓ Returning ${replies ? replies.length : 0} replies for post ${post_id}`);
        res.json(replies || []);
      }
    );
  } catch (error) {
    console.error('Get forum replies error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Helper function to generate category-specific answers
function generateGenericAnswer(category, question) {
  const answers = {
    'Crop': `For crop-related queries, I recommend consulting with your local agriculture extension officer. Based on your region's climate and soil type, they can provide specific guidance. You can also use our crop disease detection feature for image-based analysis.`,

    'Soil': `Soil health is crucial for good yields. I recommend getting a soil test done at your nearest soil testing laboratory. Based on the results, you can determine the right amendments needed. Regular addition of organic matter like FYM or compost helps improve soil structure and fertility.`,

    'Weather': `For accurate weather forecasts, please check our Weather section which provides 7-day forecasts. During extreme weather events, ensure proper drainage in fields and protect standing crops with appropriate measures. Local agriculture department can provide region-specific advisories.`,

    'Disease & Pests': `Early detection is key for pest and disease management. Try using our Crop Disease feature for image-based identification. For immediate action, consult with local plant protection experts. Always use recommended pesticides at proper dosages and follow safety guidelines.`,

    'Market': `Market prices fluctuate based on demand, supply, and seasonal factors. Check daily rates on eNAM portal or local APMC markets. Avoid distress selling - store produce properly if prices are low. Consider joining Farmer Producer Organizations (FPOs) for better bargaining power.`,

    'Fertilizers': `Fertilizer application should be based on soil test results. Over-application can harm soil health and crops. Follow recommended NPK ratios for your crop. Split application is better than single dose. Consider using bio-fertilizers and organic manures to reduce chemical dependence.`,

    'General Queries': `For general agricultural queries and government schemes, visit your local agriculture department office or Krishi Vigyan Kendra (KVK). They provide free advisory services, training programs, and information about subsidies. You can also call the Kisan Call Centre at 1800-180-1551 for expert advice.`
  };

  return answers[category] || `Thank you for your question. Our agricultural experts will review this and provide detailed guidance. In the meantime, you can explore our other features like Crop Disease Detection, Market Prices, and Weather Forecast for immediate assistance.`;
}

// Create forum post (with AUTO-ANSWER generation)
app.post('/api/forum', requireAuth, async (req, res) => {
  try {
    const userId = req.session.userId;
    const { category, question } = req.body;

    if (!category || !question) {
      return res.status(400).json({ message: 'Category and question are required' });
    }

    // Extract keywords for matching
    const { extractKeywords } = require('./keywords');
    const keywords = extractKeywords(question);

    // Insert new question with status 'Answered' and reply_count = 1 (will add auto-reply)
    db.run(
      `INSERT INTO forum_posts 
       (user_id, category, community, question, extracted_keywords, status, upvotes, reply_count, created_at)
       VALUES (?, ?, ?, ?, ?, 'Answered', 0, 1, datetime('now'))`,
      [userId, category, category, question, keywords],
      function (err) {
        if (err) {
          console.error('Error inserting forum post:', err);
          return res.status(500).json({ message: 'Failed to create post' });
        }

        const newPostId = this.lastID;
        console.log(`✓ New question inserted: ID ${newPostId}`);

        // Find similar posts for smart answer generation
        db.all(
          `SELECT id, question, extracted_keywords, community
           FROM forum_posts 
           WHERE id != ? AND community = ? AND status = 'Answered'
           ORDER BY id DESC
           LIMIT 10`,
          [newPostId, category],
          (err, similarPosts) => {
            if (err) console.error('Error finding similar posts:', err);

            // Generate answer based on similarity or use generic response
            let expertAnswer = '';
            let expertName = 'FarmIQ Expert Advisor';

            // Calculate similarity scores
            const questionKeywords = keywords.toLowerCase().split(',').map(k => k.trim());
            let bestMatch = null;
            let bestScore = 0;

            if (similarPosts && similarPosts.length > 0) {
              similarPosts.forEach(post => {
                const postKeywords = (post.extracted_keywords || '').toLowerCase().split(',').map(k => k.trim());
                let score = 0;

                questionKeywords.forEach(qk => {
                  postKeywords.forEach(pk => {
                    if (qk === pk || qk.includes(pk) || pk.includes(qk)) {
                      score++;
                    }
                  });
                });

                if (score > bestScore) {
                  bestScore = score;
                  bestMatch = post;
                }
              });
            }

            // Generate context-aware answer
            if (bestScore >= 2 && bestMatch) {
              // Found good match - get its answer
              db.get(
                `SELECT reply_text, replied_by FROM forum_replies WHERE post_id = ? LIMIT 1`,
                [bestMatch.id],
                (err, matchReply) => {
                  if (matchReply) {
                    expertAnswer = `Based on similar questions: ${matchReply.reply_text}`;
                    expertName = matchReply.replied_by;
                  } else {
                    expertAnswer = generateGenericAnswer(category, question);
                  }
                  insertReplyAndRespond();
                }
              );
            } else {
              // No good match - generate category-specific answer
              expertAnswer = generateGenericAnswer(category, question);
              insertReplyAndRespond();
            }

            function insertReplyAndRespond() {
              // Insert auto-generated answer into forum_replies
              db.run(
                `INSERT INTO forum_replies
                 (post_id, reply_text, replied_by, upvotes, created_at)
                 VALUES (?, ?, ?, 0, datetime('now'))`,
                [newPostId, expertAnswer, expertName],
                function (replyErr) {
                  if (replyErr) {
                    console.error('Error inserting auto-reply:', replyErr);
                  } else {
                    console.log(`✓ Auto-reply generated for post ${newPostId}`);
                  }

                  // Fetch the complete new post with reply to return to frontend
                  db.get(
                    `SELECT fp.*, p.full_name as user_name
                     FROM forum_posts fp
                     LEFT JOIN profiles p ON fp.user_id = p.id
                     WHERE fp.id = ?`,
                    [newPostId],
                    (err, newPost) => {
                      if (err || !newPost) {
                        return res.status(201).json({
                          id: newPostId,
                          message: 'Post created successfully'
                        });
                      }

                      // Get the reply we just created
                      db.get(
                        `SELECT * FROM forum_replies WHERE post_id = ? ORDER BY created_at DESC LIMIT 1`,
                        [newPostId],
                        (err, newReply) => {
                          // Return complete post with reply
                          res.status(201).json({
                            id: newPostId,
                            message: 'Question posted and answered successfully',
                            post: {
                              id: newPost.id,
                              user_name: newPost.user_name || 'You',
                              category: newPost.community,
                              community: newPost.community,
                              question: newPost.question,
                              keywords: newPost.extracted_keywords ? newPost.extracted_keywords.split(',') : [],
                              status: newPost.status,
                              upvotes: newPost.upvotes,
                              reply_count: newPost.reply_count,
                              created_at: newPost.created_at,
                              replies: newReply ? [{
                                id: newReply.id,
                                reply_text: newReply.reply_text,
                                replied_by: newReply.replied_by,
                                upvotes: newReply.upvotes,
                                created_at: newReply.created_at
                              }] : []
                            }
                          });
                        }
                      );
                    }
                  );
                }
              );
            }
          }
        );
      }
    );
  } catch (error) {
    console.error('Create forum post error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Create forum reply
app.post('/api/forum/reply', requireAuth, async (req, res) => {
  try {
    const { postId, replyText, repliedBy } = req.body;

    if (!postId || !replyText || !repliedBy) {
      return res.status(400).json({ message: 'Post ID, reply text, and replied by are required' });
    }

    const result = await dbHelpers.createForumReply(postId, replyText, repliedBy);
    res.status(201).json({ id: result.id, message: 'Reply added successfully' });
  } catch (error) {
    console.error('Create forum reply error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Upvote forum post
app.post('/api/forum/upvote', requireAuth, async (req, res) => {
  try {
    const { postId, action } = req.body;

    if (!postId || !action) {
      return res.status(400).json({ message: 'Post ID and action are required' });
    }

    let result;
    if (action === 'increment') {
      result = await dbHelpers.incrementPostUpvotes(postId);
    } else if (action === 'decrement') {
      result = await dbHelpers.decrementPostUpvotes(postId);
    } else {
      return res.status(400).json({ message: 'Invalid action. Use "increment" or "decrement"' });
    }

    res.json({ upvotes: result.upvotes });
  } catch (error) {
    console.error('Upvote forum post error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Search farmer forum with keyword extraction and intelligent matching
app.get('/api/farmer-forum/search', requireAuth, async (req, res) => {
  try {
    const { question, community } = req.query;

    if (!question) {
      return res.status(400).json({ message: 'Question is required' });
    }

    // Extract keywords from the question
    const { extractKeywords } = require('./keywords');
    const keywords = extractKeywords(question);
    console.log(`Extracted keywords from "${question}": ${keywords}`);

    // Search with keywords and community filter
    const results = await dbHelpers.searchFarmerForumByKeywords(keywords, community);

    if (!results || results.length === 0) {
      return res.json({
        found: false,
        message: 'No community-specific result found. Please refine your question.',
        extractedKeywords: keywords
      });
    }

    res.json({
      found: true,
      results,
      extractedKeywords: keywords
    });
  } catch (error) {
    console.error('Search farmer forum error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// ========== ADMIN MANAGEMENT ROUTES ==========

// Admin middleware to check admin role
const requireAdmin = (req, res, next) => {
  if (!req.session || !req.session.userId) {
    return res.status(401).json({ message: 'Authentication required' });
  }
  if (req.session.role !== 'admin') {
    return res.status(403).json({ message: 'Admin access required' });
  }
  next();
};

// ===== ADMIN USER/PROFILE MANAGEMENT =====

// Get all users with profiles
app.get('/api/admin/users', requireAdmin, async (req, res) => {
  try {
    db.all(
      `SELECT u.id, u.email, u.role, u.phone, u.created_at,
              p.full_name, p.phone_number, p.location, p.crops_grown, 
              p.available_quantity, p.expected_price
       FROM users u
       LEFT JOIN profiles p ON u.id = p.id
       ORDER BY u.created_at DESC`,
      [],
      (err, users) => {
        if (err) {
          console.error('Get users error:', err);
          return res.status(500).json({ message: 'Database error' });
        }
        res.json(users || []);
      }
    );
  } catch (error) {
    console.error('Get users error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Get specific user
app.get('/api/admin/users/:id', requireAdmin, async (req, res) => {
  try {
    const userId = parseInt(req.params.id);
    db.get(
      `SELECT u.*, p.full_name, p.phone_number, p.location, p.crops_grown,
              p.available_quantity, p.expected_price
       FROM users u
       LEFT JOIN profiles p ON u.id = p.id
       WHERE u.id = ?`,
      [userId],
      (err, user) => {
        if (err) {
          return res.status(500).json({ message: 'Database error' });
        }
        if (!user) {
          return res.status(404).json({ message: 'User not found' });
        }
        res.json(user);
      }
    );
  } catch (error) {
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Update user and profile
app.put('/api/admin/users/:id', requireAdmin, async (req, res) => {
  try {
    const userId = parseInt(req.params.id);
    const { email, role, phone, full_name, phone_number, location, crops_grown, available_quantity, expected_price } = req.body;

    // Update users table
    db.run(
      `UPDATE users SET email = ?, role = ?, phone = ?, updated_at = datetime('now')
       WHERE id = ?`,
      [email, role, phone, userId],
      function (err) {
        if (err) {
          console.error('Update user error:', err);
          return res.status(500).json({ message: 'Failed to update user' });
        }

        // Update profiles table
        db.run(
          `UPDATE profiles 
           SET full_name = ?, phone_number = ?, location = ?, crops_grown = ?,
               available_quantity = ?, expected_price = ?, updated_at = datetime('now')
           WHERE id = ?`,
          [full_name, phone_number, location, crops_grown, available_quantity, expected_price, userId],
          function (profileErr) {
            if (profileErr) {
              console.error('Update profile error:', profileErr);
            }
            res.json({ message: 'User updated successfully' });
          }
        );
      }
    );
  } catch (error) {
    console.error('Update user error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Delete user
app.delete('/api/admin/users/:id', requireAdmin, async (req, res) => {
  try {
    const userId = parseInt(req.params.id);

    db.run('DELETE FROM users WHERE id = ?', [userId], function (err) {
      if (err) {
        console.error('Delete user error:', err);
        return res.status(500).json({ message: 'Failed to delete user' });
      }
      res.json({ message: 'User deleted successfully' });
    });
  } catch (error) {
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Reset user password
app.post('/api/admin/users/:id/reset-password', requireAdmin, async (req, res) => {
  try {
    const userId = parseInt(req.params.id);
    const { newPassword } = req.body;

    if (!newPassword || newPassword.length < 6) {
      return res.status(400).json({ message: 'Password must be at least 6 characters' });
    }

    const bcrypt = require('bcryptjs');
    const hashedPassword = await bcrypt.hash(newPassword, 10);

    db.run(
      `UPDATE users SET password_hash = ?, updated_at = datetime('now') WHERE id = ?`,
      [hashedPassword, userId],
      function (err) {
        if (err) {
          console.error('Reset password error:', err);
          return res.status(500).json({ message: 'Failed to reset password' });
        }
        res.json({ message: 'Password reset successfully' });
      }
    );
  } catch (error) {
    console.error('Reset password error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// ===== ADMIN LAB MANAGEMENT =====

// Get all labs
app.get('/api/admin/labs', requireAdmin, async (req, res) => {
  try {
    const labs = await dbHelpers.getSoilLabs();
    res.json(labs);
  } catch (error) {
    console.error('Get labs error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Create lab
app.post('/api/admin/labs', requireAdmin, async (req, res) => {
  try {
    const result = await dbHelpers.createSoilLab(req.body);
    res.status(201).json({ id: result.id, message: 'Lab created successfully' });
  } catch (error) {
    console.error('Create lab error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Update lab
app.put('/api/admin/labs/:id', requireAdmin, async (req, res) => {
  try {
    const labId = parseInt(req.params.id);
    await dbHelpers.updateSoilLab(labId, req.body);
    res.json({ message: 'Lab updated successfully' });
  } catch (error) {
    console.error('Update lab error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Delete lab
app.delete('/api/admin/labs/:id', requireAdmin, async (req, res) => {
  try {
    const labId = parseInt(req.params.id);
    await dbHelpers.deleteSoilLab(labId);
    res.json({ message: 'Lab deleted successfully' });
  } catch (error) {
    console.error('Delete lab error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// ===== ADMIN SCHEME MANAGEMENT =====

// Get all schemes
app.get('/api/admin/schemes', requireAdmin, async (req, res) => {
  try {
    const schemes = await dbHelpers.getNgoSchemes();
    res.json(schemes);
  } catch (error) {
    console.error('Get schemes error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Create scheme
app.post('/api/admin/schemes', requireAdmin, async (req, res) => {
  try {
    const result = await dbHelpers.createNgoScheme(req.body);
    res.status(201).json({ id: result.id, message: 'Scheme created successfully' });
  } catch (error) {
    console.error('Create scheme error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Update scheme
app.put('/api/admin/schemes/:id', requireAdmin, async (req, res) => {
  try {
    const schemeId = parseInt(req.params.id);
    await dbHelpers.updateNgoScheme(schemeId, req.body);
    res.json({ message: 'Scheme updated successfully' });
  } catch (error) {
    console.error('Update scheme error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Delete scheme
app.delete('/api/admin/schemes/:id', requireAdmin, async (req, res) => {
  try {
    const schemeId = parseInt(req.params.id);
    await dbHelpers.deleteNgoScheme(schemeId);
    res.json({ message: 'Scheme deleted successfully' });
  } catch (error) {
    console.error('Delete scheme error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// ===== ADMIN EXPERT MANAGEMENT =====

// Get all experts
app.get('/api/admin/experts', requireAdmin, async (req, res) => {
  try {
    const experts = await dbHelpers.getExperts({ limit: 100, offset: 0 });
    res.json(experts);
  } catch (error) {
    console.error('Get experts error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Create expert
app.post('/api/admin/experts', requireAdmin, async (req, res) => {
  try {
    const { name, experience_years, specializations, phone_number } = req.body;

    db.run(
      `INSERT INTO experts_info (name, experience_years, specializations, phone_number, rating, consultation_count)
       VALUES (?, ?, ?, ?, 0.0, 0)`,
      [name, experience_years || 0, specializations || '', phone_number || ''],
      function (err) {
        if (err) {
          console.error('Create expert error:', err);
          return res.status(500).json({ message: 'Failed to create expert' });
        }
        res.status(201).json({ id: this.lastID, message: 'Expert created successfully' });
      }
    );
  } catch (error) {
    console.error('Create expert error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Update expert
app.put('/api/admin/experts/:id', requireAdmin, async (req, res) => {
  try {
    const expertId = parseInt(req.params.id);
    const { name, experience_years, specializations, phone_number, rating } = req.body;

    db.run(
      `UPDATE experts_info 
       SET name = ?, experience_years = ?, specializations = ?, phone_number = ?, 
           rating = ?, updated_at = datetime('now')
       WHERE id = ?`,
      [name, experience_years, specializations, phone_number, rating || 0, expertId],
      function (err) {
        if (err) {
          console.error('Update expert error:', err);
          return res.status(500).json({ message: 'Failed to update expert' });
        }
        res.json({ message: 'Expert updated successfully' });
      }
    );
  } catch (error) {
    console.error('Update expert error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Delete expert
app.delete('/api/admin/experts/:id', requireAdmin, async (req, res) => {
  try {
    const expertId = parseInt(req.params.id);

    db.run('DELETE FROM experts_info WHERE id = ?', [expertId], function (err) {
      if (err) {
        console.error('Delete expert error:', err);
        return res.status(500).json({ message: 'Failed to delete expert' });
      }
      res.json({ message: 'Expert deleted successfully' });
    });
  } catch (error) {
    console.error('Delete expert error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({ message: 'Internal server error' });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ message: 'Route not found' });
});

// Initialize database and start server
const startServer = async () => {
  try {
    await initDatabase();
    app.listen(PORT, '0.0.0.0', () => {
      console.log(`✅ Server running on http://localhost:${PORT}`);
      console.log(`✅ Server accessible at http://127.0.0.1:${PORT}`);
      console.log('✅ Database initialized successfully');
      console.log(`✅ CORS enabled for: ${allowedOrigins.join(', ')}`);
      console.log(`✅ Health check: http://localhost:${PORT}/api/health`);
    });
  } catch (error) {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  }
};

startServer();

module.exports = app;
