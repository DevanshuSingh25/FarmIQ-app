const bcrypt = require('bcryptjs');
const { dbHelpers } = require('./database');

// Password hashing configuration
const SALT_ROUNDS = 10;

// Authentication helper functions
const authHelpers = {
  // Hash password
  hashPassword: async (password) => {
    return await bcrypt.hash(password, SALT_ROUNDS);
  },

  // Compare password with hash
  comparePassword: async (password, hash) => {
    return await bcrypt.compare(password, hash);
  },

  // Register a new user (no username, aadhar, village, district, state)
  register: async (userData) => {
    try {
      const { role, full_name, email, phone, password, language_pref } = userData;

      // Validate input
      if (!email || !password) {
        throw new Error('Email and password are required');
      }

      if (!['farmer', 'vendor', 'admin'].includes(role || 'farmer')) {
        throw new Error('Invalid role');
      }

      if (phone && !/^\d{10}$/.test(phone)) {
        throw new Error('Phone number must be 10 digits');
      }

      if (password.length < 6) {
        throw new Error('Password must be at least 6 characters');
      }

      // Validate email format
      if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
        throw new Error('Invalid email format');
      }

      // Hash password
      const password_hash = await authHelpers.hashPassword(password);

      // Insert user into database (no username, aadhar)
      const result = await dbHelpers.insertUser({
        email,
        password_hash,
        role: role || 'farmer',
        phone: phone || ''
      });

      // Create corresponding profile (no username, aadhar, village, district, state)
      await dbHelpers.insertProfile({
        id: result.id,
        full_name: full_name || '',
        email,
        phone_number: phone || '',
        language_pref: language_pref || 'en',
        location: userData.location,
        crops_grown: userData.crops_grown,
        available_quantity: userData.available_quantity,
        expected_price: userData.expected_price
      });

      return { success: true, userId: result.id };
    } catch (error) {
      throw error;
    }
  },

  // Login user (email only, no username)
  login: async (role, email, password) => {
    try {
      // Find user by role and email
      const user = await dbHelpers.findUserByRoleAndEmail(role, email);

      if (!user) {
        throw new Error('Invalid email or password for selected role');
      }

      // Compare password
      const isValidPassword = await authHelpers.comparePassword(password, user.password_hash);

      if (!isValidPassword) {
        throw new Error('Invalid email or password for selected role');
      }

      // Return user data (without password hash)
      return {
        id: user.id,
        role: user.role,
        email: user.email,
        phone: user.phone,
        created_at: user.created_at
      };
    } catch (error) {
      throw error;
    }
  },

  // Get user by ID (for session validation)
  getUserById: async (id) => {
    try {
      const user = await dbHelpers.findUserById(id);
      return user;
    } catch (error) {
      throw error;
    }
  },

  // Get user profile by ID
  getUserProfile: async (userId) => {
    try {
      let profile = await dbHelpers.findProfileByUserId(userId);
      if (!profile) {
        // Auto-create profile from user data if missing
        const user = await dbHelpers.findUserById(userId);
        if (user) {
          await dbHelpers.insertProfile({
            id: user.id,
            full_name: '',
            email: user.email || '',
            phone_number: user.phone || '',
            language_pref: 'en'
          });
          profile = await dbHelpers.findProfileByUserId(userId);
        }
      }
      return profile;
    } catch (error) {
      throw error;
    }
  }
};

module.exports = authHelpers;
