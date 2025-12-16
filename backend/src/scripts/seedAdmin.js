const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const dotenv = require('dotenv');

// Load env vars
dotenv.config({ path: require('path').join(__dirname, '../../.env') });

const User = require('../models/User');

const seedAdmin = async () => {
  try {
    // Connect to MongoDB
    await mongoose.connect(process.env.MONGODB_URI);
    console.log('Connected to MongoDB');

    // Drop the firebaseUid index if it exists
    try {
      await User.collection.dropIndex('firebaseUid_1');
      console.log('Dropped firebaseUid index');
    } catch (e) {
      // Index might not exist, ignore
    }

    // Delete existing admin to recreate with correct password
    await User.deleteOne({ email: 'admin@gmail.com' });
    console.log('Deleted existing admin if any');

    // Create admin user - password will be hashed by pre-save hook
    await User.create({
      email: 'admin@gmail.com',
      password: '123456', // Will be hashed by pre-save hook
      displayName: 'Admin',
      role: 'admin',
      authProvider: 'email',
    });
    console.log('Admin user created successfully');

    console.log('\nAdmin credentials:');
    console.log('Email: admin@gmail.com');
    console.log('Password: 123456');

    process.exit(0);
  } catch (error) {
    console.error('Error seeding admin:', error);
    process.exit(1);
  }
};

seedAdmin();
