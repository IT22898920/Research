/**
 * Quick script to register an IoT device
 * ========================================
 * Run: node src/scripts/registerDevice.js
 *
 * This will:
 * 1. Create a test user (or use existing)
 * 2. Register an ESP32 device
 * 3. Print DEVICE_ID and DEVICE_KEY for config.h
 */

const mongoose = require('mongoose');
const crypto = require('crypto');
const dotenv = require('dotenv');
const path = require('path');

dotenv.config({ path: path.join(__dirname, '../../.env') });

const User = require('../models/User');
const IoTDevice = require('../models/IoTDevice');
const Plantation = require('../models/Plantation');

async function registerDevice() {
  try {
    // Connect to MongoDB
    const mongoUri = process.env.MONGODB_URI;
    if (!mongoUri) {
      console.error('\n❌ MONGODB_URI not found in .env file!');
      console.log('Create backend/.env with: MONGODB_URI=mongodb://localhost:27017/coconut_health');
      process.exit(1);
    }

    await mongoose.connect(mongoUri);
    console.log('✅ Connected to MongoDB\n');

    // Find first user
    const user = await User.findOne({ isActive: true });
    if (!user) {
      console.error('❌ No active user found! Login to the app first to create a user.');
      process.exit(1);
    }
    console.log(`👤 Using user: ${user.displayName || user.email} (${user._id})`);

    // Check if device already exists
    const existingDevice = await IoTDevice.findOne({ deviceId: 'ESP32_001' });
    if (existingDevice) {
      console.log('\n⚠️  Device ESP32_001 already registered!');
      console.log('   Generating new key...\n');

      const newKey = crypto.randomBytes(32).toString('hex');
      existingDevice.deviceKey = newKey;
      existingDevice.owner = user._id;
      await existingDevice.save();

      console.log('='.repeat(60));
      console.log('  ESP32 CONFIGURATION — Copy to your code!');
      console.log('='.repeat(60));
      console.log(`  DEVICE_ID  = "${existingDevice.deviceId}"`);
      console.log(`  DEVICE_KEY = "${newKey}"`);
      console.log('='.repeat(60));

      // Find default plantation
      const plantation = await Plantation.findOne({ owner: user._id, isActive: true });
      if (plantation) {
        existingDevice.defaultPlantation = plantation._id;
        await existingDevice.save();
        console.log(`\n🌴 Default plantation: ${plantation.name}`);
      }

      await mongoose.disconnect();
      return;
    }

    // Find a plantation for default
    const plantation = await Plantation.findOne({ owner: user._id, isActive: true });

    // Use the same key as in ESP32 firmware
    const deviceKey = '2de60b58e801fb571fbcd5818738d41e1901302d17f6fdf9365bab8b18d8d325';

    // Create device
    const device = await IoTDevice.create({
      deviceId: 'ESP32_001',
      name: 'Coconut GPS Device',
      owner: user._id,
      deviceKey: deviceKey,
      defaultPlantation: plantation ? plantation._id : undefined,
    });

    console.log('\n✅ Device registered successfully!\n');
    console.log('='.repeat(60));
    console.log('  ESP32 CONFIGURATION — Copy to your code!');
    console.log('='.repeat(60));
    console.log(`  DEVICE_ID  = "${device.deviceId}"`);
    console.log(`  DEVICE_KEY = "${deviceKey}"`);
    console.log('='.repeat(60));

    if (plantation) {
      console.log(`\n🌴 Default plantation: ${plantation.name}`);
      console.log('   Trees captured by this device will be added here.');
    } else {
      console.log('\n⚠️  No plantation found. Create one in the app first.');
    }

    console.log('\n📋 Paste these values in your ESP32 code (coconut_gps_live.ino)');
    console.log('   Then upload to ESP32!\n');

    await mongoose.disconnect();
  } catch (error) {
    console.error('Error:', error.message);
    await mongoose.disconnect();
    process.exit(1);
  }
}

registerDevice();
