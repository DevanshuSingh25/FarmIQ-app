// Test script to verify ThingSpeak API connection
const axios = require('axios');

const THINGSPEAK_API_KEY = 'OTIJXUV8A9RZ1VVC';
const CHANNEL_ID = '3189406'; // Correct channel ID
const limit = 5;

async function testThingSpeak() {
    // Try with API key in query parameter
    console.log('\n🔍 Test 1: Using API key as query parameter...');
    try {
        const url1 = `https://api.thingspeak.com/channels/${CHANNEL_ID}/feeds.json?api_key=${THINGSPEAK_API_KEY}&results=${limit}`;
        console.log('📡 URL:', url1.replace(THINGSPEAK_API_KEY, '[REDACTED]'));

        const response = await axios.get(url1, { timeout: 10000 });
        console.log('✅ Success with query parameter!');
        console.log('📊 Number of feeds:', response.data.feeds?.length || 0);
        displayLatest(response.data);
        return;
    } catch (error) {
        console.log('❌ Failed:', error.response?.status, error.response?.data || error.message);
    }

    // Try public channel (no API key)
    console.log('\n🔍 Test 2: Trying as public channel (no API key)...');
    try {
        const url2 = `https://api.thingspeak.com/channels/${CHANNEL_ID}/feeds.json?results=${limit}`;
        console.log('📡 URL:', url2);

        const response = await axios.get(url2, { timeout: 10000 });
        console.log('✅ Success as public channel!');
        console.log('📊 Number of feeds:', response.data.feeds?.length || 0);
        displayLatest(response.data);
        return;
    } catch (error) {
        console.log('❌ Failed:', error.response?.status, error.response?.data || error.message);
    }

    console.log('\n❌ Both methods failed. Possible issues:');
    console.log('   1. Channel ID might be incorrect (current: 2189406)');
    console.log('   2. Channel might be private and API key invalid');
    console.log('   3. Please verify Channel ID from ThingSpeak dashboard');
}

function displayLatest(data) {
    if (data.feeds && data.feeds.length > 0) {
        console.log('\n🎯 Latest reading:');
        const latest = data.feeds[data.feeds.length - 1];
        console.log('  Timestamp:', latest.created_at);
        console.log('  Temperature (Field1):', latest.field1);
        console.log('  Humidity (Field2):', latest.field2);
        console.log('  Soil Moisture (Field3):', latest.field3);
    }
}

testThingSpeak();
