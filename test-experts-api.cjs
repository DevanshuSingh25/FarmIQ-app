// Quick test script to verify experts API
require('dotenv').config({ path: './server/.env' });
const { dbHelpers } = require('./server/database');

async function testExpertsAPI() {
    console.log('🧪 Testing Experts API...\n');

    try {
        // Test 1: Get all experts
        console.log('Test 1: Fetching all experts...');
        const allExperts = await dbHelpers.getExperts({});
        console.log(`✅ Found ${allExperts.length} experts`);

        if (allExperts.length > 0) {
            console.log('\nFirst expert:');
            console.log(JSON.stringify(allExperts[0], null, 2));

            // Test 2: Get with search
            console.log('\nTest 2: Searching for experts...');
            const searched = await dbHelpers.getExperts({ q: 'crop' });
            console.log(`✅ Search found ${searched.length} experts`);

            // Test 3: Get with specialization filter
            if (allExperts[0].specializations) {
                const specs = JSON.parse(allExperts[0].specializations);
                if (specs.length > 0) {
                    console.log(`\nTest 3: Filtering by specialization "${specs[0]}"...`);
                    const filtered = await dbHelpers.getExperts({ specialization: specs[0] });
                    console.log(`✅ Filter found ${filtered.length} experts`);
                }
            }
        } else {
            console.log('⚠️  No experts in database. Check Supabase.');
        }

        console.log('\n✅ All tests passed!');
    } catch (error) {
        console.error('❌ Error:', error.message);
        console.error(error);
    }
}

testExpertsAPI();
