const { createClient } = require('@supabase/supabase-js');

// Supabase configuration
const SUPABASE_URL = 'https://rzrawloihpdozkvumpye.supabase.co';
// Using the secret key from Supabase dashboard (sb_secret format is the new standard)
const SUPABASE_SERVICE_KEY = 'sb_secret_cCOp_ZxA1w4ept6eaN7S3w_HVSyT_Uq';

// Create Supabase client with service role key (bypasses RLS)
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY, {
    auth: {
        autoRefreshToken: false,
        persistSession: false
    }
});

console.log('✅ Supabase client initialized');
console.log('📍 Project URL:', SUPABASE_URL);

module.exports = supabase;
