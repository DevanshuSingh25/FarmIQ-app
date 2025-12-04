const supabase = require('./supabase-client');

const initDatabase = async () => {
  console.log('✅ Supabase database ready');
  return Promise.resolve();
};

const dbHelpers = {
  insertUser: async (userData) => {
    try {
      const { email, password_hash, role, phone } = userData;
      const { data, error } = await supabase.from('users').insert([{ email, password_hash, role: role || 'farmer', phone: phone || '' }]).select().single();
      if (error) {
        if (error.code === '23505') throw new Error('Email already exists');
        throw error;
      }
      return { id: data.id };
    } catch (error) {
      throw error;
    }
  },

  findUserByRoleAndEmail: async (role, email) => {
    try {
      const { data, error } = await supabase.from('users').select('*').eq('role', role).eq('email', email).maybeSingle();
      if (error) throw error;
      return data;
    } catch (error) {
      throw error;
    }
  },

  findUserByEmail: async (email) => {
    try {
      const { data, error } = await supabase.from('users').select('*').eq('email', email).maybeSingle();
      if (error) throw error;
      return data;
    } catch (error) {
      throw error;
    }
  },

  findUserById: async (id) => {
    try {
      const { data, error } = await supabase.from('users').select('id, role, email, phone, created_at').eq('id', id).maybeSingle();
      if (error) throw error;
      return data;
    } catch (error) {
      throw error;
    }
  },

  insertProfile: async (profileData) => {
    try {
      const { id, full_name, email, phone_number, language_pref, location, crops_grown, available_quantity, expected_price } = profileData;
      const { data, error } = await supabase.from('profiles').insert([{
        id, full_name: full_name || '', email: email || '', phone_number: phone_number || '',
        language_pref: language_pref || 'en', location: location || '', crops_grown: crops_grown || '',
        available_quantity: available_quantity || '', expected_price: expected_price || ''
      }]).select().single();
      if (error) throw error;
      return { id };
    } catch (error) {
      throw error;
    }
  },

  findProfileByUserId: async (userId) => {
    try {
      const { data, error } = await supabase.from('profiles').select('*').eq('id', userId).maybeSingle();
      if (error) throw error;
      return data;
    } catch (error) {
      throw error;
    }
  },

  updateProfile: async (userId, profileData) => {
    try {
      const { full_name, phone_number, language_pref } = profileData;
      const { error } = await supabase.from('profiles').update({
        full_name, phone_number, language_pref, updated_at: new Date().toISOString()
      }).eq('id', userId);
      if (error) throw error;
      return { changes: 1 };
    } catch (error) {
      throw error;
    }
  },

  getNgoSchemes: async () => {
    try {
      const { data, error } = await supabase.from('ngo_schemes').select('*').order('created_at', { ascending: false });
      if (error) throw error;
      return data || [];
    } catch (error) {
      throw error;
    }
  },

  getNgoSchemeById: async (id) => {
    try {
      const { data, error } = await supabase.from('ngo_schemes').select('*').eq('id', id).maybeSingle();
      if (error) throw error;
      return data;
    } catch (error) {
      throw error;
    }
  },

  createNgoScheme: async (schemeData) => {
    try {
      const { name, ministry, deadline, location, contact_number, no_of_docs_required, status, benefit_text, eligibility_text } = schemeData;
      const { data, error } = await supabase.from('ngo_schemes').insert([{
        name, ministry: ministry || '', deadline: deadline || '', location: location || '',
        contact_number: contact_number || '', no_of_docs_required: no_of_docs_required || 0,
        status: status || 'active', benefit_text: benefit_text || '', eligibility_text: eligibility_text || ''
      }]).select().single();
      if (error) throw error;
      return { id: data.id };
    } catch (error) {
      throw error;
    }
  },

  updateNgoScheme: async (id, schemeData) => {
    try {
      const { name, ministry, deadline, location, contact_number, no_of_docs_required, status, benefit_text, eligibility_text } = schemeData;
      const { error } = await supabase.from('ngo_schemes').update({
        name, ministry, deadline, location, contact_number, no_of_docs_required, status, benefit_text, eligibility_text, updated_at: new Date().toISOString()
      }).eq('id', id);
      if (error) throw error;
      return { changes: 1 };
    } catch (error) {
      throw error;
    }
  },

  deleteNgoScheme: async (id) => {
    try {
      const { error } = await supabase.from('ngo_schemes').delete().eq('id', id);
      if (error) throw error;
      return { changes: 1 };
    } catch (error) {
      throw error;
    }
  },

  getEligibleSchemes: async (filters) => {
    try {
      const { state, land, category, age } = filters;
      console.log('🔍 Filtering schemes with criteria:', { state, land, category, age });
      let query = supabase.from('ngo_schemes').select('*');
      if (state) {
        query = query.or(`required_state.is.null,required_state.eq.ALL,required_state.eq.${state}`);
      }
      const { data: allSchemes, error } = await query.order('created_at', { ascending: false });
      if (error) throw error;
      let filtered = allSchemes || [];
      if (land !== undefined && land !== null) {
        filtered = filtered.filter(s => (s.min_land === null || land >= s.min_land) && (s.max_land === null || land <= s.max_land));
      }
      if (category) {
        filtered = filtered.filter(s => s.required_category === null || s.required_category === 'ALL' || s.required_category === category);
      }
      if (age !== undefined && age !== null) {
        filtered = filtered.filter(s => (s.age_min === null || age >= s.age_min) && (s.age_max === null || age <= s.age_max));
      }
      console.log(`✅ Found ${filtered.length} eligible schemes`);
      return filtered;
    } catch (error) {
      throw error;
    }
  },

  getSoilLabs: async () => {
    try {
      const { data, error } = await supabase.from('soil_lab').select('*').order('created_at', { ascending: false });
      if (error) throw error;
      return data || [];
    } catch (error) {
      throw error;
    }
  },

  getSoilLabById: async (id) => {
    try {
      const { data, error } = await supabase.from('soil_lab').select('*').eq('id', id).maybeSingle();
      if (error) throw error;
      return data;
    } catch (error) {
      throw error;
    }
  },

  createSoilLab: async (labData) => {
    try {
      const { name, location, contact_number, price, rating, tag } = labData;
      const { data, error } = await supabase.from('soil_lab').insert([{
        name, location: location || '', contact_number: contact_number || '', price: price || 0, rating: rating || 0, tag: tag || ''
      }]).select().single();
      if (error) throw error;
      return { id: data.id };
    } catch (error) {
      throw error;
    }
  },

  updateSoilLab: async (id, labData) => {
    try {
      const { name, location, contact_number, price, rating, tag } = labData;
      const { error } = await supabase.from('soil_lab').update({
        name, location, contact_number, price, rating, tag, updated_at: new Date().toISOString()
      }).eq('id', id);
      if (error) throw error;
      return { changes: 1 };
    } catch (error) {
      throw error;
    }
  },

  deleteSoilLab: async (id) => {
    try {
      const { error } = await supabase.from('soil_lab').delete().eq('id', id);
      if (error) throw error;
      return { changes: 1 };
    } catch (error) {
      throw error;
    }
  },

  getCropsByUserId: async (userId) => {
    try {
      const { data, error } = await supabase.from('crop_history').select('*').eq('user_id', userId).order('created_at', { ascending: false });
      if (error) throw error;
      return data || [];
    } catch (error) {
      throw error;
    }
  },

  getAllCrops: async () => {
    try {
      const { data, error } = await supabase.from('crop_history').select('*').order('created_at', { ascending: false });
      if (error) throw error;
      return data || [];
    } catch (error) {
      throw error;
    }
  },

  getCropById: async (id) => {
    try {
      const { data, error } = await supabase.from('crop_history').select('*').eq('id', id).maybeSingle();
      if (error) throw error;
      return data;
    } catch (error) {
      throw error;
    }
  },

  createCrop: async (userId, cropData) => {
    try {
      const { crop_name, crop_price, selling_price, crop_produced_kg } = cropData;
      const { data, error } = await supabase.from('crop_history').insert([{
        user_id: userId, crop_name, crop_price: crop_price || 0, selling_price: selling_price || 0, crop_produced_kg: crop_produced_kg || 0
      }]).select().single();
      if (error) throw error;
      return { id: data.id };
    } catch (error) {
      throw error;
    }
  },

  updateCrop: async (id, cropData) => {
    try {
      const { crop_name, crop_price, selling_price, crop_produced_kg } = cropData;
      const { error } = await supabase.from('crop_history').update({
        crop_name, crop_price, selling_price, crop_produced_kg, updated_at: new Date().toISOString()
      }).eq('id', id);
      if (error) throw error;
      return { changes: 1 };
    } catch (error) {
      throw error;
    }
  },

  deleteCrop: async (id) => {
    try {
      const { error } = await supabase.from('crop_history').delete().eq('id', id);
      if (error) throw error;
      return { changes: 1 };
    } catch (error) {
      throw error;
    }
  },

  getIotReadingByUserId: async (userId) => {
    try {
      const { data, error } = await supabase.from('iot_reading').select('*').eq('user_id', userId).order('created_at', { ascending: false }).limit(1).maybeSingle();
      if (error) throw error;
      return data;
    } catch (error) {
      throw error;
    }
  },

  createIotReading: async (userId, readingData) => {
    try {
      const { name, phone_number, location, state, district, preferred_visit_date } = readingData;
      const { data, error } = await supabase.from('iot_reading').insert([{
        user_id: userId, name, phone_number, location: location || '', state: state || '', district: district || '', preferred_visit_date: preferred_visit_date || '', status: 'pending'
      }]).select().single();
      if (error) throw error;
      return data;
    } catch (error) {
      throw error;
    }
  },

  getIotStatusByUserId: async (userId) => {
    try {
      const { data, error } = await supabase.from('iot_status').select('*').eq('user_id', userId).maybeSingle();
      if (error) throw error;
      return data;
    } catch (error) {
      throw error;
    }
  },

  upsertIotStatus: async (userId, status) => {
    try {
      const { data, error } = await supabase.from('iot_status').upsert({ user_id: userId, status, updated_at: new Date().toISOString() }).select().single();
      if (error) throw error;
      return data;
    } catch (error) {
      throw error;
    }
  },

  getProfiles: async (filter = {}) => {
    try {
      let query = supabase.from('profiles').select('id, full_name, crops_grown, available_quantity, location, expected_price, users!inner(role)').eq('users.role', 'farmer');
      if (filter.q) {
        const searchTerm = `%${filter.q}%`;
        query = query.or(`full_name.ilike.${searchTerm},location.ilike.${searchTerm}`);
      }
      query = query.order('id', { ascending: false });
      const { data, error } = await query;
      if (error) throw error;
      return (data || []).map(({ users, ...profile }) => profile);
    } catch (error) {
      throw error;
    }
  },

  fetchThingSpeakReadings: async (limit = 24) => {
    const axios = require('axios');
    const THINGSPEAK_API_KEY = 'OTIJXUV8A9RZ1VVC';
    const CHANNEL_ID = '3189406';
    try {
      const url = `https://api.thingspeak.com/channels/${CHANNEL_ID}/feeds.json?api_key=${THINGSPEAK_API_KEY}&results=${limit}`;
      const response = await axios.get(url, { timeout: 10000 });
      if (!response.data || !response.data.feeds) {
        console.warn('No data from ThingSpeak');
        return [];
      }
      const readings = response.data.feeds.map(feed => ({
        timestamp: feed.created_at,
        temperature: parseFloat(feed.field1) || 0,
        humidity: parseFloat(feed.field2) || 0,
        soil_moisture: parseFloat(feed.field3) || 0
      }));
      readings.reverse();
      console.log(`✅ Fetched ${readings.length} readings from ThingSpeak`);
      return readings;
    } catch (error) {
      console.error('❌ Error fetching ThingSpeak data:', error.message);
      return [];
    }
  },

  getExperts: async ({ q, specialization, limit = 20, offset = 0 }) => {
    try {
      let query = supabase.from('experts_info').select('id, name, experience_years, specializations, rating, consultation_count, phone_number');
      if (q) {
        query = query.or(`name.ilike.%${q}%,specializations.ilike.%${q}%`);
      }
      if (specialization) {
        query = query.ilike('specializations', `%${specialization}%`);
      }
      query = query.order('rating', { ascending: false }).range(offset, offset + limit - 1);
      const { data, error } = await query;
      if (error) throw error;
      return data || [];
    } catch (error) {
      throw error;
    }
  },

  getForumPosts: async () => {
    try {
      // Fetch posts without join
      const { data: posts, error: postsError } = await supabase
        .from('forum_posts')
        .select('id, user_id, category, community, question, extracted_keywords, status, upvotes, reply_count, created_at')
        .order('created_at', { ascending: false });

      if (postsError) throw postsError;
      if (!posts || posts.length === 0) return [];

      // Fetch all profiles to join manually
      const { data: profiles, error: profilesError } = await supabase
        .from('profiles')
        .select('id, full_name');

      if (profilesError) throw profilesError;

      // Fetch all replies
      const { data: replies, error: repliesError } = await supabase
        .from('forum_replies')
        .select('*')
        .order('created_at', { ascending: true });

      if (repliesError) {
        console.error('ERROR fetching replies:', repliesError);
        throw repliesError;
      }

      console.log(`DEBUG: Fetched ${replies ? replies.length : 0} replies from Supabase`);
      if (replies && replies.length > 0) {
        console.log('DEBUG: First reply:', JSON.stringify(replies[0]));
      }

      // Manual join: Map posts with profiles and replies
      const postsWithReplies = posts.map(post => {
        const profile = profiles?.find(p => p.id === post.user_id);
        const postReplies = (replies || []).filter(r => r.post_id === post.id);

        if (postReplies.length > 0) {
          console.log(`DEBUG: Post ${post.id} has ${postReplies.length} replies`);
        }

        return {
          ...post,
          user_name: profile?.full_name || 'Unknown User',
          replies: postReplies
        };
      });

      return postsWithReplies;
    } catch (error) {
      console.error('getForumPosts error:', error);
      throw error;
    }
  },

  createForumPost: async (userId, category, question, community, extractedKeywords) => {
    try {
      const { data, error } = await supabase.from('forum_posts').insert([{
        user_id: userId, category, community: community || category, question, extracted_keywords: extractedKeywords || '', status: 'Unanswered', upvotes: 0, reply_count: 0
      }]).select().single();
      if (error) throw error;
      return { id: data.id };
    } catch (error) {
      throw error;
    }
  },

  createForumReply: async (postId, replyText, repliedBy) => {
    try {
      const { data, error } = await supabase.from('forum_replies').insert([{ post_id: postId, reply_text: replyText, replied_by: repliedBy, upvotes: 0 }]).select().single();
      if (error) throw error;
      const { data: post } = await supabase.from('forum_posts').select('reply_count').eq('id', postId).single();
      await supabase.from('forum_posts').update({ reply_count: (post?.reply_count || 0) + 1 }).eq('id', postId);
      return { id: data.id };
    } catch (error) {
      throw error;
    }
  },

  incrementPostUpvotes: async (postId) => {
    try {
      const { data, error } = await supabase.from('forum_posts').select('upvotes').eq('id', postId).single();
      if (error) throw error;
      const newUpvotes = (data.upvotes || 0) + 1;
      await supabase.from('forum_posts').update({ upvotes: newUpvotes }).eq('id', postId);
      return { upvotes: newUpvotes };
    } catch (error) {
      throw error;
    }
  },

  decrementPostUpvotes: async (postId) => {
    try {
      const { data, error } = await supabase.from('forum_posts').select('upvotes').eq('id', postId).single();
      if (error) throw error;
      const newUpvotes = Math.max(0, (data.upvotes || 0) - 1);
      await supabase.from('forum_posts').update({ upvotes: newUpvotes }).eq('id', postId);
      return { upvotes: newUpvotes };
    } catch (error) {
      throw error;
    }
  },

  getFarmerForumPosts: async () => {
    try {
      const { data, error } = await supabase.from('farmer_forum').select('*').order('created_at', { ascending: false });
      if (error) throw error;
      return data || [];
    } catch (error) {
      throw error;
    }
  },

  searchFarmerForumByKeywords: async (keywords, community) => {
    try {
      const keywordArray = keywords.split(',').map(k => k.trim().toLowerCase()).filter(k => k);
      let query = supabase.from('farmer_forum').select('*');
      if (community) {
        query = query.eq('community', community);
      }
      const { data: allPosts, error } = await query;
      if (error) throw error;
      if (!allPosts || allPosts.length === 0) return [];
      if (keywordArray.length === 0) {
        return allPosts.slice(0, 10);
      }
      const rankedPosts = allPosts.map(post => {
        const postKeywords = (post.highlighted_keywords || '').toLowerCase();
        const postQuestion = (post.question || '').toLowerCase();
        const postAnswer = (post.answer || '').toLowerCase();
        let matchScore = 0;
        keywordArray.forEach(keyword => {
          if (postKeywords.includes(keyword)) matchScore += 3;
          if (postQuestion.includes(keyword)) matchScore += 2;
          if (postAnswer.includes(keyword)) matchScore += 1;
        });
        return { ...post, matchScore };
      });
      rankedPosts.sort((a, b) => {
        if (b.matchScore !== a.matchScore) {
          return b.matchScore - a.matchScore;
        }
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      });
      return rankedPosts.slice(0, 10).map(({ matchScore, ...post }) => post);
    } catch (error) {
      throw error;
    }
  },

  // ==== ADMIN HELPERS ====

  getAllUsers: async () => {
    try {
      const { data, error } = await supabase
        .from('users')
        .select('id, email, role, phone, created_at')
        .order('id', { ascending: true });
      if (error) throw error;
      return data || [];
    } catch (error) {
      throw error;
    }
  },

  getAllProfiles: async () => {
    try {
      const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .order('id', { ascending: true });
      if (error) throw error;
      return data || [];
    } catch (error) {
      throw error;
    }
  },

  deleteUser: async (userId) => {
    try {
      // Delete from profiles first (foreign key)
      await supabase.from('profiles').delete().eq('id', userId);
      // Delete from users
      const { error } = await supabase.from('users').delete().eq('id', userId);
      if (error) throw error;
      return { changes: 1 };
    } catch (error) {
      throw error;
    }
  },

  updateUserPassword: async (userId, newPasswordHash) => {
    try {
      const { error } = await supabase
        .from('users')
        .update({ password_hash: newPasswordHash })
        .eq('id', userId);
      if (error) throw error;
      return { changes: 1 };
    } catch (error) {
      throw error;
    }
  },

  getAllExperts: async () => {
    try {
      const { data, error } = await supabase
        .from('experts_info')
        .select('*')
        .order('id', { ascending: true });
      if (error) throw error;
      return data || [];
    } catch (error) {
      throw error;
    }
  },

  createExpert: async (expertData) => {
    try {
      const { name, experience_years, specializations, rating, phone_number } = expertData;
      const { data, error } = await supabase
        .from('experts_info')
        .insert([{
          name,
          experience_years: experience_years || 0,
          specializations: specializations || '',
          rating: rating || 0,
          consultation_count: 0,
          phone_number: phone_number || ''
        }])
        .select()
        .single();
      if (error) throw error;
      return { id: data.id };
    } catch (error) {
      throw error;
    }
  },

  updateExpert: async (expertId, expertData) => {
    try {
      const { name, experience_years, specializations, rating, phone_number } = expertData;
      const { error } = await supabase
        .from('experts_info')
        .update({
          name, experience_years, specializations, rating, phone_number,
          updated_at: new Date().toISOString()
        })
        .eq('id', expertId);
      if (error) throw error;
      return { changes: 1 };
    } catch (error) {
      throw error;
    }
  },

  deleteExpert: async (expertId) => {
    try {
      const { error } = await supabase
        .from('experts_info')
        .delete()
        .eq('id', expertId);
      if (error) throw error;
      return { changes: 1 };
    } catch (error) {
      throw error;
    }
  }
};

// SQLite-compatible wrapper for backward compatibility
// This allows server.js to use db.all(), db.get(), db.run() syntax with Supabase
const db = {
  all: (query, params, callback) => {
    // Map common SQL queries to Supabase operations
    const normalizedQuery = query.trim().toLowerCase();

    (async () => {
      try {
        let result = [];

        // Forum posts with replies
        if (normalizedQuery.includes('from forum_posts')) {
          result = await dbHelpers.getForumPosts();
          if (params && params.length > 0 && normalizedQuery.includes('where category')) {
            const category = params[0];
            if (category && category !== 'All') {
              result = result.filter(p => p.category === category);
            }
          }
        }
        // Forum replies
        else if (normalizedQuery.includes('from forum_replies')) {
          const allPosts = await dbHelpers.getForumPosts();

          // Check if filtering by post_id
          if (params && params.length > 0 && normalizedQuery.includes('where')) {
            const postId = params[0];
            const post = allPosts.find(p => p.id === parseInt(postId));
            result = post ? (post.replies || []) : [];
          } else {
            // Return ALL replies (flatten from all posts)
            result = allPosts.flatMap(post => post.replies || []);
          }
        }
        // Users with profiles
        else if (normalizedQuery.includes('from users u') && normalizedQuery.includes('left join profiles')) {
          const users = await dbHelpers.getAllUsers();
          const profiles = await dbHelpers.getAllProfiles();
          result = users.map(user => {
            const profile = profiles.find(p => p.id === user.id);
            return { ...user, ...profile };
          });
        }
        // Generic fallback - just return empty array
        else {
          console.warn(`db.all: Unhandled query pattern: ${query.substring(0, 100)}...`);
          result = [];
        }

        callback(null, result);
      } catch (error) {
        console.error('db.all error:', error);
        callback(error, null);
      }
    })();
  },

  get: (query, params, callback) => {
    const normalizedQuery = query.trim().toLowerCase();

    (async () => {
      try {
        let result = null;

        // User by ID
        if (normalizedQuery.includes('from users') && normalizedQuery.includes('where') && normalizedQuery.includes('id')) {
          const userId = params[0];
          result = await dbHelpers.findUserById(userId);
          if (result && normalizedQuery.includes('left join profiles')) {
            const profile = await dbHelpers.findProfileByUserId(userId);
            result = { ...result, ...profile };
          }
        }
        // Forum post by ID
        else if (normalizedQuery.includes('from forum_posts') && normalizedQuery.includes('where')) {
          const allPosts = await dbHelpers.getForumPosts();
          const postId = params[0];
          result = allPosts.find(p => p.id === parseInt(postId));
        }
        // Forum reply
        else if (normalizedQuery.includes('from forum_replies')) {
          const allPosts = await dbHelpers.getForumPosts();
          const postId = params[0];
          const post = allPosts.find(p => p.id === parseInt(postId));
          result = post && post.replies && post.replies.length > 0 ? post.replies[0] : null;
        }
        // Generic fallback
        else {
          console.warn(`db.get: Unhandled query pattern: ${query.substring(0, 100)}...`);
        }

        callback(null, result);
      } catch (error) {
        console.error('db.get error:', error);
        callback(error, null);
      }
    })();
  },

  run: function (query, params, callback) {
    const normalizedQuery = query.trim().toLowerCase();
    const self = { lastID: null, changes: 0 };

    (async () => {
      try {
        // INSERT operations
        if (normalizedQuery.startsWith('insert into forum_posts')) {
          const [userId, category, community, question, keywords] = params;
          const result = await dbHelpers.createForumPost(userId, category, question, community, keywords);
          self.lastID = result.id;
          self.changes = 1;
          callback.call(self, null);
        }
        else if (normalizedQuery.startsWith('insert into forum_replies')) {
          const [postId, replyText, repliedBy] = params;
          const result = await dbHelpers.createForumReply(postId, replyText, repliedBy);
          self.lastID = result.id;
          self.changes = 1;
          callback.call(self, null);
        }
        else if (normalizedQuery.startsWith('insert into experts_info')) {
          const [name, experience_years, specializations, phone_number] = params;
          const result = await dbHelpers.createExpert({ name, experience_years, specializations, phone_number, rating: 0 });
          self.lastID = result.id;
          self.changes = 1;
          callback.call(self, null);
        }
        // UPDATE operations
        else if (normalizedQuery.startsWith('update users') && normalizedQuery.includes('password_hash')) {
          const [hashedPassword, userId] = params;
          await dbHelpers.updateUserPassword(userId, hashedPassword);
          self.changes = 1;
          callback.call(self, null);
        }
        else if (normalizedQuery.startsWith('update users') && !normalizedQuery.includes('password')) {
          // Update user info
          const [email, role, phone, userId] = params;
          await supabase.from('users').update({ email, role, phone }).eq('id', userId);
          self.changes = 1;
          callback.call(self, null);
        }
        else if (normalizedQuery.startsWith('update profiles')) {
          const [full_name, phone_number, location, crops_grown, available_quantity, expected_price, userId] = params;
          await dbHelpers.updateProfile(userId, { full_name, phone_number, location, crops_grown, available_quantity, expected_price });
          self.changes = 1;
          callback.call(self, null);
        }
        else if (normalizedQuery.startsWith('update experts_info')) {
          const [name, experience_years, specializations, phone_number, rating, expertId] = params;
          await dbHelpers.updateExpert(expertId, { name, experience_years, specializations, phone_number, rating });
          self.changes = 1;
          callback.call(self, null);
        }
        // DELETE operations
        else if (normalizedQuery.startsWith('delete from users')) {
          const userId = params[0];
          await dbHelpers.deleteUser(userId);
          self.changes = 1;
          callback.call(self, null);
        }
        else if (normalizedQuery.startsWith('delete from experts_info')) {
          const expertId = params[0];
          await dbHelpers.deleteExpert(expertId);
          self.changes = 1;
          callback.call(self, null);
        }
        // Generic fallback
        else {
          console.warn(`db.run: Unhandled query pattern: ${query.substring(0, 100)}...`);
          callback.call(self, new Error('Unsupported query pattern'));
        }
      } catch (error) {
        console.error('db.run error:', error);
        callback.call(self, error);
      }
    })();
  }
};

module.exports = {
  initDatabase,
  dbHelpers,
  db  // Export db for backward compatibility
};
