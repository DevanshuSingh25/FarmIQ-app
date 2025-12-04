// Authentication service for FarmIQ
const API_BASE_URL = import.meta.env.VITE_API_URL ||
  (import.meta.env.PROD
    ? 'https://farm-backend-dqsw.onrender.com/api'
    : 'http://localhost:3001/api');

// Debug logging for production
console.log('Environment:', import.meta.env.MODE);
console.log('API Base URL:', API_BASE_URL);
console.log('VITE_API_URL:', import.meta.env.VITE_API_URL);

export interface User {
  id: number;
  role: 'farmer' | 'vendor' | 'admin';
  email: string;
  phone?: string;
}

export interface LoginCredentials {
  role: 'farmer' | 'vendor' | 'admin';
  email: string;  // email only, no username
  password: string;
}

export interface RegisterData {
  role: 'farmer' | 'vendor' | 'admin';
  full_name: string;
  email: string;
  phone: string;
  password: string;
  language_pref?: string;
  location?: string;
  crops_grown?: string;
  available_quantity?: string;
  expected_price?: string;
}

export interface ProfileData {
  id: number;
  full_name: string;
  email: string;
  phone_number: string;
  language_pref: string;
  created_at: string;
  updated_at: string;
}

export interface AuthResponse {
  success: boolean;
  user?: User;
  redirectUrl?: string;
  message?: string;
}

export interface SessionResponse {
  authenticated: boolean;
  user?: User;
}

class AuthService {
  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    const config: RequestInit = {
      credentials: 'include', // Include cookies for session
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      console.log('Making request to:', url); // Debug log
      console.log('Request config:', config); // Debug log

      const response = await fetch(url, config);

      console.log('Response status:', response.status); // Debug log
      console.log('Response headers:', response.headers); // Debug log

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Request failed:', response.status, errorText);
        throw new Error(`Request failed: ${response.status} ${errorText}`);
      }

      const data = await response.json();
      console.log('Response data:', data); // Debug log
      return data;
    } catch (error) {
      console.error('Network error details:', error);
      console.error('Error name:', error instanceof Error ? error.name : 'Unknown');
      console.error('Error message:', error instanceof Error ? error.message : 'Unknown');

      if (error instanceof Error) {
        // Provide more specific error messages
        if (error.message.includes('Failed to fetch') || error.message.includes('ERR_CONNECTION_REFUSED')) {
          throw new Error(
            `Cannot connect to backend server at ${API_BASE_URL}. ` +
            `Please ensure the server is running on port 3001. ` +
            `Run: cd server && node server.js`
          );
        }
        throw error;
      }
      throw new Error(`Network error: Unable to connect to server at ${API_BASE_URL}`);
    }
  }

  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    return this.makeRequest<AuthResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify(credentials),
    });
  }

  async register(userData: RegisterData): Promise<{ ok: boolean; userId: number }> {
    return this.makeRequest<{ ok: boolean; userId: number }>('/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  }

  async getSession(): Promise<SessionResponse> {
    return this.makeRequest<SessionResponse>('/auth/session');
  }

  async logout(): Promise<{ ok: boolean }> {
    return this.makeRequest<{ ok: boolean }>('/auth/logout', {
      method: 'POST',
    });
  }

  // Helper method to get redirect URL based on role
  getRedirectUrl(role: 'farmer' | 'vendor' | 'admin'): string {
    switch (role) {
      case 'farmer':
        return '/farmer/dashboard';
      case 'vendor':
        return '/vendor/dashboard';
      case 'admin':
        return '/admin/dashboard';
      default:
        return '/login';
    }
  }

  // Get user profile
  async getProfile(): Promise<ProfileData> {
    return this.makeRequest<ProfileData>('/me/profile');
  }

  // Update user profile (only allowed fields)
  async updateProfile(profileData: Partial<ProfileData>): Promise<{ ok: boolean; message: string }> {
    return this.makeRequest<{ ok: boolean; message: string }>('/me/profile', {
      method: 'PUT',
      body: JSON.stringify(profileData),
    });
  }
}

export const authService = new AuthService();
