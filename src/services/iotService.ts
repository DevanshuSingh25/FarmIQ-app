// IoT Sensor Service for FarmIQ
const API_BASE_URL = import.meta.env.VITE_API_URL ||
  (import.meta.env.PROD
    ? 'https://farm-backend-dqsw.onrender.com/api'
    : 'http://localhost:3001/api');

export interface InstallationRequest {
  id: number;
  user_id: number;
  name: string;
  phone_number: string;
  location?: string;
  state?: string;
  district?: string;
  preferred_visit_date?: string;
  status: 'pending' | 'acknowledged' | 'completed' | 'cancelled';
  created_at: string;
  updated_at?: string;
}

export interface IotStatus {
  user_id: number;
  status: 'inactive' | 'active' | 'booked';
  updated_at: string;
  note?: string;
}

export interface Reading {
  timestamp: string;
  temperature: number;
  humidity: number;
  soil_moisture: number;
}

class IoTService {
  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    const config: RequestInit = {
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    console.log('Making IoT request to:', url);
    const response = await fetch(url, config);

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Request failed: ${response.status} ${errorText}`);
    }

    return await response.json();
  }

  // Get IoT status for a user
  async getStatus(userId: number): Promise<IotStatus> {
    return this.makeRequest<IotStatus>(`/iot/status/${userId}`);
  }

  // Get booking request for a user
  async getBookingRequest(userId: number): Promise<InstallationRequest | null> {
    try {
      return await this.makeRequest<InstallationRequest>(`/iot/request/${userId}`);
    } catch (error) {
      // Return null if 404 (no booking found)
      if (error instanceof Error && error.message.includes('404')) {
        return null;
      }
      throw error;
    }
  }

  // Create new booking request (user_id comes from session on backend)
  async createRequest(requestData: {
    name: string;
    phone_number: string;
    location?: string;
    state?: string;
    district?: string;
    preferred_visit_date?: string;
  }): Promise<InstallationRequest> {
    return this.makeRequest<InstallationRequest>('/iot/request', {
      method: 'POST',
      body: JSON.stringify(requestData),
    });
  }

  // Get sensor readings for a user (only works if status is 'active')
  async getReadings(userId: number, limit: number = 24): Promise<Reading[]> {
    return this.makeRequest<Reading[]>(`/iot/readings/${userId}?limit=${limit}`);
  }

  // Update IoT status (admin only)
  async updateStatus(userId: number, status: 'inactive' | 'active' | 'booked'): Promise<IotStatus> {
    return this.makeRequest<IotStatus>(`/iot/status/${userId}`, {
      method: 'PUT',
      body: JSON.stringify({ status }),
    });
  }

  // Blynk LED control methods
  async getBlynkLedStatus(): Promise<{ state: boolean; value: number; timestamp: string }> {
    return this.makeRequest(`/blynk/led/status`);
  }

  async setBlynkLedState(state: boolean): Promise<{ success: boolean; state: boolean; value: number; timestamp: string }> {
    return this.makeRequest(`/blynk/led/control`, {
      method: 'POST',
      body: JSON.stringify({ state }),
    });
  }
}

export const iotService = new IoTService();
