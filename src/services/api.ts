// API functions for NGO schemes, soil labs, and crop history
const API_BASE_URL = import.meta.env.VITE_API_URL ||
    (import.meta.env.PROD
        ? 'https://farm-backend-dqsw.onrender.com/api'
        : 'http://localhost:3001/api');

export interface NgoScheme {
    id: number;
    name: string;
    ministry?: string;
    deadline?: string;
    location?: string;
    contact_number?: string;
    no_of_docs_required?: number;
    status?: string;
    benefit_text?: string;
    eligibility_text?: string;
    // New eligibility fields
    scheme_type?: string;
    required_state?: string;
    min_land?: number;
    max_land?: number;
    required_category?: string;
    age_min?: number;
    age_max?: number;
    official_link?: string;
    created_at: string;
    updated_at: string;
}

export interface EligibilityFilters {
    state?: string;
    land?: number;
    category?: string;
    age?: number;
    crop?: string;
}


export interface SoilLab {
    id: number;
    name: string;
    location?: string;
    contact_number?: string;
    price?: number;
    rating?: number;
    tag?: string;
    created_at: string;
    updated_at: string;
}

export interface CropHistory {
    id: number;
    user_id: number;
    crop_name: string;
    crop_price: number;
    selling_price: number;
    crop_produced_kg: number;
    created_at: string;
    updated_at: string;
}

async function makeRequest<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    const config: RequestInit = {
        credentials: 'include',
        headers: {
            'Content-Type': 'application/json',
            ...options.headers,
        },
        ...options,
    };

    const response = await fetch(url, config);

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Request failed: ${response.status} ${errorText}`);
    }

    return response.json();
}

// ========== NGO SCHEMES API ==========

export const getNgoSchemes = (): Promise<NgoScheme[]> => {
    return makeRequest<NgoScheme[]>('/ngo-schemes');
};

export const getNgoSchemeById = (id: number): Promise<NgoScheme> => {
    return makeRequest<NgoScheme>(`/ngo-schemes/${id}`);
};

export const createNgoScheme = (data: Omit<NgoScheme, 'id' | 'created_at' | 'updated_at'>): Promise<{ id: number; message: string }> => {
    return makeRequest('/ngo-schemes', {
        method: 'POST',
        body: JSON.stringify(data),
    });
};

export const updateNgoScheme = (id: number, data: Partial<NgoScheme>): Promise<{ message: string }> => {
    return makeRequest(`/ngo-schemes/${id}`, {
        method: 'PUT',
        body: JSON.stringify(data),
    });
};

export const deleteNgoScheme = (id: number): Promise<{ message: string }> => {
    return makeRequest(`/ngo-schemes/${id}`, {
        method: 'DELETE',
    });
};

// ========== GOVERNMENT SCHEMES ELIGIBILITY FILTER API ==========

export const filterEligibleSchemes = (filters: EligibilityFilters): Promise<NgoScheme[]> => {
    console.log('🔍 Calling filterEligibleSchemes API with:', filters);
    return makeRequest<NgoScheme[]>('/government-schemes/filter', {
        method: 'POST',
        body: JSON.stringify(filters),
    });
};


// ========== SOIL LABS API ==========

export const getSoilLabs = (): Promise<SoilLab[]> => {
    return makeRequest<SoilLab[]>('/soil-labs');
};

export const getSoilLabById = (id: number): Promise<SoilLab> => {
    return makeRequest<SoilLab>(`/soil-labs/${id}`);
};

export const createSoilLab = (data: Omit<SoilLab, 'id' | 'created_at' | 'updated_at'>): Promise<{ id: number; message: string }> => {
    return makeRequest('/soil-labs', {
        method: 'POST',
        body: JSON.stringify(data),
    });
};

export const updateSoilLab = (id: number, data: Partial<SoilLab>): Promise<{ message: string }> => {
    return makeRequest(`/soil-labs/${id}`, {
        method: 'PUT',
        body: JSON.stringify(data),
    });
};

export const deleteSoilLab = (id: number): Promise<{ message: string }> => {
    return makeRequest(`/soil-labs/${id}`, {
        method: 'DELETE',
    });
};

// ========== CROP HISTORY API ==========

export const getCrops = (): Promise<CropHistory[]> => {
    return makeRequest<CropHistory[]>('/crops');
};

export const createCrop = (data: Omit<CropHistory, 'id' | 'user_id' | 'created_at' | 'updated_at'>): Promise<{ id: number; message: string }> => {
    return makeRequest('/crops', {
        method: 'POST',
        body: JSON.stringify(data),
    });
};

export const updateCrop = (id: number, data: Partial<CropHistory>): Promise<{ message: string }> => {
    return makeRequest(`/crops/${id}`, {
        method: 'PUT',
        body: JSON.stringify(data),
    });
};

export const deleteCrop = (id: number): Promise<{ message: string }> => {
    return makeRequest(`/crops/${id}`, {
        method: 'DELETE',
    });
};
