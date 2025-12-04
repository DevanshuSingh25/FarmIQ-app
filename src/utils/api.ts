/**
 * Get the API base URL for making backend requests
 */
export function getAPIBaseURL(): string {
    // Use VITE_API_URL from environment variables if available
    // This allows for flexible deployment configurations
    const apiUrl = import.meta.env.VITE_API_URL;

    if (apiUrl) {
        return apiUrl;
    }

    // Fallback to localhost in development
    if (import.meta.env.DEV) {
        return 'http://localhost:3001/api';
    }

    // Default fallback (shouldn't reach here in production if env is set correctly)
    return '/api';
}
