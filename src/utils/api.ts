/**
 * Get the API base URL for making backend requests
 */
export function getAPIBaseURL(): string {
    // Check if we're in development mode (Vite dev server)
    const isDevelopment = import.meta.env.DEV;

    if (isDevelopment) {
        // Development: use localhost backend
        return 'http://localhost:3001/api';
    } else {
        // Production: use relative path (assumes frontend and backend on same domain)
        return '/api';
    }
}
