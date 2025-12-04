import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { authService, User, SessionResponse } from '@/services/authService';

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (user: User) => void;
  logout: () => Promise<void>;
  checkSession: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(() => {
    // Initialize from localStorage if available
    try {
      const savedUser = localStorage.getItem('farmiq_user');
      return savedUser ? JSON.parse(savedUser) : null;
    } catch {
      return null;
    }
  });
  const [isLoading, setIsLoading] = useState(true);

  const isAuthenticated = !!user;

  const login = (userData: User) => {
    setUser(userData);
    // Save to localStorage for persistence
    localStorage.setItem('farmiq_user', JSON.stringify(userData));
  };

  const logout = async () => {
    try {
      await authService.logout();
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setUser(null);
      // Clear localStorage on logout
      localStorage.removeItem('farmiq_user');
    }
  };

  const checkSession = async () => {
    try {
      setIsLoading(true);
      const response: SessionResponse = await authService.getSession();

      if (response.authenticated && response.user) {
        setUser(response.user);
        // Update localStorage with fresh user data
        localStorage.setItem('farmiq_user', JSON.stringify(response.user));
      } else {
        setUser(null);
        localStorage.removeItem('farmiq_user');
      }
    } catch (error) {
      console.error('Session check error:', error);
      // Don't clear user on network errors, only on auth errors
      if (error instanceof Error && error.message.includes('401')) {
        setUser(null);
        localStorage.removeItem('farmiq_user');
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    checkSession();

    // Set up periodic session check
    const interval = setInterval(() => {
      checkSession();
    }, 5 * 60 * 1000); // Check every 5 minutes

    return () => clearInterval(interval);
  }, []);

  const value: AuthContextType = {
    user,
    isAuthenticated,
    isLoading,
    login,
    logout,
    checkSession,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
