import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { api, getToken, setToken } from './api'

const AuthCtx = createContext(null)
export const useAuth = () => useContext(AuthCtx)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(!!getToken())

  const logout = useCallback(() => {
    setToken(null)
    setUser(null)
  }, [])

  useEffect(() => {
    if (!getToken()) return
    api.me().then(setUser).catch(() => setToken(null)).finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    window.addEventListener('videoai:unauthorized', logout)
    return () => window.removeEventListener('videoai:unauthorized', logout)
  }, [logout])

  const accept = (res) => {
    setToken(res.access_token)
    setUser(res.user)
    return res.user
  }
  const value = {
    user, loading, logout,
    login: async (e, p) => accept(await api.login(e, p)),
    signup: async (e, p) => accept(await api.signup(e, p)),
    guest: async () => accept(await api.guest()),
  }
  return <AuthCtx.Provider value={value}>{children}</AuthCtx.Provider>
}
