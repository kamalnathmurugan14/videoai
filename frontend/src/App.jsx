import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { useAuth } from './auth'
import Layout from './components/Layout'
import { PageLoader } from './components/ui'
import Landing from './pages/Landing'
import { Login, Signup } from './pages/Auth'
import Dashboard from './pages/Dashboard'
import Editor from './pages/Editor'
import VideoDetail from './pages/VideoDetail'
import SystemCheck from './pages/SystemCheck'

function Protected({ children }) {
  const { user, loading } = useAuth()
  const loc = useLocation()
  if (loading) return <PageLoader />
  return user ? children : <Navigate to="/login" replace state={{ from: loc.pathname }} />
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Landing />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/dashboard" element={<Protected><Dashboard /></Protected>} />
        <Route path="/projects/:id/edit" element={<Protected><Editor /></Protected>} />
        <Route path="/videos/:id" element={<Protected><VideoDetail /></Protected>} />
        <Route path="/system" element={<SystemCheck />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
