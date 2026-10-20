import { useEffect, useState } from 'react'
import { Link, NavLink, Outlet, useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../auth'

export function Logo() {
  return (
    <Link to="/" className="flex items-center gap-2 text-lg font-extrabold tracking-tight text-slate-900">
      <span className="grid h-8 w-8 place-items-center rounded-lg bg-brand-600 text-white">▶</span>
      VideoAI
    </Link>
  )
}

export default function Layout() {
  const { user, logout } = useAuth()
  const nav = useNavigate()
  const [sys, setSys] = useState(null)
  useEffect(() => { api.system().then(setSys).catch(() => setSys(false)) }, [])
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6">
          <Logo />
          <nav className="flex items-center gap-2 sm:gap-4 text-sm font-medium">
            {user ? (
              <>
                <NavLink to="/dashboard" className="text-slate-600 hover:text-slate-900">My videos</NavLink>
                <span className="hidden text-slate-400 sm:inline">{user.is_guest ? 'Guest' : user.email}</span>
                <button className="btn-secondary" onClick={() => { logout(); nav('/') }}>Log out</button>
              </>
            ) : (
              <>
                <Link to="/login" className="text-slate-600 hover:text-slate-900">Log in</Link>
                <Link to="/signup" className="btn-primary">Sign up free</Link>
              </>
            )}
          </nav>
        </div>
      </header>
      <main className="flex-1"><Outlet /></main>
      <footer className="border-t border-slate-200 bg-white py-6 text-center text-sm text-slate-500">
        VideoAI — free, open-source AI video creator.{' '}
        {sys && <span>v{sys.version}{sys.commit ? ` (${sys.commit})` : ''} · </span>}
        {sys === false && <span className="text-red-600">backend not reachable · </span>}
        <Link to="/system" className="underline hover:text-slate-700">System check</Link>
      </footer>
    </div>
  )
}
