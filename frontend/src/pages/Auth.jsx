import { useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth'
import { ErrorBox, Spinner } from '../components/ui'

function AuthForm({ mode }) {
  const { login, signup, guest } = useAuth()
  const nav = useNavigate()
  const loc = useLocation()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const isSignup = mode === 'signup'

  const run = async (fn) => {
    setBusy(true)
    setError('')
    try {
      await fn()
      nav(loc.state?.from || '/dashboard', { replace: true })
    } catch (e) {
      setError(e.message)
      setBusy(false)
    }
  }

  return (
    <div className="mx-auto max-w-md px-4 py-16">
      <div className="card p-8">
        <h1 className="text-2xl font-bold">{isSignup ? 'Create your free account' : 'Welcome back'}</h1>
        <form className="mt-6 space-y-4" onSubmit={(e) => { e.preventDefault(); run(() => (isSignup ? signup : login)(email, password)) }}>
          <div>
            <label className="label" htmlFor="email">Email</label>
            <input id="email" className="input" type="email" required autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} />
          </div>
          <div>
            <label className="label" htmlFor="password">Password</label>
            <input id="password" className="input" type="password" required minLength={8} maxLength={72}
              autoComplete={isSignup ? 'new-password' : 'current-password'} value={password} onChange={(e) => setPassword(e.target.value)} />
            {isSignup && <p className="mt-1 text-xs text-slate-500">At least 8 characters.</p>}
          </div>
          <ErrorBox>{error}</ErrorBox>
          <button className="btn-primary w-full" disabled={busy}>
            {busy && <Spinner className="h-4 w-4 text-white" />} {isSignup ? 'Sign up' : 'Log in'}
          </button>
        </form>
        <div className="my-5 flex items-center gap-3 text-xs text-slate-400"><span className="h-px flex-1 bg-slate-200" />or<span className="h-px flex-1 bg-slate-200" /></div>
        <button className="btn-secondary w-full" disabled={busy} onClick={() => run(guest)}>Continue as guest</button>
        <p className="mt-6 text-center text-sm text-slate-600">
          {isSignup ? <>Already have an account? <Link className="font-semibold text-brand-600" to="/login">Log in</Link></>
            : <>New here? <Link className="font-semibold text-brand-600" to="/signup">Sign up free</Link></>}
        </p>
      </div>
    </div>
  )
}

export const Login = () => <AuthForm mode="login" />
export const Signup = () => <AuthForm mode="signup" />
