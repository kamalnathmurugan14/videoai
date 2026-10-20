import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth'
import { ErrorBox, Spinner } from '../components/ui'

const FEATURES = [
  ['🧑‍🎤', 'Animated avatars', 'Pick from stylised presenters or upload your own image. The mouth follows the voice.'],
  ['🗣️', 'Natural voices', 'Dozens of free voices across 20+ languages, with speed control and instant preview.'],
  ['🎬', 'Multi-scene scripts', 'Build videos scene by scene with backgrounds, images and background music.'],
  ['💬', 'Auto captions', 'Burned-in captions with word highlighting in three styles.'],
  ['📱', 'Any format', '16:9 for YouTube, 9:16 for Reels and TikTok, 1:1 for feeds.'],
  ['🆓', 'Free for everyone', 'No watermark, no paywall. Sign up or just try it as a guest.'],
]
const STEPS = [
  ['1', 'Write your script', 'Start from a template, paste text, or import a blog post URL.'],
  ['2', 'Choose avatar & voice', 'Pick a presenter, a language and a style. Preview the voice instantly.'],
  ['3', 'Render & share', 'We render an MP4 in the background. Preview it and download.'],
]

export default function Landing() {
  const { user, guest } = useAuth()
  const nav = useNavigate()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const tryGuest = async () => {
    setBusy(true)
    setError('')
    try {
      if (!user) await guest()
      nav('/dashboard')
    } catch (e) {
      setError(e.message)
      setBusy(false)
    }
  }

  return (
    <>
      <section className="bg-gradient-to-b from-brand-50 to-slate-50">
        <div className="mx-auto max-w-5xl px-4 py-20 text-center sm:py-28">
          <span className="rounded-full bg-white px-4 py-1 text-sm font-semibold text-brand-600 shadow-sm">100% free · open source</span>
          <h1 className="mt-6 text-4xl font-extrabold tracking-tight text-slate-900 sm:text-6xl">
            Turn text into <span className="text-brand-600">talking-avatar videos</span>
          </h1>
          <p className="mx-auto mt-6 max-w-2xl text-lg text-slate-600">
            Write a script, pick an avatar and a voice, and get a captioned MP4 in minutes. No camera, no studio, no subscription.
          </p>
          <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
            <button className="btn-primary px-6 py-3 text-base" onClick={tryGuest} disabled={busy}>
              {busy && <Spinner className="h-4 w-4 text-white" />} {user ? 'Go to dashboard' : 'Try it now — no signup'}
            </button>
            {!user && <Link to="/signup" className="btn-secondary px-6 py-3 text-base">Create free account</Link>}
          </div>
          <div className="mx-auto mt-4 max-w-md"><ErrorBox>{error}</ErrorBox></div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-4 py-16 sm:px-6">
        <h2 className="text-center text-3xl font-bold text-slate-900">Everything you need to make a video</h2>
        <div className="mt-10 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map(([icon, title, text]) => (
            <div key={title} className="card p-6">
              <div className="text-3xl">{icon}</div>
              <h3 className="mt-3 text-lg font-semibold">{title}</h3>
              <p className="mt-1 text-sm text-slate-600">{text}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="bg-white py-16">
        <div className="mx-auto max-w-5xl px-4 sm:px-6">
          <h2 className="text-center text-3xl font-bold text-slate-900">How it works</h2>
          <div className="mt-10 grid gap-8 md:grid-cols-3">
            {STEPS.map(([n, title, text]) => (
              <div key={n} className="text-center">
                <div className="mx-auto grid h-12 w-12 place-items-center rounded-full bg-brand-600 text-lg font-bold text-white">{n}</div>
                <h3 className="mt-4 text-lg font-semibold">{title}</h3>
                <p className="mt-1 text-sm text-slate-600">{text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-4xl px-4 py-20 text-center">
        <div className="rounded-3xl bg-brand-600 px-6 py-14 text-white">
          <h2 className="text-3xl font-bold">Ready to create your first video?</h2>
          <p className="mt-3 text-brand-100">It takes about two minutes. Free, forever.</p>
          <button onClick={tryGuest} disabled={busy} className="btn mt-6 bg-white px-6 py-3 text-base text-brand-700 hover:bg-brand-50">Get started</button>
        </div>
      </section>
    </>
  )
}
