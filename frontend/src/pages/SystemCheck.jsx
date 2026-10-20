import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../auth'
import { ErrorBox, PageLoader, Spinner } from '../components/ui'

const Row = ({ ok, label, detail }) => (
  <li className="flex items-start gap-3 py-3">
    <span className={`mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full text-sm font-bold ${ok ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'}`}>{ok ? '✓' : '!'}</span>
    <div><div className="font-medium">{label}</div>{detail && <div className="text-sm text-slate-500">{detail}</div>}</div>
  </li>
)

export default function SystemCheck() {
  const { user } = useAuth()
  const [info, setInfo] = useState(null)
  const [error, setError] = useState('')
  const [voices, setVoices] = useState(null)
  const [busy, setBusy] = useState(false)
  useEffect(() => { api.system().then(setInfo).catch((e) => setError(e.message)) }, [])

  const testVoices = async () => {
    setBusy(true)
    setError('')
    try { setVoices(await api.voiceCheck()) } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  if (!info) return error ? <div className="mx-auto max-w-2xl p-8"><ErrorBox>{error}</ErrorBox></div> : <PageLoader />
  return (
    <div className="mx-auto max-w-2xl px-4 py-10">
      <h1 className="text-2xl font-bold">System check</h1>
      <p className="mt-1 text-slate-600">Running version <b>{info.version}</b>{info.commit && <> (commit <code>{info.commit}</code>)</>}. If this is not the latest version, update with the commands in the README and restart the backend.</p>
      <ul className="card mt-6 divide-y divide-slate-100 px-5">
        <Row ok={info.face_tracking} label="Precise face tracking (478 landmarks)"
          detail={info.face_tracking ? 'The mouth and eyes are located exactly on uploaded photos.' : 'Not available: the package ai-edge-litert is missing or failed to load, so a less accurate detector is used. Run: pip install -r requirements.txt'} />
        <Row ok={info.segmentation} label="Person cut-out model"
          detail={info.segmentation ? 'Backgrounds are removed while keeping hair, shoulders and clothes.' : 'Not available: a simpler colour-based cut-out is used, which can lose shoulders on busy backgrounds.'} />
        <Row ok={!!info.ffmpeg} label="FFmpeg (video encoding)" detail={info.ffmpeg || 'Not found on PATH: rendering will fail. Install FFmpeg and restart the backend.'} />
        <Row ok label="Voice engines (tried in this order)" detail={info.tts_order.join(' → ')} />
      </ul>

      <div className="card mt-6 p-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h2 className="font-semibold">Test voices on this computer</h2>
            <p className="text-sm text-slate-500">Synthesises a short phrase with every engine and shows exactly which ones work.</p>
          </div>
          {user ? <button className="btn-primary" disabled={busy} onClick={testVoices}>{busy && <Spinner className="h-4 w-4 text-white" />} Test voices</button>
            : <Link to="/login" className="btn-secondary">Log in to test</Link>}
        </div>
        <div className="mt-3"><ErrorBox>{error}</ErrorBox></div>
        {voices && (
          <>
            {!voices.best_quality_available && (
              <p role="alert" className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
                The Microsoft neural voices (edge) are not reachable from this computer, so a fallback engine will be used and the
                voice will not match your choice exactly. Check the internet connection, firewall/proxy or antivirus for
                <code> speech.platform.bing.com</code>, then test again.
              </p>
            )}
            <ul className="mt-3 divide-y divide-slate-100">
              {voices.engines.map((e) => (
                <Row key={e.engine} ok={e.ok} label={`${e.engine}${e.ok ? ` — works (${e.seconds}s)` : ' — failed'}`} detail={e.error} />
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  )
}
