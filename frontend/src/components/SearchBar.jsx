import { useState, useRef } from 'react'
import Toggle from './Toggle'

export default function SearchBar({ onSearch }) {
    const [mode, setMode] = useState('text')
    const [query, setQuery] = useState('')
    const [file, setFile] = useState(null)
    const [preview, setPreview] = useState(null)
    const fileInputRef = useRef(null)

    function handleModeChange(newMode) {
        setMode(newMode)
        setQuery('')
        setFile(null)
        setPreview(null)
    }

    function handleFileChange(e) {
        const selected = e.target.files[0]
        if (!selected) return

        if (preview) {
            URL.revokeObjectURL(preview)
        }

        setFile(selected)
        setPreview(URL.createObjectURL(selected))
    }

    function handleSubmit(e) {
        e.preventDefault()
        if (mode === 'text' && query.trim()) {
            onSearch({ mode, query })
        } else if (mode === 'image' && file) {
            onSearch({ mode, file })
        }
    }

    return (
        <form onSubmit={handleSubmit} className="w-full max-w-2xl mx-auto">
            <div className="flex flex-col gap-4">
                <div className="flex items-center justify-between">
                    <Toggle mode={mode} onChange={handleModeChange} />
                </div>

                {mode === 'text' ? (
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={query}
                            onChange={e => setQuery(e.target.value)}
                            placeholder="Search for fashion items..."
                            className="flex-1 px-4 py-3 rounded-xl border border-zinc-200 bg-white text-zinc-900 placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-900 text-sm"
                        />
                        <button
                            type="submit"
                            disabled={!query.trim()}
                            className="px-6 py-3 bg-zinc-900 text-white rounded-xl text-sm font-medium hover:bg-zinc-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                        >
                            Search
                        </button>
                    </div>
                ) : (
                    <div className="flex flex-col gap-3">
                        <div
                            onClick={() => fileInputRef.current.click()}
                            className="w-full h-40 border-2 border-dashed border-zinc-200 rounded-xl flex flex-col items-center justify-center cursor-pointer hover:border-zinc-400 transition-colors"
                        >
                            {preview ? (
                                <img
                                    src={preview}
                                    alt="preview"
                                    className="h-full w-full object-contain rounded-xl"
                                />
                            ) : (
                                <div className="text-center">
                                    <p className="text-zinc-400 text-sm">Click to upload an image</p>
                                    <p className="text-zinc-300 text-xs mt-1">PNG, JPG supported</p>
                                </div>
                            )}
                        </div>
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept="image/*"
                            onChange={handleFileChange}
                            className="hidden"
                        />
                        <button
                            type="submit"
                            disabled={!file}
                            className="px-6 py-3 bg-zinc-900 text-white rounded-xl text-sm font-medium hover:bg-zinc-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                        >
                            Search
                        </button>
                    </div>
                )}
            </div>
        </form>
    )
}