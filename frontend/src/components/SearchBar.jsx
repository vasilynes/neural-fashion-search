import { useState, useRef } from 'react'

export default function SearchBar({ onSearch }) {
    const [query, setQuery] = useState('')
    const [file, setFile] = useState(null)
    const [preview, setPreview] = useState(null)
    const [beta, setBeta] = useState(0.2)
    const fileInputRef = useRef(null)

    function handleFileChange(e) {
        const selected = e.target.files[0]
        if (!selected) return
        if (preview) {
            URL.revokeObjectURL(preview)
        }
        setFile(selected)
        setPreview(URL.createObjectURL(selected))
    }

    function handleRemoveImage() {
        if (preview) {
            URL.revokeObjectURL(preview)
        }
        setFile(null)
        setPreview(null)
        fileInputRef.current.value = ''
    }

    function handleSubmit(e) {
        e.preventDefault()
        if (file) {
            onSearch({ file, query: query.trim() || undefined, beta})
        } else if (query.trim()) {
            onSearch({ query })
        }
    }
    return (
        <form onSubmit={handleSubmit} className="w-full max-w-2xl mx-auto">
            <div className="flex flex-col gap-3">
                {preview && (
                    <div className="relative w-16 h-16">
                        <img src={preview} alt="preview" className="w-full h-full object-cover rounded-lg border border-zinc-200" />
                        <button
                            type="button"
                            onClick={handleRemoveImage}
                            className="absolute -top-1 -right-1 bg-zinc-800 text-white rounded-full w-4 h-4 text-xs flex items-center justify-center hover:bg-zinc-600"
                        >
                            ×
                        </button>
                    </div>
                )}
                {file && (
                    <div className="flex items-center gap-3 w-fit">
                        <span className="text-xs text-zinc-400 whitespace-nowrap">Image weight: {beta.toFixed(1)}</span>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.1"
                            value={beta}
                            onChange={e => setBeta(parseFloat(e.target.value))}
                            className="w-24 slider-thin"
                        />
                    </div>
                )}
                <div className="flex gap-2">
                    <div className="relative flex-1 flex items-center">
                        <input
                            type="text"
                            value={query}
                            onChange={e => setQuery(e.target.value)}
                            placeholder={file ? "Optional: describe specifications..." : "Search for fashion products..."}
                            className="w-full pl-4 pr-12 py-3 rounded-xl border border-zinc-200 bg-white text-zinc-900 placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-900 text-sm"
                        />
                        <button
                            type="button"
                            onClick={() => fileInputRef.current.click()}
                            className="absolute right-3 text-zinc-400 hover:text-zinc-700 transition-colors"
                            title="Search by image"
                        >
                            {/* Camera icon */}
                            <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                            </svg>
                        </button>
                    </div>
                    <button
                        type="submit"
                        disabled={!query.trim() && !file}
                        className="px-6 py-3 bg-zinc-900 text-white rounded-xl text-sm font-medium hover:bg-zinc-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    >
                        Search
                    </button>
                </div>
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="hidden"
                />
            </div>
        </form>
    )
}