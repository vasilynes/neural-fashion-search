import { useState } from 'react'
import SearchBar from './components/SearchBar'
import ResultsGrid from './components/ResultsGrid'
import { searchByText, searchByImage } from './api/search'

export default function App() {
    const [results, setResults] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    async function handleSearch({ mode, query, file }) {
        setLoading(true)
        setError(null)
        setResults(null)

        try {
            const data = mode === 'text'
                ? await searchByText(query)
                : await searchByImage(file)
            setResults(data)
        } catch (err) {
            const detail = err.response?.data?.detail
            if (Array.isArray(detail)) {
                setError(detail.map(d => d.msg).join(', '))
            } else {
                setError(detail ?? 'Something went wrong. Please try again.')
            }
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-zinc-50">
            <div className="max-w-7xl mx-auto px-4 py-12 flex flex-col gap-10">
                <div className="flex flex-col gap-3 items-center text-center">
                    <h1 className="text-3xl font-semibold text-zinc-900 tracking-tight">
                        Fashion Search
                    </h1>
                    <p className="text-zinc-400 text-sm">
                        Search by text description or upload an image
                    </p>
                </div>

                <SearchBar onSearch={handleSearch} />

                {(loading || error || results) && (
                    <ResultsGrid
                        results={results}
                        loading={loading}
                        error={error}
                    />
                )}
            </div>
        </div>
    )
}