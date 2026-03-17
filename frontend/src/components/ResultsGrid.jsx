import ResultCard from './ResultCard'

export default function ResultsGrid({ results, loading, error }) {
    if (loading) {
        return (
            <div className="w-full grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                {Array.from({ length: 10 }).map((_, i) => (
                    <div key={i} className="bg-zinc-100 rounded-2xl overflow-hidden animate-pulse">
                        <div className="aspect-square bg-zinc-200" />
                        <div className="p-4 flex flex-col gap-2">
                            <div className="h-4 bg-zinc-200 rounded w-3/4" />
                            <div className="h-3 bg-zinc-200 rounded w-1/2" />
                        </div>
                    </div>
                ))}
            </div>
        )
    }

    if (error) {
        return (
            <div className="w-full flex items-center justify-center py-20">
                <p className="text-zinc-400 text-sm">{error}</p>
            </div>
        )
    }

    if (!results || results.length === 0) {
        return (
            <div className="w-full flex items-center justify-center py-20">
                <p className="text-zinc-400 text-sm">No results found</p>
            </div>
        )
    }

    return (
        <div className="w-full grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
            {results.map(result => (
                <ResultCard key={result.article_id} result={result} />
            ))}
        </div>
    )
}