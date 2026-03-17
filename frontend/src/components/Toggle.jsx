export default function Toggle({ mode, onChange }) {
    return (
        <div className="flex items-center gap-1 bg-zinc-100 rounded-full p-1">
            <button
                onClick={() => onChange('text')}
                className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200 ${
                    mode === 'text'
                        ? 'bg-white text-zinc-900 shadow-sm'
                        : 'text-zinc-500 hover:text-zinc-700'
                }`}
            >
                Text
            </button>
            <button
                onClick={() => onChange('image')}
                className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-200 ${
                    mode === 'image'
                        ? 'bg-white text-zinc-900 shadow-sm'
                        : 'text-zinc-500 hover:text-zinc-700'
                }`}
            >
                Image
            </button>
        </div>
    )
}