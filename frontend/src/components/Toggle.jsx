export default function Toggle({ fusion, onChange }) {
    return (
        <div className="flex items-center gap-2">
            <button
                type="button"
                onClick={() => onChange(fusion === 'weighted' ? 'rrf' : 'weighted')}
                className={`relative rounded-full transition-colors duration-300 ${
                    fusion === 'weighted' ? 'bg-violet-600' : 'bg-indigo-600'
                }`}
                style={{ width: '36px', height: '20px' }}
            >
                <span
                    className="absolute bg-white rounded-full shadow transition-transform duration-300"
                    style={{
                        width: '14px',
                        height: '14px',
                        top: '3px',
                        left: '3px',
                        transform: fusion === 'weighted' ? 'translateX(0px)' : 'translateX(16px)',
                    }}
                />
            </button>
            <span className={`text-xs w-14 text-left font-medium ${
                fusion === 'weighted' ? 'text-violet-500' : 'text-indigo-600'
            }`}>
                {fusion === 'weighted' ? 'Weighted' : 'RRF'}
            </span>
        </div>
    )
}