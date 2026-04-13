export default function WeightSlider({ label, value, onChange, step = 0.1}) {
    const decimals = step.toString().includes('.') ? step.toString().split('.')[1].length : 0
    return (
        <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-400 whitespace-nowrap">{label}: {value.toFixed(decimals)}</span>
            <input
                type="range"
                min="0"
                max="1"
                step={step}
                value={value}
                onChange={e => onChange(parseFloat(e.target.value))}
                className="w-24 slider-thin"
            />
        </div>
    )
}