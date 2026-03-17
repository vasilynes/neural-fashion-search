export default function ResultCard({ result }) {
    const { article_id, caption, colour_group_name, product_type_name, score, image_path } = result

    return (
        <div className="bg-white rounded-2xl overflow-hidden border border-zinc-100 hover:shadow-md transition-shadow duration-200">
            <div className="aspect-square bg-zinc-50 overflow-hidden">
                <img
                    src={`/images/${article_id}`}
                    alt={caption}
                    className="w-full h-full object-cover"
                    onError={e => {
                        e.target.style.display = 'none'
                    }}
                />
            </div>
            <div className="p-4 flex flex-col gap-2">
                <div className="flex items-start justify-between gap-2">
                    <p className="text-sm text-zinc-900 font-medium leading-snug line-clamp-2">
                        {caption}
                    </p>
                    <span className="text-xs text-zinc-400 shrink-0 font-mono">
                        {score.toFixed(3)}
                    </span>
                </div>
                <div className="flex gap-2 flex-wrap">
                    {colour_group_name && (
                        <span className="text-xs px-2 py-0.5 bg-zinc-100 text-zinc-600 rounded-full">
                            {colour_group_name}
                        </span>
                    )}
                    {product_type_name && (
                        <span className="text-xs px-2 py-0.5 bg-zinc-100 text-zinc-600 rounded-full">
                            {product_type_name}
                        </span>
                    )}
                </div>
                <p className="text-xs text-zinc-400 font-mono">
                    {article_id}
                </p>
            </div>
        </div>
    )
}