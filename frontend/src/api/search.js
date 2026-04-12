import axios from 'axios'

export async function searchByText(query, limit = 10) {
    const response = await axios.post('/search/text', {
        query,
        limit
    })
    return response.data
}

export async function searchByImage(file, query, beta, limit = 10) {
    const formData = new FormData()
    formData.append('file', file)

    const params = new URLSearchParams({ limit })
    if (query) {
        params.append('query', query)
        params.append('beta', beta)
    }
    
    const response = await axios.post(`/search/image?${params}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    })
    return response.data
}