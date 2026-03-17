import axios from 'axios'

export async function searchByText(query, limit = 10) {
    const response = await axios.post('/search/text', {
        query,
        limit
    })
    return response.data
}

export async function searchByImage(file, limit = 10) {
    const formData = new FormData()
    formData.append('file', file)

    const response = await axios.post(`/search/image?limit=${limit}`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data'
        }
    })
    return response.data
}