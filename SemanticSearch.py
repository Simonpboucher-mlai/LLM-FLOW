class SemanticSearch:
    def __init__(self, api_client: APIClient, words_per_chunk: int = 100):
        self.api_client = api_client
        self.words_per_chunk = words_per_chunk

    @staticmethod
    def split_text(text: str, words_per_chunk: int) -> List[str]:
        words = text.split()
        return [' '.join(words[i:i+words_per_chunk]) for i in range(0, len(words), words_per_chunk)]

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    async def search(self, session: aiohttp.ClientSession, query: str, text: str, top_k: int) -> List[Tuple[str, float]]:
        chunks = self.split_text(text, self.words_per_chunk)
        query_embedding = await self.api_client.get_embeddings(session, [query])
        chunk_embeddings = await self.api_client.get_embeddings(session, chunks)
        
        similarities = [self.cosine_similarity(query_embedding[0], chunk_emb) for chunk_emb in chunk_embeddings]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(chunks[i], similarities[i]) for i in top_indices]

