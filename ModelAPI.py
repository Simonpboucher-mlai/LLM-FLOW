class ModelAPI(ABC):
    @abstractmethod
    async def generate_text(self, session: aiohttp.ClientSession, model: str, prompt: str, 
                            temperature: float, max_tokens: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def extract_text_from_response(self, response: Dict[str, Any]) -> str:
        pass

class OpenAIAPI(ModelAPI):
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def generate_text(self, session: aiohttp.ClientSession, model: str, prompt: str, 
                            temperature: float, max_tokens: int) -> Dict[str, Any]:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                raise APIException(f"OpenAI API error: {await response.text()}")
            return await response.json()

    def extract_text_from_response(self, response: Dict[str, Any]) -> str:
        return response["choices"][0]["message"]["content"].strip()

    async def get_embeddings(self, session: aiohttp.ClientSession, texts: List[str]) -> List[List[float]]:
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": "text-embedding-ada-002",
            "input": texts
        }
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                raise APIException(f"OpenAI Embedding API error: {await response.text()}")
            result = await response.json()
            return [item['embedding'] for item in result['data']]
