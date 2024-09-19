class APIData(ExternalData):
    async def load_data(self, session: aiohttp.ClientSession) -> str:
        async with session.get(self.source) as response:
            response.raise_for_status()
            data = await response.json()
            return json.dumps(data, indent=2)

