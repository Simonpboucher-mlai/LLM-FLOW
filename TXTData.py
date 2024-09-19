class TXTData(ExternalData):
    async def load_data(self, session: aiohttp.ClientSession) -> str:
        if self.source.startswith(('http://', 'https://')):
            async with session.get(self.source) as response:
                response.raise_for_status()
                return await response.text()
        else:
            with open(self.source, 'r') as file:
                return file.read()
