class CSVData(ExternalData):
    async def load_data(self, session: aiohttp.ClientSession) -> str:
        if self.source.startswith(('http://', 'https://')):
            async with session.get(self.source) as response:
                response.raise_for_status()
                content = await response.text()
        else:
            with open(self.source, 'r') as file:
                content = file.read()

        csv_data = []
        csv_reader = csv.reader(io.StringIO(content))
        for row in csv_reader:
            csv_data.append(row)
        return json.dumps(csv_data, indent=2)
