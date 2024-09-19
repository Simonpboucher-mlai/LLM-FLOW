class WebData(ExternalData):
    async def load_data(self, session: aiohttp.ClientSession) -> str:
        async with session.get(self.source) as response:
            response.raise_for_status()
            text = await response.text()
            soup = BeautifulSoup(text, 'html.parser')
            
            all_text = soup.get_text(separator='\n', strip=True)
            all_text = re.sub(r'<[^>]+>', '', all_text)
            all_text = re.sub(r'\s+', ' ', all_text).strip()
            
            links = [a.get('href') for a in soup.find_all('a', href=True)]
            
            tables = []
            for table in soup.find_all('table'):
                table_data = []
                for row in table.find_all('tr'):
                    row_data = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
                    table_data.append(row_data)
                tables.append(table_data)
            
            combined_data = {
                "text_content": all_text,
                "links": links,
                "tables": tables
            }
            
            return json.dumps(combined_data, indent=2)
