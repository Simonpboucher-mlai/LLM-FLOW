class PromptBlock:
    def __init__(self, prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 2000, 
                 temperature: float = 0.7, external_data: Optional[ExternalData] = None,
                 semantic_search: Optional[Dict[str, Any]] = None, 
                 save_output: Optional[Dict[str, str]] = None):
        self.prompt = prompt
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.output = None
        self.input_blocks: List[Tuple[int, int]] = []
        self.external_data = external_data
        self.semantic_search = semantic_search
        self.save_output = save_output

    def add_input(self, step_index: int, block_index: int):
        self.input_blocks.append((step_index, block_index))

    async def load_external_data(self, session: aiohttp.ClientSession) -> str:
        if not self.external_data:
            return ""
        try:
            return await self.external_data.load_data(session)
        except Exception as e:
            logging.error(f"Error loading external data from {self.external_data.source}: {str(e)}")
            return f"Error loading external data: {str(e)}"

    def save_block_output(self):
        if self.save_output and self.output:
            output_format = self.save_output.get('format', 'txt').lower()
            filename = self.save_output.get('filename', f'output_{id(self)}.{output_format}')
            
            if output_format == 'txt':
                OutputSaver.save_txt(self.output, filename)
            elif output_format == 'pdf':
                OutputSaver.save_pdf(self.output, filename)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            print(f"Output saved to {filename}")
