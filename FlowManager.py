class FlowManager:
    def __init__(self, api_keys: Dict[str, str], words_per_chunk: int = 100, default_top_k: int = 3):
        self.steps: List[Step] = []
        self.api_client = APIClient(api_keys)
        self.semantic_search = SemanticSearch(self.api_client, words_per_chunk)
        self.default_top_k = default_top_k

    def add_step(self, step: Step):
        self.steps.append(step)

    async def process_step(self, step_index: int):
        if step_index == 0:
            for block in self.steps[step_index].blocks:
                if block.input_blocks:
                    logging.warning(f"Step 1, Block cannot have inputs from other blocks.")
                    return

        async with aiohttp.ClientSession() as session:
            tasks = []
            for block in self.steps[step_index].blocks:
                all_input_texts = self.collect_input_texts(step_index, block)
                
                external_data_text = await block.load_external_data(session)
                logging.info(f"External data for Step {step_index + 1}, Block {len(tasks) + 1}: {external_data_text[:100]}...")
                
                if block.semantic_search:
                    query = block.semantic_search.get('query', '')
                    top_k = block.semantic_search.get('top_k', self.default_top_k)
                    search_inputs = block.semantic_search.get('inputs', ['all'])
                    
                    relevant_chunks = []
                    
                    if 'all' in search_inputs or 'previous' in search_inputs:
                        input_search_text = '\n'.join(all_input_texts)
                        relevant_input_chunks = await self.semantic_search.search(session, query, input_search_text, top_k)
                        relevant_chunks.extend(relevant_input_chunks)
                    
                    if 'all' in search_inputs or 'external' in search_inputs:
                        if external_data_text:
                            relevant_external_chunks = await self.semantic_search.search(session, query, external_data_text, top_k)
                            relevant_chunks.extend(relevant_external_chunks)
                    
                    input_texts = [chunk for chunk, _ in relevant_chunks]
                    logging.info(f"Semantic search results: {[(chunk[:100], score) for chunk, score in relevant_chunks[:2]]}")
                else:
                    input_texts = all_input_texts + ([external_data_text] if external_data_text else [])

                if input_texts:
                    merged_input = "\n".join(input_texts)
                    block.prompt = f"{merged_input}\n\n{block.prompt}"

                logging.info(f"Processing Step {step_index + 1}, Block {len(tasks) + 1}")
                logging.info(f"Prompt: {block.prompt[:100]}...")

                task = self.api_client.generate_text(
                    session, block.model, block.prompt, block.temperature, block.max_tokens
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            for block, result in zip(self.steps[step_index].blocks, results):
                block.output = result
                block.save_block_output()

    def collect_input_texts(self, step_index: int, block: PromptBlock) -> List[str]:
        return [self.steps[input_step].blocks[input_block].output 
                for input_step, input_block in block.input_blocks 
                if input_step < step_index]

    async def run_flow(self):
        self.visualize_flow()
        for i in range(len(self.steps)):
            await self.process_step(i)
            self.display_step_results(i)

    def display_step_results(self, step_index: int):
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Step {step_index + 1} Results:{Style.RESET_ALL}")
        for j, block in enumerate(self.steps[step_index].blocks):
            print(f"{Fore.BLUE}{Style.BRIGHT}Block {j + 1} (Model: {block.model}, Max Tokens: {block.max_tokens}):{Style.RESET_ALL}")
            print(f"\n{Fore.YELLOW}Output:{Style.RESET_ALL} {block.output}\n")

    def visualize_flow(self):
        dot_code = self.generate_dot_code()
        graph = Source(dot_code)
        display(graph)

    def generate_dot_code(self) -> str:
        dot_code = """
        digraph G {
            node [style="filled", fontname="Arial", shape="box", margin="0.2,0.1"];
            edge [fontname="Arial"];
        """
        
        colors = ["#FFB3BA", "#BAFFC9", "#BAE1FF", "#FFFFBA", "#FFD8B9"]
        
        for i, step in enumerate(self.steps):
            step_color = colors[i % len(colors)]
            for j, block in enumerate(step.blocks):
                node_id = f"Block{i+1}_{j+1}"
                external_data_info = f"\\nExternal Data: {type(block.external_data).__name__}" if block.external_data else ""
                semantic_search_info = "\\nSemantic Search" if block.semantic_search else ""
                label = f"Step {i+1}, Block {j+1}\\nModel: {block.model}{external_data_info}{semantic_search_info}\\nPrompt: {block.prompt[:50]}..."
                dot_code += f'    {node_id} [label="{label}", fillcolor="{step_color}"];\n'
                for input_step, input_block in block.input_blocks:
                    input_id = f"Block{input_step+1}_{input_block+1}"
                    dot_code += f"    {input_id} -> {node_id};\n"
        
        dot_code += "}"
        return dot_code
