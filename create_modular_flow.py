def create_modular_flow(steps_config: List[Dict[str, Any]], api_keys: Dict[str, str], words_per_chunk: int = 100, default_top_k: int = 3) -> FlowManager:
    flow_manager = FlowManager(api_keys, words_per_chunk=words_per_chunk, default_top_k=default_top_k)
    
    for step_config in steps_config:
        step = Step()
        for block_config in step_config['blocks']:
            external_data = None
            if 'external_data' in block_config:
                data_type = block_config['external_data']['type']
                source = block_config['external_data']['source']
                if data_type == 'web':
                    external_data = WebData(source)
                elif data_type == 'api':
                    external_data = APIData(source)
                elif data_type == 'txt':
                    external_data = TXTData(source)
                elif data_type == 'csv':
                    external_data = CSVData(source)

            block = PromptBlock(
                prompt=block_config['prompt'],
                model=block_config.get('model', 'gpt-3.5-turbo'),
                max_tokens=block_config.get('max_tokens', 2000),
                temperature=block_config.get('temperature', 0.7),
                external_data=external_data,
                semantic_search=block_config.get('semantic_search'),
                save_output=block_config.get('save_output')
            )
            for input_ref in block_config.get('inputs', []):
                block.add_input(input_ref[0], input_ref[1])
            step.add_block(block)
        flow_manager.add_step(step)
    
    return flow_manager
