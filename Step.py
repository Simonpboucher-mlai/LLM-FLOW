class Step:
    def __init__(self):
        self.blocks: List[PromptBlock] = []

    def add_block(self, block: PromptBlock):
        self.blocks.append(block)
