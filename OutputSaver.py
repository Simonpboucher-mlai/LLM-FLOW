class OutputSaver:
    @staticmethod
    def save_txt(content: str, filename: str):
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)

    @staticmethod
    def save_pdf(content: str, filename: str):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, content)
        pdf.output(filename)
