import PyPDF2

def read_pdf_file(filename):
    file_obj = open(filename, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(file_obj)
    print(pdf_reader.numPages)
    pageObj = pdf_reader.getPage(32)
    print(len(pageObj.extractText()))
