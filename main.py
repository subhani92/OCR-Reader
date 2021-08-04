import click
import cv2
import pdf2image
import pytesseract as pt
import logging
from data_utils import *



logger = logging.getLogger("test.py")
logger.setLevel(logging.DEBUG)
class ocr_reader:

    def __init__(self, input_file, output_file, text=""):
        self.input_file = input_file
        self.output_file = output_file
        self.text = text

    def ocr_utils(self): 
        image = cv2.imread(self.input_file)
        image = preprocess(image, self.input_file) 
        text = pt.image_to_string(image)
        return text

    def pdf_to_img(self):
        return pdf2image.convert_from_path(self.input_file)

    def pdf_pages(self):
        images = self.pdf_to_img()
        for pg, img in enumerate(images):
            img = preprocess(np.array(img), self.input_file)
            return pt.image_to_string(img)
            
    def write_file(self, text):
        text = postprocess(text)
        with open(self.output_file, 'a') as out:
            out.write(self.text)

@click.command()
@click.option('--input', type=str)
@click.option('--output', type=str)
@click.option('--verbose', '--verbose', count=True)

def main(input, output, verbose):
    #click.echo(input)
    if input.endswith("png") or input.endswith("jpeg") or input.endswith("jpg"):
        text = ocr_reader(input,output).ocr_utils()
        ocr_reader(input,output,text).write_file(text)

    elif input.endswith("pdf"):
        #images = ocr_reader(input,output).pdf_to_img()
        text = ocr_reader(input,output).pdf_pages()
        ocr_reader(input,output,text).write_file(text)
    
    else:
        logger.error("File type not supported!")
    click.echo('Ocr reading finished! output is saved in specified filename')

if __name__ == '__main__':
    logger.info("Ocr reading started !")
    main()
    ##logger.info("Ocr reading finished! output is saved in specified filename.")
   
   
        