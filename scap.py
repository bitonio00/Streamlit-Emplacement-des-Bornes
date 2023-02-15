import re
import pandas as pd
import PyPDF4

# Ouvre le fichier PDF en mode lecture binaire
with open('Liste-de-garage.pdf', 'rb') as pdf_file:
    # Crée un objet de type PyPDF2.PdfFileReader à partir du fichier PDF
    pdf_reader = PyPDF4.PdfFileReader(pdf_file)

    # Initialise la liste qui stockera les adresses de garage
    adresses_garages = []

    # Parcourt chaque page du document
    for page_num in range(pdf_reader.getNumPages()):
        # Récupère le contenu de la page en chaîne de caractères
        page_text = pdf_reader.getPage(page_num).extractText()

        # Utilise une expression régulière pour trouver les adresses de garage
        regex = r"[0-9]+ .+ \d{5} [A-Z][a-z]+(?: [A-Z][a-z]+)?(?: \d{5})?"
        adresses = re.findall(regex, page_text)

        # Ajoute les adresses de garage trouvées à la liste
        adresses_garages.extend(adresses)

    # Crée un DataFrame à partir de la liste d'adresses de garage
    df = pd.DataFrame(adresses_garages, columns=['adresse'])

    # Affiche le DataFrame
    print(df)