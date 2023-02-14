# --------------------------------------------------- Import des bibliothèques --------------------------------------------------- #


import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
from scipy.spatial import cKDTree
from geopy.distance import great_circle


# --------------------------------------------------- Configuration de la page streamlit --------------------------------------------------- #


st.set_page_config(layout="wide", page_title="Proxigital", page_icon=":house:")

st.markdown("Tops 10")


# --------------------------------------------------- Fonctions de texte et d'affichage --------------------------------------------------- #


def title():
    st.header("Analyse d'un potentiel point de vente pour Proxigital.")
    st.subheader("Cette page regroupe le top 10 des bornes sur l'année 2022 selon plusieurs critères.")

def code_to_region(code): # traduction des codes région en région normale (merci chatgpt)
    regions = {
        11: "Ile-de-France",
        24: "Centre-Val de Loire",
        27: "Bourgogne-Franche-Comté",
        28: "Normandie",
        32: "Hauts-de-France",
        44: "Grand Est",
        52: "Pays de la Loire",
        53: "Bretagne",
        75: "Nouvelle-Aquitaine",
        76: "Occitanie",
        84: "Auvergne-Rhône-Alpes",
        93: "Provence-Alpes-Côte d'Azur",
        94: "Corse",
    }
    return regions.get(code, "Région non définie")

def closest_station(row, stations):
    stations = stations.dropna(subset=['x_wgs84', 'y_wgs84'])  # Remove rows with missing values
    station_locations = stations[['y_wgs84', 'x_wgs84']].values
    station_tree = cKDTree(station_locations)
    _, closest_index = station_tree.query([row['adresseLatitude'], row['adresseLongitude']])
    closest_station = stations.iloc[closest_index]
    station_location = (closest_station['y_wgs84'], closest_station['x_wgs84'])
    address_location = (row['adresseLatitude'], row['adresseLongitude'])
    distance = great_circle(address_location, station_location).km
    return closest_station['libelle'], distance

def str_to_densite(code): # transforme le str de densité en score
    densite = {
        'Communes très peu denses': 1,
        'Communes peu denses': 2,
        'Communes de densité intermédiaire': 3,
        'Communes densément peuplées': 4
    }
    return densite.get(code, "Région non définie")


@st.cache(allow_output_mutation=True)
def get_dataframe():
    transactions = pd.read_csv("transactions.csv", delimiter=";") # import du dataframe transactions 2022
    transactions.drop(columns=['heure', 'siret', 'representant_legal', 'partenaire', 'facture_int'], inplace=True) # je supprime les colonnes qui ne me serviront pas
    first_column = transactions.pop("code_barre") # je déplace la colonne code_barre en premier pour mieux la voir (la plus importante)
    transactions.insert(0, 'code_barre', first_column)

    df_gares = pd.read_csv("liste-des-gares.csv", delimiter =";") # import de la liste des gares
    df_gares.drop(columns=['code_uic', 'fret', 'voyageurs', 'code_ligne', 'rg_troncon', 'pk', 'idgaia', 'x_l93', 'y_l93', 'c_geo',
                 'geo_point_2d', 'geo_shape', 'idreseau'], inplace=True) # je supprime les colonnes qui ne me serviront pas

    bornes = pd.read_excel("20230130 - Organisations.xlsx") # import de la base organisations à jour du 1er février 2023
    bornes.drop(bornes[(bornes['borneInstallee'] == 'Non')].index, inplace=True) # je supprime les bornes pas installées (à voir plus tard s'il faut les laisser pour améliorer l'algorithme de machine learning
    bornes.drop(
        columns=['idOrganisation', 'dateEnregistrement', 'siret', 'raisonSociale', 'codeDouanes', 'adresseLigne2',
                 'adresseLigne3', 'representantLegalNom', 'representantLegalMobile', 'representantLegalEmail',
                 'representantLegalCiv', 'representantLegalPrenom', 'aEuUneBorne', 'exDebutBorneOrga', 'exFinBorneOrga',
                 'borneVersion', 'borneVersionNom', 'dateLiaisonBorneOrga', 'datePremiereInstallBorne',
                 'demandeInstallationStatut', 'demandeInstallationDateStatut', 'demandeInstallationDatePlannif',
                 'servicedonsADOSSPP', 'servicecataloguePDFUssac', 'servicecontactFormAIBImmo',
                 'servicecontactFormIADCourbot', 'servicecontactFormIADMagne', 'servicecontactFormSAFIDamien',
                 'servicehorairesPDFCassis', 'servicecontactFormCAPI', 'servicemeilleurTaux', 'serviceCarteGriseFI',
                 'borneInstallee', 'nomCommercial', 'adresseCodePostal', 'adresseCommune'], inplace=True) # drop des colonnes inutiles
    first_column = bornes.pop('borneCodeBarre') # déplacement de code_barre en premier
    bornes.insert(0, 'code_barre', first_column)
    bornes.dropna(subset=['adresseLatitude', 'adresseLongitude'], inplace=True) # suppression des bornes qui n'ont pas de coordoonnées gps car font planter l'algo geopy

    merged_data = pd.merge(transactions, bornes, left_on='code_barre', right_on='code_barre') # merge des bases bornes et transactions
    merged_data = merged_data[['code_barre', 'adresseLongitude', 'adresseLatitude', 'adresseLigne1', 'code_postal', 'commune', 'populationCommune', 'statut_paiement', 'adresseCodeRegion']] # je garde uniquement les colonnes qui m'intéressent
    merged_data['region'] = merged_data['adresseCodeRegion'].apply(code_to_region) # traduction du code région en région pleine (str)

    merged_data['gareLaPlusProche'], merged_data['distanceGare'] = zip(
        *merged_data.apply(closest_station, axis=1, stations=df_gares))

    merged_data['nombre_paiements'] = merged_data[merged_data['statut_paiement'] == 'Payé'].groupby('code_barre')['statut_paiement'].transform('count') # assigne à chaque borne son nombre de contacts et son nombre de ventes
    df_grouped = merged_data.groupby('code_barre').size().reset_index(name='nombre_contacts')
    merged_data = merged_data.merge(df_grouped, on='code_barre', how='left') # je les regroupe par code barre
    df_densite = pd.read_excel('FET2021-19.xlsx')
    df_niveau_de_vie = pd.read_excel('communes_niveau_de_vie.xlsx')
    merged_data = merged_data.merge(df_densite, on='commune', how='left')
    merged_data['densite'] = merged_data['densite'].apply(str_to_densite) # traduction de la densité en score
    merged_data = merged_data.merge(df_niveau_de_vie, on='commune', how='left')
    merged_data = merged_data.groupby('code_barre').agg({'adresseLongitude': 'first',
                                            'adresseLatitude': 'first',
                                            'adresseLigne1': 'first',
                                            'code_postal': 'first',
                                            'region': 'first',
                                            'commune':'first',
                                            'populationCommune':'first',
                                            'densite':'first',
                                            'niveau_de_vie':'first',
                                            'gareLaPlusProche':'first',
                                            'distanceGare': 'first',
                                            'nombre_paiements': 'first',
                                            'nombre_contacts': 'first'}) # je remets dans l'ordre les colonnes
    merged_data ['taux de transformation'] = (merged_data['nombre_paiements'] / merged_data['nombre_contacts']) * 100 # rajout de la colonne taux de transformation
    return merged_data # je retourne le dataframe complet et prêt pour ne plus y toucher par la suite

def tops_10_contacts(df):
    df_top_10_contacts = df.sort_values(by='nombre_contacts', ascending=False).head(10)
    mean_contacts = df_top_10_contacts.mean()
    return df_top_10_contacts, mean_contacts

def tops_10_paiements(df):
    df_top_10_paiements = df.sort_values(by='nombre_paiements', ascending=False).head(10)
    mean_TT = df_top_10_paiements.mean()
    return df_top_10_paiements, mean_TT

def tops_10_TT(df):
    df_top_10_TT = df.sort_values(by='taux de transformation', ascending=False).head(10)
    mean_TT = df_top_10_TT.mean()
    return df_top_10_TT, mean_TT

def main():
    title()
    bornes = get_dataframe()
    top_10_contacts, mean_contacts = tops_10_contacts(bornes)
    top_10_paiements, means_paiements = tops_10_paiements(bornes)
    top_10_TT, means_TT = tops_10_TT(bornes)
    l_col, r_col = st.columns(2)
    with l_col:
        st.write('Top 10 selon le nombre de contacts :')
        st.write(top_10_contacts)
    with r_col:
        st.write('Top 10 selon le nombre de paiements :')
        st.write(top_10_paiements)
    st.write('Top 10 selon le taux de transformation :')
    st.write(top_10_TT)

main()