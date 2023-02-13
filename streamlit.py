# --------------------------------------------------- Import des bibliothèques --------------------------------------------------- #


import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from scipy.spatial import cKDTree
from geopy.distance import great_circle
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------- Configuration de la page streamlit --------------------------------------------------- #


st.set_page_config(layout="wide", page_title="Proxigital", page_icon=":house:")

def home():
    st.markdown("#Home")
    st.sidebar.markdown("#Home")

def page2():
    st.markdown("#Tops 10")
    st.sidebar.markdown("#Tops 10")



try:

# --------------------------------------------------- Fonctions d'import et de préparation des dataframes --------------------------------------------------- #

    def potentiel_pdv(bornes, address): # algo permettant de compter le nombre de points de vente dans un rayon de 10 km (à voir s'il faut aggrandir le rayon)
        df_potentiel_pdv = pd.DataFrame(columns=['adresseLatitude', 'adresseLongitude', 'commune', 'Région'])
        pop_commune = pd.read_excel('pop commune.xlsx')
        df_gares = pd.read_csv("liste-des-gares.csv", delimiter =";") # import de la liste des gares
        df_densite = pd.read_excel('FET2021-19.xlsx')
        df_niveau_de_vie = pd.read_excel('communes_niveau_de_vie.xlsx')
        df_gares.drop(columns=['code_uic', 'fret', 'voyageurs', 'code_ligne', 'rg_troncon', 'pk', 'idgaia', 'x_l93', 'y_l93', 'c_geo',
                     'geo_point_2d', 'geo_shape', 'idreseau'], inplace=True) # je supprime les colonnes qui ne me serviront pas
        geolocator = Nominatim(user_agent="geoapiExercises")
        location = geolocator.geocode(address, addressdetails=True, country_codes='FR')
        df_potentiel_pdv = df_potentiel_pdv.append({'adresseLongitude': location.longitude,
                                                    'adresseLatitude': location.latitude,
                                                    'Région': location.raw.get('address').get('state'),
                                                    'commune': location.raw.get('address').get('city')}, ignore_index=True)
        store_coord = location.point
        df_nearby_stores = pd.DataFrame(columns=['adresseLatitude', 'adresseLongitude'])
        nearby_stores = 0
        for i, row in bornes.iterrows():
            if distance_calc(store_coord, (row['adresseLatitude'], row['adresseLongitude'])) <= 10:
                nearby_stores += 1
                df_nearby_stores = df_nearby_stores.append({'adresseLatitude':row['adresseLatitude'],
                                                            'adresseLongitude':row['adresseLongitude']}, ignore_index=True)

        cartographie_pdv(nearby_stores, df_potentiel_pdv, df_nearby_stores)
        df_potentiel_pdv['gareLaPlusProche'], df_potentiel_pdv['distanceGare'] = zip(
            *df_potentiel_pdv.apply(closest_station, axis=1, stations=df_gares))

        df_potentiel_pdv = df_potentiel_pdv.merge(df_densite, on='commune', how='left')
        df_potentiel_pdv = df_potentiel_pdv.merge(df_niveau_de_vie, on='commune', how='left')
        df_potentiel_pdv['densite'] = df_potentiel_pdv['densite'].apply(str_to_densite) # traduction de la densité en score
        df_potentiel_pdv = df_potentiel_pdv.merge(pop_commune, on='commune', how='left')


        st.write("Il y a", nearby_stores, "point(s) de vente dans un rayon de 10 kilomètres")
        st.write("Les données du potentiel point de vente sont stockées ici : ", df_potentiel_pdv)

        return nearby_stores, df_potentiel_pdv

    @st.cache
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
        return merged_data # je retourne le dataframe complet et prêt pour ne plus y toucher par la suite (mise en cache pour executer qu'une seule fois)


    # --------------------------------------------------- Fonctions de texte et d'affichage --------------------------------------------------- #


    def title(): # simple affichage des titres de la page
        st.header("Analyse d'un potentiel point de vente pour Proxigital.")
        st.subheader("Saisissez l'adresse d'un potentiel point de vente pour afficher son score :")

    def print_map(data): # affichage de carte
        data_map = data.copy()
        data_map.rename(columns={'adresseLongitude':'lon','adresseLatitude':'lat'}, inplace=True)
        st.map(data_map)


    # --------------------------------------------------- Fonctions de traduction de la donnée --------------------------------------------------- #


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

    def str_to_densite(code): # transforme le str de densité en score
        densite = {
            'Communes très peu denses': 1,
            'Communes peu denses': 2,
            'Communes de densité intermédiaire': 3,
            'Communes densément peuplées': 4
        }
        return densite.get(code, "Région non définie")


    # --------------------------------------------------- Fonctions techniques --------------------------------------------------- #


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

    def distance_calc(point1, point2): # simple algo de calcul de distance
        return geodesic(point1, point2).km

    def cartographie_pdv(nearby_stores, df_potentiel_pdv, df_nearby_stores): # préparation à l'appel de la fonction print_map
        if nearby_stores >= 1:
                l_col, r_col = st.columns(2)
                with l_col:
                    st.subheader('Cartographie du point de vente : ')
                    print_map(df_potentiel_pdv)
                with r_col:
                    st.subheader('Cartographie des points de ventes dans un rayon de 10 km : ')
                    print_map(df_nearby_stores)
        else:
            st.subheader('Cartographie du point de vente (aucun point de vente dans un rayon de 10 km) : ')
            print_map(df_potentiel_pdv)


    # --------------------------------------------------- Machine Learning --------------------------------------------------- #

    @st.cache
    def prediction(df):
        # Transformation des données
        le = LabelEncoder()
        df["region"] = le.fit_transform(df["region"])
        df["commune"] = le.fit_transform(df["commune"])
        df["gareLaPlusProche"] = le.fit_transform(df["gareLaPlusProche"])

        # Normalisation des données
        scaler = MinMaxScaler()
        df[["adresseLongitude", "adresseLatitude", "populationCommune", "densite", "niveau_de_vie", "distanceGare",
            "nombre_paiements", "nombre_contacts", "taux de transformation"]] = scaler.fit_transform(df[["adresseLongitude",
                                                                                                         "adresseLatitude",
                                                                                                         "populationCommune",
                                                                                                         "densite",
                                                                                                         "niveau_de_vie",
                                                                                                         "distanceGare",
                                                                                                         "nombre_paiements",
                                                                                                         "nombre_contacts",
                                                                                                         "taux de transformation"]])

        # Normalisation des données
        scaler = MinMaxScaler()
        df[["adresseLongitude", "adresseLatitude", "populationCommune", "densite", "niveau_de_vie", "distanceGare",
            "nombre_paiements", "nombre_contacts", "taux de transformation"]] = scaler.fit_transform(df[["adresseLongitude",
                                                                                                         "adresseLatitude",
                                                                                                         "populationCommune",
                                                                                                         "densite",
                                                                                                         "niveau_de_vie",
                                                                                                         "distanceGare",
                                                                                                         "nombre_paiements",
                                                                                                         "nombre_contacts",
                                                                                                         "taux de transformation"]])

        df = df.drop(["code_barre"], axis=1)
        # Séparation des données en ensembles d'entraînement et de test
        X = df[['adresseLongitude', 'adresseLatitude', 'code_postal', 'region', 'commune', 'populationCommune', 'densite',
                'niveau_de_vie', 'gareLaPlusProche', 'distanceGare', 'nombre_paiements', 'nombre_contacts']]
        y = df['taux de transformation']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        classifier = DecisionTreeClassifier(random_state=0)
        classifier.fit(X_train, y_train)


    # --------------------------------------------------- Main --------------------------------------------------- #


    def main(): # appel des fonctions nécessaires

        title()
        adress = st.text_input(" ", placeholder="23-25 rue Chaptal 75009 Paris")
        bornes = get_dataframe()
        nearby_stores, df_potentiel_pdv = potentiel_pdv(bornes, adress)
        bornes

    main()

except:
    pass