from languageDetectinator.datasets import Language, Vocabulary
import numpy as np

def outputVector(index: int, totalLangs: int) -> np.array:
    stringVec = str(0)*index + str(1) + str(0)*(totalLangs-1-index)
    return np.array([float(v) for v in stringVec])

language_tags = {

                'en':['actor', 'alcohol', 'cheque', 'cancer', 'chocolate', 'debate', 'hobby', 'melon', 'propaganda',
                      'religion', 'violin', 'england', 'MediaWiki'],

                'cs': ['praha', 'evropa', 'pyreneje', 'voda', 'housle', 'Náboženství', 'Příroda', 'Ekosystém',
                    'vzdělání', 'Irkso', 'Dům', 'Zpěvák', 'Zeus', 'Mykény', 'Starověké_Řecko', 'Renesance',
                    'Andrej_Babiš', 'Správa_železniční_dopravní_cesty', 'Kraje_v_Česku', 'Česko', 'Slezsko',
                    'Latina', 'Spojené_království', 'Římský_senát'],

                'de': ['Deutsche_Sprache', 'Deutschland', 'Kommunistische_Partei_der_Sowjetunion', 'Wasser',
                    'Festkörper', 'Seele', 'Geist', 'Dreifaltigkeit', 'Große', 'Christentum', 'Gott'],

                'sv': ['Svenska', 'Sverige', 'Danmark', 'Europeiska_unionen', 'Medeltiden', 'Feodalism', 'Kung',
                    'Kejsare', 'Monarki', 'Valmonarki', 'Choklad', 'Mjölk', 'Prolaktin', 'Kvinna', 'Eldvapen',
                    'Kina', 'Götar', 'Romantiken', 'Ideologi', 'Tänkande', 'Pedagogik', 'Sekund', 'Solen', 'Väder',
                    'Mellanöstern', 'Väte', 'Anatomi', 'Hjärta', 'Puls', 'Grekiska', 'Cypern'],

                'fr': ['Français', 'Langues_romanes', 'Charlemagne', 'Traité_de_Verdun', 'Louis_le_Pieux',
                    'Son_(physique)', 'Zoologie', 'Intelligence_animale', 'Intelligence', 'Tautologie',
                    'Pléonasme', 'Figure_de_style']

                 }

def main():
    i = 0
    vecs = []
    for key, value in language_tags.items():
        lang = Language(key)
        vocab = lang.generateVocabulary(topics=value)

        vocab.pruneVocabulary(12, duplicate=False)
        print(f"Found {len(vocab.words)} words.")
        vocabVec = vocab.vectorizeVocabulary(12)
        outVec = outputVector(i, len(language_tags)).reshape(1, len(language_tags))
        i += 1

        combVec = np.concatenate((vocabVec,np.repeat(outVec,vocabVec.shape[0],axis=0)),axis=1)
        vecs.append(combVec)

    vecs = np.vstack(vecs)
    np.save("./data/processed/vectors.npy", vecs)

if __name__ == "__main__":
    main()