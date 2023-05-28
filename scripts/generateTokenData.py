from languageDetectinator.datasets import Language

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
    for key, value in language_tags.items():
        lang = Language(key)
        vocab = lang.generateVocabulary(topics=value, decodeLang=False)

        with open(f"data/raw/{key}_full.txt","w",encoding="utf-8") as txtFile:
            txtFile.write(vocab.text)

        vocab.pruneVocabulary(12, duplicate=False, keepAccents=True)
        print(f"Found {len(vocab.words)} words.")
        
        with open(f"data/processed/{key}_pruned.txt","w",encoding="utf-8") as txtFile:
            txtFile.write(" ".join(vocab.words))

if __name__ == "__main__":
    main()