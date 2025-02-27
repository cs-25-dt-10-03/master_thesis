from database.dataManager import fetch_all_offers

def main():
    result = fetch_all_offers()

    for sfo in result["sfo"]:
        print(sfo)

    for tec in result["tec"]:
        print(tec)
    

if __name__ == "__main__":
    main()
