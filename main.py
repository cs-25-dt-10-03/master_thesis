from time import time
from database.dataManager import fetchEvData


def main():
    start = time()
    fetchEvData()
    end = time()
    print(end-start)


if __name__ == "__main__":
    main()
