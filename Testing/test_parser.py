from database.parser import fo_parser


def test_foParser():
    result = fo_parser()
    for element in result:
        print(element)
